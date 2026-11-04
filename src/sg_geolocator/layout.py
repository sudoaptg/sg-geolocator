"""2D layout search for drone and other high oblique views.

A drone photo shows the layout of an estate: here, rows of long slab blocks lying parallel,
one behind another, with forest further away. Those patterns need only footprints, so the
whole island can be filtered in seconds before any rendering. The places that fit are then
ranked by render-and-compare from drone heights (camera 60-150 m up, standing back from the
nearest slab and looking along the stack).
"""

from __future__ import annotations

import numpy as np

from . import raster as R
from .plan2d import _bearing, to_latlon


def slab_blocks(plan: "R.Plan", buildings=None, min_height: float = 30.0, min_len: float = 50.0):
    """(centre, unit long axis, length) of long slab blocks: footprint 600 m2+, at least 3
    times as long as wide, at least `min_height` tall."""
    heights = R.building_heights(plan.data, buildings)
    out = []
    for item, h in zip(plan.data["buildings"], heights):
        if h < min_height:
            continue
        xy = R.to_xy(*np.asarray(item["pts"], float).T)
        if len(xy) < 4 or R._area(xy) < 600:
            continue
        q = xy - xy.mean(0)
        w, v = np.linalg.eigh(q.T @ q / len(q))
        if w[1] < 9 * max(w[0], 1e-6):
            continue
        length = 2 * np.sqrt(3 * w[1])
        if length >= min_len:
            out.append((xy.mean(0), v[:, 1] / np.linalg.norm(v[:, 1]), float(length)))
    return out


def stacks(slabs, min_count: int = 4, gap=(25.0, 70.0), max_angle: float = 12.0):
    """Rows of parallel slabs one behind another (centres `gap` apart across their long axis).
    Returns (first slab index, chain of indices, unit direction from the first to the last)."""
    if not slabs:
        return []
    c = np.array([s[0] for s in slabs])
    u = np.array([s[1] for s in slabs])
    L = np.array([s[2] for s in slabs])
    cos_max = np.cos(np.radians(max_angle))
    found = []
    for i in range(len(slabs)):
        best = None
        for sign in (1, -1):
            chain, cur = [i], i
            while len(chain) <= 8:
                n = np.array([-u[cur][1], u[cur][0]]) * sign
                d = c - c[cur]
                across, along = d @ n, np.abs(d @ u[cur])
                ok = np.flatnonzero((np.abs(u @ u[cur]) > cos_max) & (across > gap[0]) & (across < gap[1])
                                    & (along < 0.5 * L[cur]))
                if not len(ok):
                    break
                cur = ok[np.argmin(across[ok])]
                chain.append(cur)
            if best is None or len(chain) > len(best[0]):
                best = (chain, np.array([-u[chain[-1]][1], u[chain[-1]][0]]) * sign)
        if len(best[0]) >= min_count:
            found.append((i, best[0], best[1]))
    return found


def forest_beyond(plan: "R.Plan", at, direction, near: float = 500.0, far: float = 2500.0, width: float = 800.0,
                  min_area: float = 1_000_000.0) -> bool:
    """Is there a big green area (1 km2+) ahead of `at` in `direction`?"""
    for p in plan._pts.get("green", []):
        if len(p) < 4 or R._area(p) < min_area:
            continue
        d = p[::5] - at
        t, side = d @ direction, np.abs(d @ np.array([direction[1], -direction[0]]))
        if ((t > near) & (t < far) & (side < width)).any():
            return True
    return False


def candidate_views(plan: "R.Plan", buildings=None, min_slabs: int = 4, forest: bool = True,
                    dedupe_m: float = 300.0):
    """Distinct (lat, lon, compass direction) where a drone could see `min_slabs` stacked slabs
    (and, with `forest`, a big green area beyond them)."""
    slabs = slab_blocks(plan, buildings)
    out, seen = [], []
    for i0, chain, n in sorted(stacks(slabs, min_slabs), key=lambda s: -len(s[1])):
        far = slabs[chain[-1]][0]
        if forest and not forest_beyond(plan, far, n):
            continue
        c = slabs[i0][0]
        if any(np.hypot(*(c - s)) < dedupe_m for s in seen):
            continue
        seen.append(c)
        la, lo = to_latlon(c)
        out.append((la, lo, float(_bearing(n))))
    return out


def drone_search(plan: "R.Plan", labels: np.ndarray, buildings=None, min_slabs: int = 4, forest: bool = True,
                 top: int = 5, log=None) -> list[dict]:
    """Layout filter, then render-and-compare from drone heights at every place that fits."""
    views = candidate_views(plan, buildings, min_slabs, forest)
    if log:
        log(f"  {len(views)} places with {min_slabs}+ stacked slabs" + (" and forest beyond" if forest else ""))
    out = []
    for la, lo, view in views:
        pr = R.PlanRaster(plan.near(la, lo, 2500), la, lo, 2500, buildings)
        c = R.to_xy(la, lo)
        u = np.array([np.sin(np.radians(view)), np.cos(np.radians(view))])
        n = np.array([u[1], -u[0]])
        spots = [c - u * back + n * side for back in (100, 200, 300) for side in (-100, 0, 100)]
        ag, cam, hd, hf, eye, hz = R.search(pr, labels, spots, lambda cm, h: list(np.arange(view - 20, view + 21, 5.0)),
                                            [55.0, 70.0], [60.0, 90.0, 120.0, 150.0], np.arange(-0.10, 0.21, 0.05),
                                            keep=12)
        a, b = to_latlon(cam)
        out.append({"lat": round(a, 6), "lon": round(b, 6), "heading": round(hd, 1), "hfov": hf, "eye_m": eye,
                    "horizon": round(hz, 3), "agreement": round(ag, 4), "looking_at": [round(la, 6), round(lo, 6)],
                    "map": f"https://www.google.com/maps?q={a:.6f},{b:.6f}"})
    return sorted(out, key=lambda o: -o["agreement"])[:top]
