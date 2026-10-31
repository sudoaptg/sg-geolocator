"""Find where a photo was taken inside an area narrowed down by hints.

Hints are things a person sees in the photo or knows about it: the bus service number on a
bus, an MRT line or an elevated viaduct in view, the rough area. They turn into candidate
spots (along the bus route, near the track, inside the area); every spot is then scored by
how well the surrounding buildings match the photo: the sky outline over every heading, plus,
for daytime photos, how well the 3D view of built vs open areas overlaps. The best few
distinct spots are returned.
"""

from __future__ import annotations

import numpy as np

from .skyline import (COS_LAT, M_PER_DEG, MASK_WEIGHT, Buildings, match, photo_scene, photo_skyline,
                      render, render_built, view_overlap)

EYE_HEIGHTS = {"street": (1.6,), "bus": (4.2,), "window": (15.0, 30.0, 50.0)}


def _densify(lines, step_m: float) -> np.ndarray:
    out = []
    for pts in lines:
        p = np.asarray(pts, dtype=float)
        if len(p) == 1:
            out.append(p[0])
        for a, b in zip(p[:-1], p[1:]):
            d = np.hypot((b[0] - a[0]) * M_PER_DEG, (b[1] - a[1]) * M_PER_DEG * COS_LAT)
            n = max(1, int(d / step_m))
            out += [a + (b - a) * k / n for k in range(n)]
    return np.unique(np.round(np.array(out), 5), axis=0) if out else np.zeros((0, 2))


def _within(spots: np.ndarray, ref: np.ndarray, radius_m: float) -> np.ndarray:
    if len(spots) == 0 or len(ref) == 0:
        return spots[:0]
    keep = np.zeros(len(spots), dtype=bool)
    for i, (la, lo) in enumerate(spots):
        d = np.hypot((ref[:, 0] - la) * M_PER_DEG, (ref[:, 1] - lo) * M_PER_DEG * COS_LAT)
        keep[i] = d.min() <= radius_m
    return spots[keep]


def _grid(lat: float, lon: float, radius_m: float, step_m: float) -> np.ndarray:
    k = int(radius_m / step_m)
    d = step_m / M_PER_DEG
    return np.array([(lat + i * d, lon + j * d / COS_LAT) for i in range(-k, k + 1) for j in range(-k, k + 1)
                     if i * i + j * j <= k * k])


def candidate_spots(transit: dict | None = None, area: tuple[float, float, float] | None = None,
                    bus: str | None = None, line: str | None = None, line_radius_m: float = 300.0,
                    elevated_radius_m: float | None = None, step_m: float = 30.0) -> np.ndarray:
    """Spots that satisfy every hint given. `area` is (lat, lon, radius_m); `bus` a service
    number (spots along its route); `line` part of an MRT/LRT line name (spots within
    `line_radius_m` of its track); `elevated_radius_m` keeps spots that close to elevated
    track (a viaduct in view)."""
    if bus is not None:
        if not transit or bus not in transit["bus"]:
            raise ValueError(f"no route for bus service {bus}")
        spots = _densify(transit["bus"][bus], step_m)
    elif area is not None:
        spots = _grid(*area, step_m)
    else:
        raise ValueError("give at least a bus service or an area")
    if area is not None and bus is not None:
        spots = _within(spots, np.array([[area[0], area[1]]]), area[2])
    if line:
        track = _densify([r["pts"] for r in transit["rail"] if line.lower() in r["line"].lower()], 50.0)
        spots = _within(spots, track, line_radius_m)
    if elevated_radius_m:
        track = _densify([r["pts"] for r in transit["rail"] if r["elevated"]], 50.0)
        spots = _within(spots, track, elevated_radius_m)
    return spots


def search(buildings: Buildings, photo, spots, view: str = "street", hfov: float | None = None,
           top: int = 5, min_separation_m: float = 60.0, log=None) -> dict:
    """Score every spot; return the best `top` distinct spots and the median score."""
    sky = photo_skyline(photo, hfov)
    if sky is None:
        raise ValueError("no usable skyline in the photo")
    az, el = sky
    scene = photo_scene(photo, hfov)
    scored = []
    for k, (la, lo) in enumerate(spots):
        if view != "window":  # outdoors: not inside a building footprint
            sel, _ = buildings._select(la, lo, 6.0)
            if len(sel):
                continue
        rel, hs = buildings.near(la, lo)
        if len(hs) < 80:
            continue
        for eye in EYE_HEIGHTS[view]:
            c, heading, tilt = match(render(rel, hs, eye), az, el)
            if scene is not None:
                c += MASK_WEIGHT * (1 - view_overlap(scene, render_built(rel, hs, eye), heading, tilt))
            scored.append((c, float(la), float(lo), eye, heading))
        if log and k % 500 == 0:
            log(f"  {k}/{len(spots)} spots")
    scored.sort()
    picked = []
    for r in scored:
        if all(np.hypot((r[1] - p[1]) * M_PER_DEG, (r[2] - p[2]) * M_PER_DEG * COS_LAT) > min_separation_m
               for p in picked):
            picked.append(r)
        if len(picked) == top:
            break
    return {"spots": [{"score": round(c, 3), "lat": round(la, 6), "lon": round(lo, 6), "eye_m": eye,
                       "heading": h} for c, la, lo, eye, h in picked],
            "typical": round(float(np.median([r[0] for r in scored])), 3) if scored else None,
            "n": len(scored)}
