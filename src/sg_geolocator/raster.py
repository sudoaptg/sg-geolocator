"""Fast render-and-compare on a map grid.

The map near an area is turned into a 4 m grid: what covers each cell (building, water,
greenery, open ground) and how tall it is (building heights; tree canopy for greenery).
A view is drawn by stepping along each image column's line of sight: a pixel shows the first
building or tree whose side spans its angle, else the ground it looks down on (water,
greenery or plain ground), else sky. This is tens of times faster than drawing polygons, so
whole areas can be searched, and cheap checks on what the photo shows (how much water,
greenery, buildings) can throw out wrong places before any of it.
"""

from __future__ import annotations

import numpy as np

from .plan2d import angle_from_x, to_xy
from .scene import BUILDING, GREEN, GROUND, SKY, WATER

TREE_M = 10.0
CONDO_M = 60.0  # 20 storeys: default for flat blocks (building=apartments) with no height in the map  # canopy height given to mapped greenery (parks, mangrove, wood)
_SAMPLES = np.concatenate([np.arange(2, 60, 2.0), np.arange(60, 300, 5.0), np.arange(300, 1500, 15.0)])


def default_height(tags: dict, area_m2: float) -> float:
    """Roof height when the map gives none: tagged levels/height, else by footprint size
    (a landed house, a shophouse row or small block, a bigger building)."""
    try:  # a tagged "0" means unknown, not flat
        if tags.get("height") and float(str(tags["height"]).replace("m", "").strip()) > 0:
            return float(str(tags["height"]).replace("m", "").strip())
        if tags.get("building:levels") and float(tags["building:levels"]) > 0:
            return float(tags["building:levels"]) * 3.0
    except ValueError:
        pass
    if tags.get("building") in ("apartments", "residential") and area_m2 >= 300:
        return CONDO_M  # an unmapped flat block: most newer condos are about 20 storeys
    return 9.0 if area_m2 < 300 else 12.0 if area_m2 < 1500 else 15.0


def _area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2)


def _known_heights(buildings) -> dict:
    """Address -> roof height from a Buildings file (e.g. HDB storeys), computed once."""
    cached = getattr(buildings, "_known_heights", None)
    if cached is None:
        cached = {}
        if buildings.labels:
            ids, first = np.unique(buildings.ids, return_index=True)
            for i, k in zip(ids, first):
                label = buildings.labels[int(i)] if int(i) < len(buildings.labels) else ""
                if label and buildings.heights[k] > 0:
                    cached[label.lower()] = float(buildings.heights[k])
        buildings._known_heights = cached
    return cached


class PlanRaster:
    """Kinds and heights on a grid around (lat, lon). `plan` is an export-area dict (buildings,
    water, green); `buildings` (optional) supplies known heights (e.g. HDB storeys) by address."""

    def __init__(self, plan: dict, lat: float, lon: float, radius_m: float, buildings=None, res: float = 4.0):
        from PIL import Image, ImageDraw

        self.res = res
        c = to_xy(lat, lon)
        self.origin = c - radius_m
        n = int(2 * radius_m / res) + 1
        kind = Image.new("L", (n, n), GROUND)
        height = Image.new("F", (n, n), 0.0)
        dk, dh = ImageDraw.Draw(kind), ImageDraw.Draw(height)
        known = _known_heights(buildings) if buildings is not None else {}

        def cells(pts):
            xy = to_xy(*np.asarray(pts, float).T)
            if np.abs(xy - c).max(1).min() > radius_m + 50:
                return None, None
            g = (xy - self.origin) / res
            return xy, [(float(x), float(n - 1 - y)) for x, y in g]  # row 0 = north edge

        for item in plan.get("green", []):
            _, poly = cells(item["pts"])
            if poly and len(poly) > 2:
                dk.polygon(poly, fill=GREEN)
                dh.polygon(poly, fill=TREE_M)
        for item in plan.get("water", []):
            _, poly = cells(item["pts"])
            if poly and len(poly) > 2:
                dk.polygon(poly, fill=WATER)
                dh.polygon(poly, fill=0.0)
        for item in plan.get("buildings", []):
            xy, poly = cells(item["pts"])
            if not poly or len(poly) < 3:
                continue
            label = f"blk {item.get('addr:housenumber', '')} {item.get('addr:street', '')}".lower().strip()
            h = known.get(label) or default_height(item, _area(xy))
            dk.polygon(poly, fill=BUILDING)
            dh.polygon(poly, fill=float(h))
        self.kind = np.asarray(kind)[::-1].T.copy()  # [x index, y index] with y increasing north
        self.height = np.asarray(height, dtype=np.float32)[::-1].T.copy()
        self.n = n

    def near(self, cam, kind: int, radius_m: float) -> bool:
        """Is there any `kind` (water, green, ...) within `radius_m` of `cam`?"""
        i, j = ((np.asarray(cam) - self.origin) / self.res).astype(int)
        r = int(radius_m / self.res)
        sub = self.kind[max(0, i - r): i + r + 1, max(0, j - r): j + r + 1]
        return bool((sub == kind).any())

    def render(self, cam, eye: float, heading: float, hfov: float, horizon: float, width: int, height: int) -> np.ndarray:
        cols = (np.arange(width) + 0.5) / width
        az = np.radians(heading + angle_from_x(cols, hfov))  # (W,)
        pos = np.asarray(cam)[None, None, :] + _SAMPLES[None, :, None] * np.stack([np.sin(az), np.cos(az)], -1)[:, None, :]
        ij = ((pos - self.origin) / self.res).astype(int)
        inside = (ij[..., 0] >= 0) & (ij[..., 0] < self.n) & (ij[..., 1] >= 0) & (ij[..., 1] < self.n)
        ij = np.clip(ij, 0, self.n - 1)
        k = np.where(inside, self.kind[ij[..., 0], ij[..., 1]], GROUND)  # (W, S)
        h = np.where(inside, self.height[ij[..., 0], ij[..., 1]], 0.0)
        f = (width / 2) / np.tan(np.radians(hfov / 2))
        rows = (np.arange(height) + 0.5)
        el = np.arctan((horizon * height - rows) / f)  # (H,) elevation angle of each row
        r = _SAMPLES[None, :]
        top = np.arctan2(h - eye, r)  # (W, S)
        base = np.arctan2(-eye, r) * np.ones_like(top)
        solid = h > 0
        hit = solid[:, None, :] & (base[:, None, :] <= el[None, :, None]) & (el[None, :, None] <= top[:, None, :])  # W,H,S
        any_hit = hit.any(-1)
        first = hit.argmax(-1)  # (W, H)
        obj = np.take_along_axis(k[:, None, :].repeat(height, 1), first[..., None], -1)[..., 0]
        # ground the row looks down on
        with np.errstate(divide="ignore"):
            rg = np.where(el < 0, eye / np.tan(-el), np.inf)  # (H,)
        gi = np.clip(np.searchsorted(_SAMPLES, rg), 0, len(_SAMPLES) - 1)  # (H,)
        ground = np.where(np.isfinite(rg)[None, :], k[:, gi], SKY)
        ground = np.where(np.isin(ground, (BUILDING,)), GROUND, ground)  # a roof seen from above: just ground here
        out = np.where(any_hit, obj, ground).T  # (H, W)
        return out.astype(np.uint8)


def agreement(photo: np.ndarray, rendered: np.ndarray) -> float:
    """Mean IoU over sky, building, water and greenery (water/greenery only when either side
    has some). Photo pixels labelled other/ground don't count against map buildings or trees."""
    scores = []
    for kk in (SKY, BUILDING, WATER, GREEN):
        p, q = photo == kk, rendered == kk
        if kk in (WATER, GREEN) and not (p.any() or q.any()):
            continue
        union = (p | q).sum()
        scores.append((p & q).sum() / union if union else 1.0)
    return float(np.mean(scores)) if scores else 0.0


def composition(labels: np.ndarray) -> dict:
    """Share of the photo that is sky, building, water, greenery."""
    return {name: float((labels == kk).mean()) for name, kk in (("sky", SKY), ("building", BUILDING), ("water", WATER),
                                                                 ("green", GREEN))}


_JOB = {}  # shared with forked workers: the raster, labels and ranges of the current search


def _rough(task):
    cam, hd, hf = task
    J = _JOB
    cw, ch = J["coarse_size"]
    return max(agreement(J["small"], J["raster"].render(cam, e, hd, hf, hz, cw, ch))
               for e in J["eyes_c"] for hz in J["horizons_c"])


def _fine(task):
    cam, hd, hf = task
    J = _JOB
    h, w = J["labels"].shape
    return max((agreement(J["labels"], J["raster"].render(cam, e, hd, hf, hz, w, h)), e, hz)
               for e in J["eyes"] for hz in J["horizons"])


def _map(fn, tasks, workers):
    if workers <= 1 or len(tasks) < 64:
        return [fn(t) for t in tasks]
    import multiprocessing as mp

    with mp.get_context("fork").Pool(workers) as pool:  # workers inherit the raster; nothing big is sent
        return pool.map(fn, tasks, chunksize=max(16, len(tasks) // (workers * 8)))


def search(raster: PlanRaster, labels: np.ndarray, spots, headings_for, hfovs, eyes, horizons, keep: int = 30,
           coarse_size=(24, 18), workers: int | None = None):
    """Coarse to fine: every (spot, heading) at a small size with a middle eye height and
    view, then the full grid of eye heights, views and horizon rows for the best `keep`.
    Both stages run on all CPU cores."""
    import os

    from PIL import Image

    workers = workers or os.cpu_count() or 1
    hfovs, eyes, horizons = list(hfovs), list(eyes), list(horizons)
    _JOB.update(raster=raster, labels=labels, coarse_size=coarse_size, eyes=eyes, horizons=horizons,
                small=np.asarray(Image.fromarray(labels).resize(coarse_size, Image.NEAREST)),
                eyes_c=eyes[len(eyes) // 2: len(eyes) // 2 + 1], horizons_c=horizons[::2])
    hf_mid = hfovs[len(hfovs) // 2]
    tasks = [(np.asarray(cam, float), float(hd), hf_mid) for cam in spots for hd in headings_for(cam, hf_mid)]
    scores = _map(_rough, tasks, workers)
    order = np.argsort(scores)[::-1][:keep]
    fine = []
    for k in order:
        cam, hd0, _ = tasks[k]
        for hf in hfovs:
            pinned = headings_for(cam, hf)
            for hd in (pinned if len(pinned) == 1 else (hd0 - 6, hd0 - 3, hd0, hd0 + 3, hd0 + 6)):
                fine.append((cam, float(hd), float(hf)))
    results = _map(_fine, fine, workers)
    best = (-1.0, None, 0.0, 0.0, 0.0, 0.0)
    for (cam, hd, hf), (sc, e, hz) in zip(fine, results):
        if sc > best[0]:
            best = (sc, cam, hd % 360, hf, e, hz)
    return best


class Plan:
    """An export-area file (buildings, water, green), indexed so the part near a spot can be
    cut out quickly."""

    def __init__(self, data: dict):
        self.data = data
        self.centres = {}
        for kind in ("buildings", "water", "green"):
            items = data.get(kind, [])
            self.centres[kind] = to_xy(*np.array([np.mean(np.asarray(it["pts"], float), 0) for it in items]).T) \
                if items else np.zeros((0, 2))
            # big polygons (rivers, parks) need their extent, not just their centre
            self.data[kind] = items
        self._pts = {kind: [to_xy(*np.asarray(it["pts"], float).T) for it in self.data.get(kind, [])]
                     for kind in ("water", "green")}

    @classmethod
    def load(cls, path) -> "Plan":
        import gzip
        import json

        return cls(json.load(gzip.open(path, "rt")))

    @classmethod
    def from_buildings(cls, buildings, water=None) -> "Plan":
        """Without an export file: footprints with known heights only, plus water rings."""
        items = []
        for i in np.unique(buildings.ids):
            pts, h = buildings.footprint(int(i))
            items.append({"pts": pts.tolist(), "height": str(h)})
        return cls({"buildings": items, "water": [{"pts": w} for w in (water or [])], "green": []})

    def near(self, lat: float, lon: float, radius_m: float) -> dict:
        c = to_xy(lat, lon)
        out = {}
        for kind in ("buildings", "water", "green"):
            items = self.data.get(kind, [])
            if kind == "buildings":
                keep = np.flatnonzero(np.hypot(*(self.centres[kind] - c).T) < radius_m + 100) if len(items) else []
            else:
                keep = [i for i, p in enumerate(self._pts[kind]) if np.hypot(*(p - c).T).min() < radius_m + 100]
            out[kind] = [items[i] for i in keep]
        return out

    def has(self, kind: str, lat: float, lon: float, radius_m: float) -> bool:
        """Any water / green within `radius_m`."""
        c = to_xy(lat, lon)
        return any(np.hypot(*(p - c).T).min() < radius_m for p in self._pts.get(kind, []))


def facade_spots(plan_part: dict, lat: float, lon: float, radius_m: float, min_area: float = 300.0,
                 step_m: float = 24.0) -> np.ndarray:
    """Points along the outlines of block-sized buildings (footprint over `min_area`) within
    `radius_m`: where a photo from a high window or a rooftop can have been taken."""
    c = to_xy(lat, lon)
    out = []
    for item in plan_part.get("buildings", []):
        xy = to_xy(*np.asarray(item["pts"], float).T)
        if len(xy) < 4 or _area(xy) < min_area or np.hypot(*(xy - c).T).min() > radius_m:
            continue
        for a, b in zip(xy[:-1], xy[1:]):
            n = max(1, int(np.hypot(*(b - a)) / step_m))
            out += [a + (b - a) * t for t in np.arange(n) / n]
    return np.array(out) if out else np.zeros((0, 2))


def building_heights(plan_part: dict, buildings=None) -> list[float]:
    """Roof height of each building in a plan-map part: HDB's storey data by address when
    there is a match, else the map's own height/levels, else the default by type and size."""
    known = _known_heights(buildings) if buildings is not None else {}
    out = []
    for item in plan_part.get("buildings", []):
        label = f"blk {item.get('addr:housenumber', '')} {item.get('addr:street', '')}".lower().strip()
        xy = to_xy(*np.asarray(item["pts"], float).T)
        out.append(known.get(label) or default_height(item, _area(xy)))
    return out


# --------------------------------------------------------------------------- island-wide water search


def candidate_waters(plan: "Plan", buildings=None, min_area: float = 10000.0, min_tall: int = 6,
                     tall_m: float = 45.0, within_m: float = 400.0) -> list[np.ndarray]:
    """Lakes, ponds and river stretches (outlines in metres) at least `min_area` big with at
    least `min_tall` tall blocks within `within_m` of the water's edge."""
    heights = np.array(building_heights(plan.data, buildings))
    tall = plan.centres["buildings"][heights >= tall_m]
    out = []
    for pts in plan._pts["water"]:
        if len(pts) < 4 or _area(pts) < min_area:
            continue
        edge = pts[:: max(1, len(pts) // 40)]
        d = np.min(np.hypot(*(tall[:, None, :] - edge[None, :, :]).transpose(2, 0, 1)), axis=1)
        if (d < within_m).sum() >= min_tall:
            out.append(pts)
    return out


def shore_spots(pts: np.ndarray, step_m: float = 25.0, cap: int = 40) -> np.ndarray:
    """Points along a water outline, about `step_m` apart (at most `cap`, spread evenly)."""
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.hypot(*(b - a)) / step_m))
        out += [a + (b - a) * t for t in np.arange(n) / n]
    out = np.array(out) if out else pts[:1]
    return out[np.linspace(0, len(out) - 1, min(cap, len(out))).astype(int)]


_LAKE = {}


def _quick_lake(pts):
    J = _LAKE
    lat, lon = _to_latlon_xy(pts.mean(0))
    r = float(np.hypot(*(pts - pts.mean(0)).T).max())
    pr = PlanRaster(J["plan"].near(lat, lon, r + 900), lat, lon, r + 900, J["buildings"])
    cw, ch = J["small"].shape[1], J["small"].shape[0]
    best = (-1.0, None, 0.0, 0.0)
    for cam in shore_spots(pts):
        for hd in range(0, 360, 10):
            ahead = cam + 30 * np.array([np.sin(np.radians(hd)), np.cos(np.radians(hd))])
            i, j = ((ahead - pr.origin) / pr.res).astype(int)
            if not (0 <= i < pr.n and 0 <= j < pr.n) or pr.kind[i, j] != WATER:
                continue  # look across the water, not away from it
            for hf in J["hfovs_c"]:
                for hz in (0.4, 0.5, 0.6):
                    s = agreement(J["small"], pr.render(cam, 1.5, hd, hf, hz, cw, ch))
                    if s > best[0]:
                        best = (s, cam, float(hd), float(hf))
    return best


def _to_latlon_xy(xy):
    from .plan2d import to_latlon

    return to_latlon(xy)


def search_waters(plan: "Plan", labels: np.ndarray, buildings=None, top: int = 5, workers: int | None = None,
                  log=None) -> list[dict]:
    """No area given, photo taken at the water: every candidate lake/pond/river stretch gets a
    quick render search from its shore looking across the water (10 degree steps, normal to
    ultra-wide lens); the best `top` get the full search."""
    import multiprocessing as mp
    import os

    from PIL import Image

    lakes = candidate_waters(plan, buildings)
    if log:
        log(f"  {len(lakes)} candidate lakes/ponds/river stretches")
    _LAKE.update(plan=plan, buildings=buildings, hfovs_c=(70.0, 100.0),
                 small=np.asarray(Image.fromarray(labels).resize((24, 18), Image.NEAREST)))
    with mp.get_context("fork").Pool(workers or os.cpu_count() or 1) as pool:
        quick = pool.map(_quick_lake, lakes, chunksize=2)
    order = [k for k in np.argsort([-q[0] for q in quick]) if quick[k][1] is not None][:top]
    out = []
    for k in order:
        _, cam0, hd0, hf0 = quick[k]
        lat, lon = _to_latlon_xy(cam0)
        pr = PlanRaster(plan.near(lat, lon, 1300), lat, lon, 1300, buildings)
        spots = [cam0 + (dx, dy) for dx in range(-40, 41, 10) for dy in range(-40, 41, 10)]
        ag, cam, hd, hf, eye, hz = search(pr, labels, spots, lambda c, h: list(np.arange(hd0 - 15, hd0 + 15.1, 5.0)),
                                          [hf0 - 10, hf0, hf0 + 10], [1.0, 2.0, 4.0], np.arange(0.35, 0.66, 0.03),
                                          keep=15, workers=workers)
        la, lo = _to_latlon_xy(cam)
        out.append({"lat": round(la, 6), "lon": round(lo, 6), "heading": round(hd, 1), "hfov": hf, "eye_m": eye,
                    "horizon": round(hz, 3), "agreement": round(ag, 4), "quick": round(quick[k][0], 4),
                    "map": f"https://www.google.com/maps?q={la:.6f},{lo:.6f}"})
        if log:
            log(f"  {la:.5f},{lo:.5f}: quick {quick[k][0]:.3f} -> full {ag:.3f}")
    return sorted(out, key=lambda o: -o["agreement"])
