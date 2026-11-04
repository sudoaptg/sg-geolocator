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

TREE_M = 10.0  # canopy height given to mapped greenery (parks, mangrove, wood)
_SAMPLES = np.concatenate([np.arange(2, 60, 2.0), np.arange(60, 300, 5.0), np.arange(300, 1500, 15.0)])


def default_height(tags: dict, area_m2: float) -> float:
    """Roof height when the map gives none: tagged levels/height, else by footprint size
    (a landed house, a shophouse row or small block, a bigger building)."""
    try:
        if tags.get("height"):
            return float(str(tags["height"]).replace("m", "").strip())
        if tags.get("building:levels"):
            return float(tags["building:levels"]) * 3.0
    except ValueError:
        pass
    return 9.0 if area_m2 < 300 else 12.0 if area_m2 < 1500 else 15.0


def _area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2)


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
        known = {}
        if buildings is not None and buildings.labels:
            for i, label in enumerate(buildings.labels):
                if label:
                    known[label.lower()] = float(buildings.heights[buildings.ids == i][0])

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


def search(raster: PlanRaster, labels: np.ndarray, spots, headings_for, hfovs, eyes, horizons, keep: int = 30,
           coarse_size=(24, 18)):
    """Coarse to fine: every (spot, heading) at a small size with a middle eye height and
    view, then the full grid of eye heights, views and horizon rows for the best `keep`."""
    from PIL import Image

    hfovs, eyes, horizons = list(hfovs), list(eyes), list(horizons)
    small = np.asarray(Image.fromarray(labels).resize(coarse_size, Image.NEAREST))
    cw, ch = coarse_size
    rough = []
    for cam in spots:
        hf = hfovs[len(hfovs) // 2]
        for hd in headings_for(cam, hf):
            best = max((agreement(small, raster.render(cam, e, hd, hf, hz, cw, ch)), e, hz)
                       for e in eyes[:: max(1, len(eyes) // 3)] for hz in horizons[:: max(1, len(horizons) // 4)])
            rough.append((best[0], np.asarray(cam, float), float(hd)))
    rough.sort(key=lambda r: -r[0])
    h, w = labels.shape
    best = (-1.0, None, 0.0, 0.0, 0.0, 0.0)
    for _, cam, hd0 in rough[:keep]:
        for hf in hfovs:
            pinned = headings_for(cam, hf)
            for hd in (pinned if len(pinned) == 1 else (hd0 - 3, hd0, hd0 + 3)):
                for e in eyes:
                    for hz in horizons:
                        s = agreement(labels, raster.render(cam, e, hd, hf, hz, w, h))
                        if s > best[0]:
                            best = (s, cam, hd % 360, hf, e, hz)
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
