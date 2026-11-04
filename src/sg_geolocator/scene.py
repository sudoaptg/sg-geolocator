"""Render-and-compare: label each pixel of the photo (sky, building, water, ...) and compare
it with the map drawn from a candidate camera.

The photo side uses SegFormer (ADE20K classes, grouped into a few kinds). The map side
draws, from a camera spot, eye height, heading, field of view and horizon row: sky above the
horizon, ground below, mapped water on the ground, and every building as walls and a roof,
far ones first so near ones hide them. The score is how much the two label images agree
on sky, building and water. It uses every cue at once (outline, building bulk, water shape)
and is much harder to fool than edges alone.
"""

from __future__ import annotations

import numpy as np

from .plan2d import to_xy

SKY, BUILDING, WATER, GROUND, GREEN, OTHER = 0, 1, 2, 3, 4, 5
KINDS = {SKY: "sky", BUILDING: "building", WATER: "water", GROUND: "ground", GREEN: "green", OTHER: "other"}

# ADE20K label names -> our kinds (anything else is OTHER)
_ADE = {
    SKY: {"sky"},
    BUILDING: {"building", "house", "skyscraper", "wall", "tower", "windowpane", "column", "awning", "booth"},
    WATER: {"water", "sea", "river", "lake", "swimming pool", "waterfall", "fountain"},
    GROUND: {"road", "sidewalk", "earth", "floor", "path", "field", "sand", "land", "runway", "dirt track", "stairway",
             "bridge", "car", "bus", "truck"},
    GREEN: {"tree", "grass", "plant", "palm", "flower", "mountain", "hill"},
}


class Segmenter:
    """SegFormer-B0 (ADE20K). Weights from $SG_GEOLOCATOR_MODELS/segformer (the segmenter
    release) or the Hugging Face hub."""

    def __init__(self, model: str = "nvidia/segformer-b0-finetuned-ade-512-512"):
        import os
        from pathlib import Path

        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        root = os.environ.get("SG_GEOLOCATOR_MODELS")
        local = Path(root) / "segformer" if root else None
        src = str(local) if local and local.exists() else model
        self.processor = SegformerImageProcessor.from_pretrained(src)
        self.model = SegformerForSemanticSegmentation.from_pretrained(src).eval()
        self._torch = torch
        names = self.model.config.id2label
        self.lut = np.full(len(names), OTHER, dtype=np.uint8)
        for i, name in names.items():
            for kind, group in _ADE.items():
                if any(n.strip() in group for n in name.split(";")) or name.split(",")[0].strip() in group:
                    self.lut[int(i)] = kind

    def labels(self, path, width: int = 256) -> np.ndarray:
        """Kind of every pixel, at `width` columns."""
        from PIL import Image

        img = Image.open(path).convert("RGB")
        h = int(round(img.height * width / img.width))
        inputs = self.processor(images=img, return_tensors="pt")
        with self._torch.no_grad():
            logits = self.model(**inputs).logits  # (1, classes, H/4, W/4)
        up = self._torch.nn.functional.interpolate(logits, size=(h, width), mode="bilinear", align_corners=False)
        return self.lut[up.argmax(1)[0].numpy()]


class MapScene:
    """Buildings (footprints with heights) and water near an area, ready to draw from any camera."""

    def __init__(self, buildings, lat: float, lon: float, radius_m: float = 1500.0, water=None,
                 default_height: float = 9.0):
        sel, _ = buildings._select(lat, lon, radius_m)
        ids = np.unique(buildings.ids[sel])
        self.blocks = []
        for i in ids:
            m = buildings.ids == i
            pts = buildings.xy[m][::2]  # footprints are densified every 8 m; every other point is plenty
            if len(pts) < 3:
                continue
            h = float(buildings.heights[m][0]) or default_height
            self.blocks.append((pts, h))
        c = to_xy(lat, lon)
        self.water = []
        for w in water or []:
            p = to_xy(*np.array(w).T)
            if len(p) > 3 and np.hypot(*(p - c).T).min() < radius_m:
                self.water.append(p)

    def render(self, cam, eye: float, heading: float, hfov: float, horizon: float, width: int, height: int,
               max_dist: float = 1500.0) -> np.ndarray:
        """Kinds image as seen from `cam` (metres, Buildings.xy grid). `horizon` is the row
        (0..1 down the frame) where the true horizon sits."""
        from PIL import Image, ImageDraw

        th = np.radians(heading)
        f = (width / 2) / np.tan(np.radians(hfov / 2))
        hrow = horizon * height
        img = Image.new("L", (width, height), GROUND)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, width, max(0, hrow)], fill=SKY)

        def proj(p, z):
            rel = p - cam
            fwd = rel[:, 0] * np.sin(th) + rel[:, 1] * np.cos(th)
            rt = rel[:, 0] * np.cos(th) - rel[:, 1] * np.sin(th)
            fwd = np.maximum(fwd, 0.5)
            return np.c_[width / 2 + f * rt / fwd, hrow - f * (z - eye) / fwd]

        def ahead(p):
            rel = p - cam
            return rel[:, 0] * np.sin(th) + rel[:, 1] * np.cos(th)

        for p in self.water:  # water lies on the ground; clip to what is in front
            a = ahead(p)
            if (a > 1).sum() >= 3:
                d.polygon([tuple(v) for v in proj(p[a > 1], 0.0)], fill=WATER)
        order = []
        for pts, h in self.blocks:
            a = ahead(pts)
            dist = np.hypot(*(pts - cam).T)
            if dist.min() < 3 or dist.min() > max_dist or (a > 1).sum() < 2:
                continue
            order.append((dist.min(), pts, h))
        for _, pts, h in sorted(order, key=lambda o: -o[0]):  # far first
            a = ahead(pts)
            g, r = proj(pts, 0.0), proj(pts, h)
            for k in range(len(pts)):
                k2 = (k + 1) % len(pts)
                if a[k] > 1 and a[k2] > 1:
                    d.polygon([tuple(g[k]), tuple(g[k2]), tuple(r[k2]), tuple(r[k])], fill=BUILDING)
            if eye > h and (a > 1).all():
                d.polygon([tuple(v) for v in r], fill=BUILDING)
        return np.asarray(img)


def agreement(photo: np.ndarray, rendered: np.ndarray) -> float:
    """Mean IoU over sky, building and water (water only when either side has some).
    Photo pixels that are greenery, ground or other don't count against building: trees
    and street furniture hide buildings the map knows about."""
    scores = []
    for k in (SKY, BUILDING, WATER):
        p, r = photo == k, rendered == k
        if k == BUILDING:  # map building hidden behind photo greenery/other is not wrong
            r = r & ~np.isin(photo, (GREEN, OTHER))
        if k == WATER and not (p.any() or r.any()):
            continue
        union = (p | r).sum()
        scores.append((p & r).sum() / union if union else 1.0)
    return float(np.mean(scores)) if scores else 0.0


def refine(scene: MapScene, labels: np.ndarray, cam, heading: float, hfov: float, eyes=None, horizons=None,
           headings=None) -> dict:
    """Best eye height, horizon row and (small) heading change at one spot, by agreement."""
    h, w = labels.shape
    eyes = np.arange(1.5, 60, 3.0) if eyes is None else eyes
    horizons = np.arange(0.25, 0.76, 0.05) if horizons is None else horizons
    headings = [heading - 2, heading, heading + 2] if headings is None else headings
    best = (-1.0, 0.0, 0.0, heading)
    for hd in headings:
        for eye in eyes:
            for hz in horizons:
                s = agreement(labels, scene.render(cam, eye, hd, hfov, hz, w, h))
                if s > best[0]:
                    best = (s, float(eye), float(hz), float(hd))
    s, eye, hz, hd = best
    return {"agreement": round(s, 4), "eye_m": eye, "horizon": round(hz, 3), "heading": round(hd % 360, 1)}


def inside(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Which points lie inside a polygon (even-odd rule)."""
    x, y = pts[:, 0], pts[:, 1]
    res = np.zeros(len(pts), bool)
    px, py = poly[:, 0], poly[:, 1]
    j = len(poly) - 1
    for i in range(len(poly)):
        cross = (py[i] > y) != (py[j] > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i]) + px[i]
        res ^= cross & (x < xi)
        j = i
    return res


def water_spots(scene: MapScene, near, radius_m: float, step_m: float = 15.0) -> np.ndarray:
    """Points on mapped water within `radius_m` of `near` (metres), about `step_m` apart:
    for photos taken at the water's edge or from a boat."""
    out = []
    for p in scene.water:
        lo, hi = p.min(0), p.max(0)
        gx, gy = np.meshgrid(np.arange(lo[0], hi[0], step_m), np.arange(lo[1], hi[1], step_m))
        g = np.c_[gx.ravel(), gy.ravel()]
        g = g[np.hypot(*(g - near).T) < radius_m]
        if len(g):
            out.append(g[inside(p, g)])
    return np.concatenate(out) if out else np.zeros((0, 2))


def load_water(path) -> list:
    """Water outlines ([lat, lon] rings) from an export-area file (e.g. the sgwater release)."""
    import gzip
    import json

    data = json.load(gzip.open(path, "rt"))
    return [w["pts"] for w in data["water"] if len(w["pts"]) > 3 and w["pts"][0] == w["pts"][-1]]


def search(scene: MapScene, labels: np.ndarray, spots, headings_for, hfovs, eyes, horizons, log=None):
    """Best (agreement, spot, heading, hfov, eye, horizon) over candidate spots.
    `headings_for(spot, hfov)` gives the headings to try at a spot (e.g. pinned by a
    landmark, or around the plan-view answer)."""
    h, w = labels.shape
    best = (-1.0, None, 0.0, 0.0, 0.0, 0.0)
    for n, cam in enumerate(spots):
        for hf in hfovs:
            for hd in headings_for(cam, hf):
                for eye in eyes:
                    for hz in horizons:
                        s = agreement(labels, scene.render(cam, eye, hd, hf, hz, w, h))
                        if s > best[0]:
                            best = (s, np.asarray(cam, float), float(hd) % 360, float(hf), float(eye), float(hz))
        if log and n and n % 50 == 0:
            log(f"    {n}/{len(spots)} spots")
    return best
