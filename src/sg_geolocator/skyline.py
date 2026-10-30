"""Skyline matching for elevated views (experimental).

From a high floor the outline of the surrounding blocks against the sky is close to a
fingerprint of the spot. This module renders that outline from building footprints and
heights (OpenStreetMap `height` / `building:levels`, HDB storeys from data.gov.sg) for
candidate viewpoints and compares it with the outline found in the photo, trying every
compass heading, a range of camera tilts and several floor heights.

It only re-ranks candidates that another stage (the map matcher) already proposed: on its
own a skyline is ambiguous across Singapore's many similar estates. It is not evaluated on
held-out data yet (there is no set of geotagged elevated photos), so it is used only when
the photo shows a clear skyline, and its weight is deliberately modest.
"""

from __future__ import annotations

import numpy as np

from .mapmatch import fetch_hdb_years, normalise_street

M_PER_DEG = 111_000.0
COS_LAT = np.cos(np.radians(1.35))
FLOOR_M = 3.0  # HDB floor-to-floor height is ~2.8-3 m
EYE_HEIGHTS_M = (8.0, 15.0, 25.0, 35.0, 50.0, 70.0)
MAX_DIST_M = 1500.0
MIN_DIST_M = 20.0  # ignore the building the photo is taken from
AZ_BINS = 360
PITCH_RANGE_DEG = np.arange(-15, 16, 1.0)
HEADINGS_DEG = np.arange(0, 360, 2.0)
TAU_DEG = 1.5  # a profile error of this many degrees costs one log unit
MIN_VALID_COLUMNS = 0.25  # share of photo columns with a usable skyline


class Buildings:
    """Footprint edge points (lat, lon) with the building's roof height in metres."""

    def __init__(self, points: np.ndarray, heights: np.ndarray):
        self.points = points
        self.heights = heights
        self.xy = np.stack([(points[:, 1] - 103.8) * M_PER_DEG * COS_LAT, (points[:, 0] - 1.35) * M_PER_DEG], axis=1)
        # Bucket by 500 m tiles for fast radius queries.
        self._tile = 500.0
        keys = np.floor(self.xy / self._tile).astype(np.int64)
        order = np.lexsort((keys[:, 1], keys[:, 0]))
        self._order = order
        self._keys = keys[order]
        uniq, start = np.unique(self._keys, axis=0, return_index=True)
        self._start = {tuple(k): (s, e) for k, s, e in zip(uniq, start, list(start[1:]) + [len(order)])}

    def near(self, lat: float, lon: float, radius_m: float = MAX_DIST_M) -> tuple[np.ndarray, np.ndarray]:
        """(xy relative to the point in metres, heights) of points within radius."""
        p = np.array([(lon - 103.8) * M_PER_DEG * COS_LAT, (lat - 1.35) * M_PER_DEG])
        k0 = np.floor(p / self._tile).astype(int)
        r = int(np.ceil(radius_m / self._tile))
        idx = [np.arange(*self._start[(i, j)]) for i in range(k0[0] - r, k0[0] + r + 1)
               for j in range(k0[1] - r, k0[1] + r + 1) if (i, j) in self._start]
        if not idx:
            return np.zeros((0, 2)), np.zeros(0)
        sel = self._order[np.concatenate(idx)]
        rel = self.xy[sel] - p
        ok = (rel**2).sum(axis=1) <= radius_m**2
        return rel[ok], self.heights[sel][ok]

    @classmethod
    def build(cls, log=print) -> "Buildings":
        from . import osmdata

        osm = osmdata.load(log)
        try:
            storeys = fetch_hdb_storeys(log)
        except Exception as exc:
            log(f"  ::warning::HDB storeys unavailable ({exc})")
            storeys = {}
        pts, hs = [], []
        for ring, tags in osm["buildings"]:
            h = building_height(tags, storeys)
            if h is None:
                continue
            dense = _densify(ring, 8.0)
            pts.append(dense)
            hs.append(np.full(len(dense), h, dtype=np.float32))
        log(f"  buildings with a height: {len(pts)}")
        return cls(np.concatenate(pts), np.concatenate(hs))

    def save(self, path) -> None:
        np.savez_compressed(path, points=self.points.astype(np.float64), heights=self.heights.astype(np.float16))

    @classmethod
    def load(cls, path) -> "Buildings":
        d = np.load(path)
        return cls(d["points"], d["heights"].astype(np.float32))


def fetch_hdb_storeys(log=print) -> dict[tuple[str, str], int]:
    """(block, street) -> number of storeys, from the same HDB dataset as the block ages."""
    return fetch_hdb_years(log, field="max_floor_lvl")


def building_height(tags: dict, storeys: dict) -> float | None:
    try:
        if "height" in tags:
            return float(tags["height"].replace("m", "").strip())
        if "building:levels" in tags:
            return float(tags["building:levels"]) * FLOOR_M
    except ValueError:
        pass
    key = (tags.get("addr:housenumber", "").strip().upper(), normalise_street(tags.get("addr:street", "")))
    return storeys[key] * FLOOR_M if key in storeys else None


def _densify(ring: np.ndarray, step_m: float) -> np.ndarray:
    out = [ring[:1]]
    for a, b in zip(ring[:-1], ring[1:]):
        d = np.hypot((b[0] - a[0]) * M_PER_DEG, (b[1] - a[1]) * M_PER_DEG * COS_LAT)
        n = max(1, int(np.ceil(d / step_m)))
        out.append(a + (b - a) * (np.arange(1, n + 1)[:, None] / n))
    return np.concatenate(out)


def render(rel_xy: np.ndarray, heights: np.ndarray, eye_m: float) -> np.ndarray:
    """Skyline elevation angle (degrees) for each 1° azimuth bin (0 = north, clockwise)."""
    d = np.hypot(rel_xy[:, 0], rel_xy[:, 1])
    ok = d >= MIN_DIST_M
    az = (np.degrees(np.arctan2(rel_xy[ok, 0], rel_xy[ok, 1])) % 360).astype(int) % AZ_BINS
    el = np.degrees(np.arctan2(heights[ok] - eye_m, d[ok]))
    out = np.full(AZ_BINS, -5.0)  # open sky: the horizon, slightly below eye level
    np.maximum.at(out, az, el)
    return out


# --------------------------------------------------------------------------- photo side


def photo_skyline(path, hfov_deg: float | None = None, width: int = 240) -> tuple[np.ndarray, np.ndarray] | None:
    """(azimuth offset from the image centre, elevation above the image centre) in degrees
    for each image column where the sky meets something, or None if there is no usable
    skyline. Columns where the top of the frame is not sky (window frames, trees, ceiling)
    are dropped."""
    from PIL import Image

    img = Image.open(path)
    hfov = hfov_deg or _exif_hfov(img) or 70.0
    img = img.convert("L")
    h = int(round(img.height * width / img.width))
    a = np.asarray(img.resize((width, h)), dtype=np.float32)
    top = a[: max(2, h // 20)]
    sky_level, sky_spread = float(np.median(top)), float(np.percentile(np.abs(top - np.median(top)), 90)) + 8.0
    # Smooth vertically so one lit window doesn't end the sky.
    sm = (a[:-2] + a[1:-1] + a[2:]) / 3
    not_sky = np.abs(sm - sky_level) > 2.5 * sky_spread
    first = np.where(not_sky.any(axis=0), not_sky.argmax(axis=0), -1)
    valid = (first > 2) & (first < h * 0.9)
    if valid.mean() < MIN_VALID_COLUMNS:
        return None
    vfov = hfov * h / width
    cols = np.flatnonzero(valid)
    az = (cols + 0.5 - width / 2) / width * hfov
    el = (h / 2 - first[cols]) / h * vfov
    return az, el


def _exif_hfov(img) -> float | None:
    try:
        f35 = img.getexif().get_ifd(0x8769).get(0xA405)  # FocalLengthIn35mmFilm
        return float(np.degrees(2 * np.arctan(36 / (2 * f35)))) if f35 else None
    except Exception:
        return None


def match_cost(profile: np.ndarray, az: np.ndarray, el: np.ndarray) -> float:
    """Lowest median error (degrees) over every heading and camera tilt."""
    bins = (np.round(HEADINGS_DEG[:, None] + az[None, :]).astype(int)) % AZ_BINS
    diff = profile[bins] - el[None, :]  # (headings, columns)
    costs = np.median(np.abs(diff[:, :, None] - PITCH_RANGE_DEG[None, None, :]), axis=1)
    return float(costs.min())


def rerank(buildings: Buildings, candidates: np.ndarray, scores: np.ndarray, photo) -> tuple[np.ndarray, list[float]] | None:
    """Add a skyline term to candidate scores. `candidates` is (n, 2) lat/lon. Returns the
    new scores and each candidate's best skyline error, or None if the photo has no
    usable skyline."""
    sky = photo if isinstance(photo, tuple) else photo_skyline(photo)
    if sky is None:
        return None
    az, el = sky
    costs = []
    for lat, lon in candidates:
        rel, hs = buildings.near(lat, lon)
        if len(hs) == 0:
            costs.append(float("inf"))
            continue
        costs.append(min(match_cost(render(rel, hs, eye), az, el) for eye in EYE_HEIGHTS_M))
    costs = np.array(costs)
    finite = np.isfinite(costs)
    if not finite.any():
        return None
    costs[~finite] = costs[finite].max()
    return scores - costs / TAU_DEG, costs.tolist()
