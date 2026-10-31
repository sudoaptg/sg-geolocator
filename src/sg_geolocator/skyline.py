"""Skyline matching for elevated views (experimental).

From a high floor the outline of the surrounding blocks against the sky is close to a
fingerprint of the spot. This module renders that outline from building footprints and
heights (OpenStreetMap `height` / `building:levels`, HDB storeys from data.gov.sg) for
candidate viewpoints and compares it with the outline found in the photo, trying every
compass heading, a range of camera tilts and several floor heights.

Findings so far (one elevated night photo with a known area): the skyline near the true
spot matched better than all of 300 random spots, but searching all ~2,000 built-up areas
found a spot 10 km away that matched better still. So the skyline narrows things down but
does not identify a spot on its own; it is reported as hints ("skyline also matches near
...") and does not change the prediction.
"""

from __future__ import annotations

import warnings

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
HEADINGS_DEG = _HEADINGS = np.arange(0, 360, 2.0)
TAU_DEG = 0.25  # a profile error of this many degrees costs one log unit
# Weight of the map-match / area score next to the skyline when ranking skyline spots.
PRIOR_WEIGHT = 0.05
REFINE_TOP = 10
REFINE_RADIUS_M = 250.0
REFINE_STEP_M = 50.0
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
    first = _sky_boundary_day(a) if np.median(a[: max(2, h // 10)]) > 90 else _sky_boundary_night(a)
    valid = (first > 2) & (first < h * 0.9)
    # Drop isolated outliers (a cloud edge, a smooth painted stripe): points far from the
    # median of their neighbours.
    pad = np.pad(np.where(valid, first, np.nan).astype(float), 3, constant_values=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows
        local = np.nanmedian(np.stack([pad[i : i + width] for i in range(7)]), axis=0)
    valid &= np.abs(first - local) <= max(3, h // 40)
    if valid.mean() < MIN_VALID_COLUMNS:
        return None
    vfov = hfov * h / width
    cols = np.flatnonzero(valid)
    az = (cols + 0.5 - width / 2) / width * hfov
    el = (h / 2 - first[cols]) / h * vfov
    return az, el


def _sky_boundary_night(a: np.ndarray) -> np.ndarray:
    """Night: the sky is a dark, even band; lit buildings differ from its level."""
    h = len(a)
    top = a[: max(2, h // 20)]
    sky_level, sky_spread = float(np.median(top)), float(np.percentile(np.abs(top - np.median(top)), 90)) + 8.0
    # Smooth vertically so one lit window doesn't end the sky.
    sm = (a[:-2] + a[1:-1] + a[2:]) / 3
    not_sky = np.abs(sm - sky_level) > 2.5 * sky_spread
    return np.where(not_sky.any(axis=0), not_sky.argmax(axis=0), -1)


def _sky_boundary_day(a: np.ndarray, depth: int = 15, share: float = 0.7) -> np.ndarray:
    """Day: sky (blue or cloud) is bright and smooth; white HDB blocks are just as bright
    but textured by windows and corridors. The boundary in each column is the first row
    where most of the next `depth` rows look textured or dark (a thin cloud edge does not
    qualify; a building does)."""
    gy = np.abs(np.diff(a, axis=0, append=a[-1:]))
    gx = np.abs(np.diff(a, axis=1, append=a[:, -1:]))
    g = gx + gy
    # 3x3 box blur of the gradient.
    p = np.pad(g, 1, mode="edge")
    g = sum(p[i : i + len(a), j : j + a.shape[1]] for i in range(3) for j in range(3)) / 9
    top = g[: max(2, len(a) // 10)]
    g_thr = float(np.median(top)) * 3 + 4
    not_sky = ~((g < g_thr) & (a > 100))
    c = np.cumsum(np.vstack([np.zeros((1, a.shape[1])), not_sky]), axis=0)
    ahead = (c[depth:] - c[:-depth]) / depth  # share of non-sky rows in [r, r + depth)
    hit = not_sky[: len(ahead)] & (ahead >= share)
    return np.where(hit.any(axis=0), hit.argmax(axis=0), -1)


def _exif_hfov(img) -> float | None:
    try:
        f35 = img.getexif().get_ifd(0x8769).get(0xA405)  # FocalLengthIn35mmFilm
        return float(np.degrees(2 * np.arctan(36 / (2 * f35)))) if f35 else None
    except Exception:
        return None


def match_cost(profile: np.ndarray, az: np.ndarray, el: np.ndarray) -> float:
    """Lowest median error (degrees) over every heading and camera tilt."""
    return match(profile, az, el)[0]


def match(profile: np.ndarray, az: np.ndarray, el: np.ndarray, headings: np.ndarray | None = None) -> tuple[float, float]:
    """(lowest median error in degrees, the camera heading that gives it, 0 = north), over
    `headings` (default: all)."""
    HEADINGS_DEG = _HEADINGS if headings is None else headings
    bins = (np.round(HEADINGS_DEG[:, None] + az[None, :]).astype(int)) % AZ_BINS
    diff = profile[bins] - el[None, :]  # (headings, columns)
    costs = np.median(np.abs(diff[:, :, None] - PITCH_RANGE_DEG[None, None, :]), axis=1)
    h, _ = np.unravel_index(costs.argmin(), costs.shape)
    return float(costs.min()), float(HEADINGS_DEG[h])


def _best_cost(buildings: Buildings, lat: float, lon: float, az, el, min_points: int = 50) -> float | None:
    rel, hs = buildings.near(lat, lon)
    if len(hs) < min_points:
        return None
    return min(match_cost(render(rel, hs, eye), az, el) for eye in EYE_HEIGHTS_M)


def search(buildings: Buildings, candidates: np.ndarray, prior: np.ndarray, photo, top: int = 5) -> dict | None:
    """Skyline search over candidate areas (one per block, island-wide), refined on a 50 m
    grid around the best few. Returns the best few spots and how they compare with a
    typical spot, or None if the photo has no usable skyline."""
    sky = photo if isinstance(photo, tuple) else photo_skyline(photo)
    if sky is None:
        return None
    az, el = sky
    prior = np.asarray(prior, dtype=float)
    prior = prior - prior.max()
    costs = np.array([_best_cost(buildings, la, lo, az, el) or np.nan for la, lo in candidates])
    ok = np.isfinite(costs)
    if ok.sum() < 10:
        return None
    combined = np.where(ok, -costs / TAU_DEG + PRIOR_WEIGHT * prior, -np.inf)
    step = REFINE_STEP_M / M_PER_DEG
    k = int(REFINE_RADIUS_M / REFINE_STEP_M)
    spots = []
    for i in np.argsort(-combined)[:REFINE_TOP]:
        la0, lo0 = candidates[i]
        best = None
        for di in range(-k, k + 1):
            for dj in range(-k, k + 1):
                if di * di + dj * dj > k * k:
                    continue
                la, lo = la0 + di * step, lo0 + dj * step / COS_LAT
                cost = _best_cost(buildings, la, lo, az, el)
                if cost is not None and (best is None or cost < best[2]):
                    best = (la, lo, cost)
        if best:
            spots.append({"lat": best[0], "lon": best[1], "cost": best[2],
                          "value": -best[2] / TAU_DEG + PRIOR_WEIGHT * prior[i]})
    spots.sort(key=lambda x: -x["value"])
    return {"spots": spots[:top], "median_cost": float(np.nanmedian(costs)), "n": int(ok.sum())}


def check(buildings: Buildings, photo, lat: float, lon: float, radius_m: float = 300.0, n_random: int = 300,
          seed: int = 0) -> dict | None:
    """Diagnostic: best skyline error near a known location vs at random spots with buildings
    around them. If the true area does not beat most random spots, skyline matching cannot
    help for this photo."""
    sky = photo_skyline(photo)
    if sky is None:
        return None
    az, el = sky

    def best(la, lo):
        return _best_cost(buildings, la, lo, az, el)

    step = 40.0 / M_PER_DEG
    k = int(radius_m / 40)
    near = [best(lat + i * step, lon + j * step / COS_LAT) for i in range(-k, k + 1) for j in range(-k, k + 1)
            if i * i + j * j <= k * k]
    near = [x for x in near if x is not None]
    rng = np.random.default_rng(seed)
    pts = buildings.points[rng.choice(len(buildings.points), n_random * 3, replace=False)]
    rand = [x for x in (best(la, lo) for la, lo in pts) if x is not None][:n_random]
    if not near or not rand:
        return None
    b = min(near)
    return {"valid_columns": int(len(az)), "best_near_deg": round(b, 2), "random_median_deg": round(float(np.median(rand)), 2),
            "share_of_random_spots_beaten": round(float(np.mean(np.array(rand) > b)), 3), "n_random": len(rand)}


# --------------------------------------------------------------------------- viewpoint near an anchor

EYE_ABOVE_FLOOR_M = 1.5
FLOORS = range(1, 51)


def viewpoint(buildings: Buildings, photo, lat: float, lon: float, radius_m: float = 600.0) -> dict | None:
    """Where exactly, and how high, a photo was taken, given a place it shows (e.g. a school
    whose sign was read) at (lat, lon). Searches positions within `radius_m` on a 50 m grid
    at a few heights, refines the best ones on a 15 m grid, then tries every floor at the
    best few. Returns the best viewpoint with its floor and heading, how much better it
    matches than a typical spot in the area, and the spread of near-best positions (a
    rough uncertainty), or None if the photo has no usable skyline."""
    sky = photo if isinstance(photo, tuple) else photo_skyline(photo)
    if sky is None:
        return None
    az, el = sky
    # The landmark is in the photo, so the camera faced it: only headings that keep it in
    # the frame are tried (without this, spots looking away from it can fit by chance).
    half_fov = float(np.abs(az).max()) + 3.0
    cache: dict = {}

    def facing_landmark(la, lo):
        """Headings that keep the landmark (whose sign was read) inside the frame."""
        bearing = np.degrees(np.arctan2((lon - lo) * COS_LAT, lat - la)) % 360
        off = (_HEADINGS - bearing + 180) % 360 - 180
        return _HEADINGS[np.abs(off) <= half_fov]

    def near(la, lo):
        key = (round(la, 6), round(lo, 6))
        if key not in cache:
            cache[key] = buildings.near(la, lo)
        return cache[key]

    def cost(la, lo, eye):
        rel, hs = near(la, lo)
        if len(hs) < 50:
            return None
        return match(render(rel, hs, eye), az, el, facing_landmark(la, lo))

    def grid(la0, lo0, radius, step):
        k = int(radius / step)
        d = step / M_PER_DEG
        return [(la0 + i * d, lo0 + j * d / COS_LAT) for i in range(-k, k + 1) for j in range(-k, k + 1)
                if i * i + j * j <= k * k]

    coarse = []
    for la, lo in grid(lat, lon, radius_m, 50.0):
        best = min((c for c in (cost(la, lo, e) for e in EYE_HEIGHTS_M) if c), default=None)
        if best:
            coarse.append((best[0], la, lo))
    if len(coarse) < 5:
        return None
    coarse.sort()
    typical = float(np.median([c for c, _, _ in coarse]))

    fine = []
    for _, la0, lo0 in coarse[:5]:
        for la, lo in grid(la0, lo0, 45.0, 15.0):
            best = min((c for c in (cost(la, lo, e) for e in EYE_HEIGHTS_M) if c), default=None)
            if best:
                fine.append((best[0], la, lo))
    fine = sorted(set(fine))

    results = []
    for _, la, lo in fine[:8]:
        for f in FLOORS:
            c = cost(la, lo, f * FLOOR_M + EYE_ABOVE_FLOOR_M)
            if c:
                results.append((c[0], la, lo, f, c[1]))
    results.sort()
    best = results[0]
    # Positions whose best floor matches nearly as well: how far apart are they?
    close = [r for r in results if r[0] <= best[0] * 1.1 + 0.02]
    spread = max(float(np.hypot((r[1] - best[1]) * M_PER_DEG, (r[2] - best[2]) * M_PER_DEG * COS_LAT)) for r in close)
    floors = sorted({r[3] for r in close})
    return {"lat": best[1], "lon": best[2], "floor": best[3], "heading": best[4], "cost": best[0],
            "typical_cost": typical, "spread_m": spread, "floor_range": (floors[0], floors[-1])}
