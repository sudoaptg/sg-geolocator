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
    """Footprint edge points (lat, lon) with the building's roof height in metres. `ids`
    gives each point's building, `labels` each building's address ("Blk 123 ANG MO KIO AVE
    3", or "" when unknown)."""

    def __init__(self, points: np.ndarray, heights: np.ndarray, ids: np.ndarray | None = None,
                 labels: list[str] | None = None):
        self.points = points
        self.heights = heights
        self.ids = np.arange(len(points)) if ids is None else ids
        self.labels = labels or []
        self.xy = np.stack([(points[:, 1] - 103.8) * M_PER_DEG * COS_LAT, (points[:, 0] - 1.35) * M_PER_DEG], axis=1)
        # Bucket by 500 m tiles for fast radius queries.
        self._tile = 500.0
        keys = np.floor(self.xy / self._tile).astype(np.int64)
        order = np.lexsort((keys[:, 1], keys[:, 0]))
        self._order = order
        self._keys = keys[order]
        uniq, start = np.unique(self._keys, axis=0, return_index=True)
        self._start = {tuple(k): (s, e) for k, s, e in zip(uniq, start, list(start[1:]) + [len(order)])}

    def _select(self, lat: float, lon: float, radius_m: float):
        p = np.array([(lon - 103.8) * M_PER_DEG * COS_LAT, (lat - 1.35) * M_PER_DEG])
        k0 = np.floor(p / self._tile).astype(int)
        r = int(np.ceil(radius_m / self._tile))
        idx = [np.arange(*self._start[(i, j)]) for i in range(k0[0] - r, k0[0] + r + 1)
               for j in range(k0[1] - r, k0[1] + r + 1) if (i, j) in self._start]
        if not idx:
            return np.zeros(0, dtype=int), p
        sel = self._order[np.concatenate(idx)]
        return sel[((self.xy[sel] - p) ** 2).sum(axis=1) <= radius_m**2], p

    def near(self, lat: float, lon: float, radius_m: float = MAX_DIST_M, exclude: int | None = None):
        """(xy relative to the point in metres, heights) of points within radius, leaving
        out building `exclude` (the one the photo is taken from)."""
        sel, p = self._select(lat, lon, radius_m)
        if exclude is not None:
            sel = sel[self.ids[sel] != exclude]
        return self.xy[sel] - p, self.heights[sel]

    def addressed_near(self, lat: float, lon: float, radius_m: float) -> list[int]:
        """Ids of buildings with an address that have points within radius."""
        if not self.labels:
            return []
        sel, _ = self._select(lat, lon, radius_m)
        return sorted({int(i) for i in self.ids[sel] if self.labels[i]})

    def footprint(self, building: int) -> tuple[np.ndarray, float]:
        """(lat, lon) points and roof height of one building."""
        m = self.ids == building
        return self.points[m], float(self.heights[m][0])

    @classmethod
    def build(cls, log=print) -> "Buildings":
        from . import osmdata

        osm = osmdata.load(log)
        try:
            storeys = fetch_hdb_storeys(log)
        except Exception as exc:
            log(f"  ::warning::HDB storeys unavailable ({exc})")
            storeys = {}
        pts, hs, ids, labels = [], [], [], []
        for ring, tags in osm["buildings"]:
            h = building_height(tags, storeys)
            if h is None:
                continue
            dense = _densify(ring, 8.0)
            pts.append(dense)
            hs.append(np.full(len(dense), h, dtype=np.float32))
            ids.append(np.full(len(dense), len(labels), dtype=np.int32))
            hn, street = tags.get("addr:housenumber", "").strip(), tags.get("addr:street", "").strip()
            labels.append(f"Blk {hn} {street}" if hn and street else "")
        log(f"  buildings with a height: {len(pts)} ({sum(1 for x in labels if x)} with an address)")
        return cls(np.concatenate(pts), np.concatenate(hs), np.concatenate(ids), labels)

    def save(self, path) -> None:
        np.savez_compressed(path, points=self.points.astype(np.float64), heights=self.heights.astype(np.float16),
                            ids=self.ids.astype(np.int32), labels=np.asarray(self.labels))

    @classmethod
    def load(cls, path) -> "Buildings":
        d = np.load(path)
        if "ids" in d:
            return cls(d["points"], d["heights"].astype(np.float32), d["ids"], [str(x) for x in d["labels"]])
        return cls(d["points"], d["heights"].astype(np.float32))


def fetch_hdb_storeys(log=print) -> dict[tuple[str, str], int]:
    """(block, street) -> number of storeys, from the same HDB dataset as the block ages."""
    return fetch_hdb_years(log, field="max_floor_lvl")


def building_height(tags: dict, storeys: dict) -> float | None:
    """Roof height: HDB's own storey count when the address is an HDB block (OSM levels are
    sometimes wrong, and a few metres decide between neighbouring blocks), else OSM."""
    key = (tags.get("addr:housenumber", "").strip().upper(), normalise_street(tags.get("addr:street", "")))
    if key in storeys:
        return storeys[key] * FLOOR_M
    try:
        if "height" in tags:
            return float(tags["height"].replace("m", "").strip())
        if "building:levels" in tags:
            return float(tags["building:levels"]) * FLOOR_M
    except ValueError:
        pass
    return None


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


def match(profile: np.ndarray, az: np.ndarray, el: np.ndarray, headings: np.ndarray | None = None):
    """(lowest median error in degrees, the camera heading that gives it (0 = north), the
    camera tilt: true elevation = elevation in the photo + tilt), over `headings`
    (default: all)."""
    HEADINGS_DEG = _HEADINGS if headings is None else headings
    bins = (np.round(HEADINGS_DEG[:, None] + az[None, :]).astype(int)) % AZ_BINS
    diff = profile[bins] - el[None, :]  # (headings, columns)
    costs = np.median(np.abs(diff[:, :, None] - PITCH_RANGE_DEG[None, None, :]), axis=1)
    h, t = np.unravel_index(costs.argmin(), costs.shape)
    return float(costs.min()), float(HEADINGS_DEG[h]), float(PITCH_RANGE_DEG[t])


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


def photo_hfov(path) -> float:
    """Horizontal field of view in degrees, from EXIF when the camera recorded it."""
    from PIL import Image

    return _exif_hfov(Image.open(path)) or 70.0


SIGN_TOLERANCE_DEG = 15.0  # the sign's spot vs the landmark's mapped point (a campus is ~100 m wide)
MIN_VIEW_STOREYS = 5  # elevated window views: low-rise buildings (carparks, shops) are not candidates


def viewpoint(buildings: Buildings, photo, lat: float, lon: float, radius_m: float = 600.0,
              landmark_x: float | None = None, landmark_y: float | None = None) -> dict | None:
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
    hfov = 2 * float(np.abs(az).max()) if isinstance(photo, tuple) else photo_hfov(photo)
    sign_el = None  # elevation of the sign in the photo, degrees above the image centre
    if landmark_y is not None and not isinstance(photo, tuple):
        from PIL import Image

        w, h = Image.open(photo).size
        vfov = 2 * np.degrees(np.arctan(np.tan(np.radians(hfov / 2)) * h / w))
        sign_el = (0.5 - landmark_y) * vfov
    cache: dict = {}

    def facing_landmark(la, lo):
        """Headings that keep the landmark (whose sign was read) inside the frame, or, when
        we know where across the photo the sign is, pointing that way."""
        bearing = np.degrees(np.arctan2((lon - lo) * COS_LAT, lat - la)) % 360
        if landmark_x is None:
            target, tol = bearing, hfov / 2 + 3
        else:
            target, tol = bearing - (landmark_x - 0.5) * hfov, SIGN_TOLERANCE_DEG
        off = (_HEADINGS - target + 180) % 360 - 180
        return _HEADINGS[np.abs(off) <= tol]

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

    blocks = buildings.addressed_near(lat, lon, radius_m)
    if blocks:
        scene = None if isinstance(photo, tuple) else photo_scene(photo, hfov)
        return _facade_search(buildings, blocks, az, el, facing_landmark, typical_from=(lat, lon, radius_m, cost, grid),
                              sign_el=sign_el, scene=scene)

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


FLOOR_TOLERANCE = 3  # floors: how far the skyline floor may be from the sign's height line


def _sign_height(buildings: Buildings, lat: float, lon: float) -> float:
    """Roughly how high a landmark's sign is: the top of its tallest building nearby."""
    _, hs = buildings.near(lat, lon, radius_m=60.0)
    return float(hs.max()) if len(hs) else 15.0


def _line_of_sight(buildings: Buildings, la, lo, eye, own, lat0, lon0, target_h, skip_end_m=40.0) -> bool:
    """True if nothing stands between the window and the landmark's sign: no building
    (other than our own, or the landmark's own buildings near the end) rises above the
    straight line from eye to sign."""
    d = float(np.hypot((lat0 - la) * M_PER_DEG, (lon0 - lo) * M_PER_DEG * COS_LAT))
    for t in np.arange(15.0, d - skip_end_m, 10.0):
        f = t / d
        sel, _ = buildings._select(la + (lat0 - la) * f, lo + (lon0 - lo) * f, 8.0)
        sel = sel[buildings.ids[sel] != own]
        if len(sel) and buildings.heights[sel].max() > eye + (target_h - eye) * f:
            return False
    return True


def _facade_search(buildings: Buildings, blocks: list[int], az, el, facing, typical_from, sign_el=None,
                   scene=None) -> dict | None:
    """Viewpoint search on the walls of addressed buildings (a window photo is taken from
    one): positions every ~16 m along each footprint, the building itself left out of the
    rendered skyline, and only floors it has."""
    lat0, lon0, radius_m, grid_cost, grid = typical_from

    def cost(la, lo, eye, own):
        """(score, heading, tilt): outline error, plus, for daytime photos, how badly the 3D
        view (which parts are built, which open) disagrees with the photo."""
        rel, hs = buildings.near(la, lo, exclude=own)
        if len(hs) < 50:
            return None
        c, heading, tilt = match(render(rel, hs, eye), az, el, facing(la, lo))
        if scene is not None:
            c += MASK_WEIGHT * (1 - view_overlap(scene, render_built(rel, hs, eye), heading, tilt))
        return c, heading, tilt

    cands = []  # (lat, lon, building, storeys)
    for b in blocks:
        pts, height = buildings.footprint(b)
        storeys = max(1, int(round(height / FLOOR_M)))
        if storeys < MIN_VIEW_STOREYS:
            continue
        cands += [(la, lo, b, storeys) for la, lo in pts[::2]]
    coarse = []
    for la, lo, b, n in cands:
        tries = sorted({max(1, round(n * q)) for q in (0.25, 0.5, 0.75, 1.0)})
        best = min((c for c in (cost(la, lo, f * FLOOR_M + EYE_ABOVE_FLOOR_M, b) for f in tries) if c), default=None)
        if best:
            coarse.append((best[0], la, lo, b, n))
    if len(coarse) < 5:
        return None
    coarse.sort(key=lambda x: x[0])
    # What a typical spot in the area scores, for comparison (open-grid positions).
    typical = []
    for la, lo in grid(lat0, lon0, radius_m, 100.0):
        c = grid_cost(la, lo, 25.0)
        if c:
            typical.append(c[0])
    typical_cost = float(np.median(typical)) if typical else float(np.median([c[0] for c in coarse]))

    # Refine the 10 best spots, plus the 2 best of each of the 6 best blocks, so the block
    # ranking compares every plausible block at its best floor.
    picks, per_block = list(coarse[:10]), {}
    for c in coarse:
        per_block.setdefault(c[3], []).append(c)
    for b in list(per_block)[:6]:
        picks += per_block[b][:2]
    results = []
    for _, la, lo, b, n in dict.fromkeys(picks):
        for f in range(1, n + 1):
            c = cost(la, lo, f * FLOOR_M + EYE_ABOVE_FLOOR_M, b)
            if c:
                results.append((c[0], la, lo, f, c[1], b, c[2]))
    results.sort(key=lambda x: x[0])
    # Draw lines to the landmark: (1) the sign's height in the photo fixes how high the
    # camera is for a given distance; (2) the sign must be visible from the window.
    if sign_el is not None:
        sign_h = _sign_height(buildings, lat0, lon0)
        kept = []
        for r in results:
            _, la, lo, f, _, b, tilt = r
            d = float(np.hypot((lat0 - la) * M_PER_DEG, (lon0 - lo) * M_PER_DEG * COS_LAT))
            eye_pred = sign_h - d * np.tan(np.radians(sign_el + tilt))
            if abs(f - (eye_pred - EYE_ABOVE_FLOOR_M) / FLOOR_M) > FLOOR_TOLERANCE:
                continue
            if not _line_of_sight(buildings, la, lo, f * FLOOR_M + EYE_ABOVE_FLOOR_M, b, lat0, lon0, sign_h):
                continue
            kept.append(r)
        if kept:
            results = kept
    best = results[0]
    close = [r for r in results if r[0] <= best[0] * 1.1 + 0.02]
    same = [r for r in close if r[5] == best[5]]
    # Map heights are good to a few metres, so the floor is a range of at least +-2.
    floors = sorted({r[3] for r in same} | {best[3] - FLOOR_SLACK, best[3] + FLOOR_SLACK})
    floors = [max(1, floors[0]), floors[-1]]
    spread = max(float(np.hypot((r[1] - best[1]) * M_PER_DEG, (r[2] - best[2]) * M_PER_DEG * COS_LAT)) for r in close)
    others = list(dict.fromkeys(buildings.labels[r[5]] for r in close if r[5] != best[5]))
    ranked, seen = [], set()  # best fit per block
    for r in results:
        if r[5] not in seen:
            seen.add(r[5])
            ranked.append({"block": buildings.labels[r[5]], "floor": r[3], "cost": r[0]})
    return {"lat": best[1], "lon": best[2], "floor": best[3], "heading": best[4], "cost": best[0],
            "typical_cost": typical_cost, "spread_m": spread, "floor_range": (floors[0], floors[-1]),
            "block": buildings.labels[best[5]], "other_blocks": others[:3], "ranked": ranked[:4]}


# --------------------------------------------------------------------------- 3D view: what is built where

PANO_STEP = 0.5  # degrees per panorama cell
PANO_EL_MIN = -60.0
MASK_WEIGHT = 1.0  # degrees of outline error one unit of view mismatch (1 - IoU) is worth
FLOOR_SLACK = 2  # floors either side of the best: map heights are only good to a few metres


def photo_scene(path, hfov_deg: float | None = None, width: int = 160) -> dict | None:
    """Daytime photo as a coarse grid of 'built' (walls, roofs, roads) vs 'open' (sky,
    vegetation) cells, with each column's azimuth and each row's elevation relative to
    the image centre. Columns where the top of the frame isn't sky (window frame) are
    dropped. None for night photos (colours unreliable) or without a usable sky."""
    from PIL import Image

    img = Image.open(path)
    hfov = hfov_deg or _exif_hfov(img) or 70.0
    h = int(round(img.height * width / img.width))
    rgb = np.asarray(img.convert("RGB").resize((width, h)), dtype=np.float32)
    grey = rgb.mean(axis=2)
    if np.median(grey[: max(2, h // 10)]) <= 90:
        return None
    first = _sky_boundary_day(grey)
    valid = (first > 2) & (first < h * 0.9)
    if valid.mean() < MIN_VALID_COLUMNS:
        return None
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    green = (g > r + 8) & (g > b + 4)
    rows = np.arange(h)[:, None]
    built = (rows > first[None, :]) & ~green
    cols = np.flatnonzero(valid)
    vfov = hfov * h / width
    return {"az": (cols + 0.5 - width / 2) / width * hfov, "el": (h / 2 - np.arange(h) - 0.5) / h * vfov,
            "built": built[:, cols]}


def render_built(rel_xy: np.ndarray, heights: np.ndarray, eye_m: float) -> np.ndarray:
    """Panorama (azimuth x elevation cells) of where buildings are seen from eye height:
    each wall point covers from its base to its roof, and nearby walls are widened to
    close the gaps between footprint points (~8 m apart). Union over buildings, so a short
    block in front and a tall one behind both count, and open ground between stays open."""
    n_az = int(360 / PANO_STEP)
    n_el = int(-2 * PANO_EL_MIN / PANO_STEP)
    d = np.hypot(rel_xy[:, 0], rel_xy[:, 1])
    ok = d >= MIN_DIST_M
    d, x, y, hh = d[ok], rel_xy[ok, 0], rel_xy[ok, 1], heights[ok]
    az = (np.degrees(np.arctan2(x, y)) % 360) / PANO_STEP
    lo = np.clip(((np.degrees(np.arctan2(-eye_m, d)) - PANO_EL_MIN) / PANO_STEP).astype(int), 0, n_el - 1)
    hi = np.clip(((np.degrees(np.arctan2(hh - eye_m, d)) - PANO_EL_MIN) / PANO_STEP).astype(int) + 1, 0, n_el)
    width = np.clip(np.ceil(np.degrees(8.0 / d) / PANO_STEP).astype(int), 1, 40)
    diff = np.zeros((n_az, n_el + 1), dtype=np.int32)
    keep = hi > lo
    for k in range(int(width.max()) if len(width) else 0):
        m = keep & (width > k)
        ai = (az[m].astype(int) + k - width[m] // 2) % n_az
        np.add.at(diff, (ai, lo[m]), 1)
        np.add.at(diff, (ai, hi[m]), -1)
    return np.cumsum(diff, axis=1)[:, :n_el] > 0


def view_overlap(scene: dict, pano: np.ndarray, heading: float, tilt: float) -> float:
    """Intersection-over-union of 'built' cells between the photo and a rendered view."""
    n_az, n_el = pano.shape
    ai = (np.round((heading + scene["az"]) / PANO_STEP).astype(int)) % n_az
    ei = np.round((scene["el"] + tilt - PANO_EL_MIN) / PANO_STEP).astype(int)
    inside = (ei >= 0) & (ei < n_el)
    rendered = np.zeros_like(scene["built"])
    rendered[inside] = pano[ai][:, ei[inside]].T  # (rows, columns), like the photo
    photo = scene["built"]
    union = (photo | rendered).sum()
    return float((photo & rendered).sum() / union) if union else 0.0
