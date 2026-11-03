"""Find where a photo was taken in two stages: the plan view first, the height second.

Stage 1 (plan view) uses only directions, which a photo gives reliably: each vertical
building edge in the photo is a direction left or right of the centre. From any candidate
spot on the map, the building outlines you would see first in each direction (nearer ones
hiding farther ones) give the same kind of list. The spot, heading and field of view that
line the two lists up win. A recognised landmark ("Lau Pa Sat is at column 545") pins
the heading and makes the search far sharper.

Stage 2 (height) runs only once the spot and heading are fixed: the eye height (floor) and
camera tilt are then the only unknowns, read from where the skyline sits in the photo.

Splitting the two keeps height, tilt and lens guesses (and buildings missing a height in
the map) from trading off against the position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .skyline import COS_LAT, FLOOR_M, M_PER_DEG, Buildings, photo_skyline

BIN_DEG = 0.2  # plan-view direction resolution
EDGE_TOL = 0.012  # an edge matches if within this share of the photo width
OWN_BUILDING_M = 4.0  # outlines this close belong to the building the photo is taken from
EYE_ABOVE_FLOOR_M = 1.5


def to_xy(lat, lon) -> np.ndarray:
    """Metres east/north on the same grid as Buildings.xy."""
    return np.stack([(np.asarray(lon) - 103.8) * M_PER_DEG * COS_LAT, (np.asarray(lat) - 1.35) * M_PER_DEG], -1)


def to_latlon(xy) -> tuple[float, float]:
    return float(1.35 + xy[1] / M_PER_DEG), float(103.8 + xy[0] / (M_PER_DEG * COS_LAT))


def _bearing(v) -> np.ndarray:
    v = np.asarray(v)
    return np.degrees(np.arctan2(v[..., 0], v[..., 1])) % 360


def _wrap(a):
    return (np.asarray(a) + 180) % 360 - 180


def x_from_angle(a_deg, hfov_deg: float):
    """Photo column (0..1 across) of a direction `a_deg` right of the photo centre."""
    return 0.5 + np.tan(np.radians(a_deg)) / (2 * np.tan(np.radians(hfov_deg / 2)))


def angle_from_x(x, hfov_deg: float):
    return np.degrees(np.arctan((np.asarray(x) - 0.5) * 2 * np.tan(np.radians(hfov_deg / 2))))


# --------------------------------------------------------------------------- photo side


def photo_edges(path, width: int = 600, sep: float = 0.012) -> np.ndarray:
    """Columns (0..1 across) of building outline edges in the photo.

    By day, the clearest edges are where the skyline jumps (one building ends, the next
    starts higher or lower); hazy towers against a pale sky have too little contrast for
    anything else. At night the sky outline is broken up by lit windows, so long vertical
    brightness steps (building sides against the dark sky or each other) are used."""
    from PIL import Image

    from .skyline import _sky_boundary_day, _sky_boundary_night

    img = Image.open(path).convert("L")
    h = int(round(img.height * width / img.width))
    a = np.asarray(img.resize((width, h)), dtype=np.float32)
    day = np.median(a[: max(2, h // 10)]) > 90
    cols = list(_contrast_edges(a))
    if day:
        cols = list(_skyline_jumps(a, _sky_boundary_day(a))) + cols
    picked: list[int] = []
    for col in cols:  # skyline jumps first (most reliable), then drop close neighbours
        if width * 0.01 < col < width * 0.99 and all(abs(col - p) > sep * width for p in picked):
            picked.append(int(col))
    return np.sort((np.array(picked, dtype=float) + 0.5) / width)


def _skyline_jumps(a: np.ndarray, first: np.ndarray, jump: float = 0.012) -> np.ndarray:
    """Columns where the sky outline steps up or down, biggest steps first."""
    h, width = a.shape
    f = first.astype(float)
    f[(first <= 2) | (first >= h * 0.9)] = np.nan
    pad = np.pad(f, 2, constant_values=np.nan)
    stack = np.stack([pad[i : i + width] for i in range(5)])
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fm = np.nanmedian(stack, axis=0)
    d = np.nan_to_num(np.r_[0, np.abs(fm[2:] - fm[:-2]), 0])
    cand = np.flatnonzero(d > jump * h)
    return cand[np.argsort(-d[cand])]


def _contrast_edges(a: np.ndarray, side: int = 4, smooth: int = 15, contrast: float = 12.0,
                    min_run: float = 0.12) -> np.ndarray:
    """Columns where the brightness to the left and right differs the same way over a long
    vertical stretch (averaged over `side` pixels each side and `smooth` rows, so a facade's
    window grid, which flips back and forth, does not count), longest first."""
    h, width = a.shape
    c = np.cumsum(np.pad(a, ((0, 0), (1, 0))), axis=1)
    i = np.arange(side, width - side)
    step = np.zeros_like(a)
    step[:, i] = (c[:, i + side + 1] - c[:, i + 1] - (c[:, i] - c[:, i - side])) / side  # right minus left
    cv = np.cumsum(np.pad(step, ((1, 0), (0, 0))), axis=0)
    sm = (cv[smooth:] - cv[:-smooth]) / smooth
    run = np.zeros(width)
    for m in (sm > contrast, sm < -contrast):  # one consistent direction of change
        cur = np.zeros(width)
        for row in m:
            cur = np.where(row, cur + 1, 0)
            run = np.maximum(run, cur)
    strength = np.abs(sm).mean(0)
    cand = np.flatnonzero(run >= min_run * h)
    return cand[np.lexsort((-strength[cand], -run[cand]))]


# --------------------------------------------------------------------------- map side


class Outlines:
    """Building footprints near an area, for plan-view sight lines."""

    def __init__(self, buildings: Buildings, lat: float, lon: float, radius_m: float, min_height: float = 20.0):
        sel, _ = buildings._select(lat, lon, radius_m)
        ids = buildings.ids[sel]
        keep = buildings.heights[sel] >= min_height
        sel, ids = sel[keep], ids[keep]
        order = np.argsort(ids, kind="stable")
        sel, ids = sel[order], ids[order]
        self.xy = buildings.xy[sel]
        self.ids = ids
        starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
        self.starts, self.ends = starts, np.r_[starts[1:], len(ids)]
        self.bid = ids[starts]

    def edges(self, cam: np.ndarray, max_dist: float = 900.0) -> np.ndarray:
        """Compass directions (degrees) where the nearest visible building changes, seen
        from `cam` (metres, Buildings.xy grid)."""
        if not len(self.starts):
            return np.zeros(0)
        rel = self.xy - cam
        d = np.hypot(rel[:, 0], rel[:, 1])
        dmin = np.minimum.reduceat(d, self.starts)
        n_pts = self.ends - self.starts
        cen = np.add.reduceat(rel, self.starts) / n_pts[:, None]
        mid = _bearing(cen)
        off = _wrap(_bearing(rel) - np.repeat(mid, n_pts))
        lo = mid + np.minimum.reduceat(off, self.starts)
        hi = mid + np.maximum.reduceat(off, self.starts)
        ok = (dmin >= OWN_BUILDING_M) & (dmin <= max_dist) & (hi - lo < 180)
        nbins = int(round(360 / BIN_DEG))
        who = np.full(nbins, -1)
        for k in np.flatnonzero(ok)[np.argsort(-dmin[ok])]:  # far first; nearer ones paint over
            i0, i1 = int(np.floor(lo[k] / BIN_DEG)), int(np.ceil(hi[k] / BIN_DEG))
            who[np.arange(i0, i1 + 1) % nbins] = k
        change = np.flatnonzero(who != np.roll(who, 1))
        return (change * BIN_DEG) % 360

    def extent(self, building: int, cam: np.ndarray) -> tuple[float, float, float] | None:
        """(left, centre, right) compass directions of one building seen from `cam`."""
        k = np.flatnonzero(self.bid == building)
        if not len(k):
            return None
        s, e = self.starts[k[0]], self.ends[k[0]]
        a = _bearing(self.xy[s:e] - cam)
        mid = _bearing(self.xy[s:e].mean(0) - cam)
        off = _wrap(a - mid)
        return float(mid + off.min()), float(mid), float(mid + off.max())


@dataclass
class Landmark:
    """Something recognised in the photo: its column (0..1 across) and where it is.
    `part` says which part of it sits at that column: its centre, left or right edge."""

    x: float
    lat: float
    lon: float
    part: str = "centre"
    building: int | None = None  # with a building id, left/right use its outline

    def direction(self, cam: np.ndarray, outlines: Outlines | None) -> float:
        if self.building is not None and outlines is not None:
            ext = outlines.extent(self.building, cam)
            if ext:
                return {"left": ext[0], "centre": ext[1], "right": ext[2]}[self.part]
        return float(_bearing(to_xy(self.lat, self.lon) - cam))


def edge_f1(photo_x: np.ndarray, map_x: np.ndarray, tol: float = EDGE_TOL) -> float:
    """How well two lists of edge columns line up (0-1): an F1 score where each match
    counts by closeness (Gaussian, sigma = tol / 2), so near-misses are graded rather than
    all-or-nothing and ties between candidate spots are rare."""
    if not len(photo_x) or not len(map_x):
        return 0.0
    w = np.exp(-0.5 * ((photo_x[:, None] - map_x[None, :]) / (tol / 2)) ** 2)
    rec, prec = w.max(1).mean(), w.max(0).mean()
    return float(2 * rec * prec / (rec + prec)) if rec + prec else 0.0


# --------------------------------------------------------------------------- stage 1


def _f1_all_headings(edges_az: np.ndarray, photo_x: np.ndarray, hfov: float) -> np.ndarray:
    """edge_f1 for every heading on the BIN_DEG grid at once (circular correlations by
    FFT). Same Gaussian closeness as edge_f1, with tolerance turned into degrees at the
    photo centre; overlapping matches are summed and capped at 1."""
    n = int(round(360 / BIN_DEG))
    if not len(edges_az) or not len(photo_x):
        return np.zeros(n)
    sigma = (EDGE_TOL / 2) * 2 * np.tan(np.radians(hfov / 2)) * 180 / np.pi / BIN_DEG  # in bins
    k = np.arange(-int(4 * sigma) - 1, int(4 * sigma) + 2)
    g = np.zeros(n)
    g[k % n] = np.exp(-0.5 * (k / sigma) ** 2)
    G = np.fft.rfft(g)
    m = np.zeros(n)
    np.add.at(m, np.round(np.asarray(edges_az) / BIN_DEG).astype(int) % n, 1.0)
    p = np.zeros(n)
    np.add.at(p, np.round(angle_from_x(photo_x, hfov) / BIN_DEG).astype(int) % n, 1.0)
    half = int(round(hfov / 2 / BIN_DEG))
    box = np.zeros(n)
    box[np.arange(-half, half + 1) % n] = 1.0
    M, P_, Bx = np.fft.rfft(m), np.fft.rfft(p), np.fft.rfft(box)
    corr = lambda A, Bf: np.fft.irfft(A * np.conj(Bf), n)  # out[h] = sum_j a[h + j] b[j]
    blurred_map = np.minimum(np.fft.irfft(M * G, n), 1.0)
    recall = corr(np.fft.rfft(blurred_map), P_) / len(photo_x)
    blurred_photo = np.minimum(np.fft.irfft(P_ * G, n), 1.0)
    count = np.maximum(corr(M, Bx), 1e-9)
    prec = np.minimum(corr(M, np.fft.rfft(blurred_photo)) / count, 1.0)
    recall = np.clip(recall, 0, 1)
    prec = np.where(count > 0.5, np.clip(prec, 0, 1), 0.0)
    return np.where(recall + prec > 0, 2 * recall * prec / np.maximum(recall + prec, 1e-9), 0.0)


def _score_spot(outlines: Outlines, cam, photo_x, landmarks, hfovs, headings, lm_scale):
    """Best (score, heading, hfov, f1, landmark error) at one spot. `headings=None` tries
    every heading (BIN_DEG apart) in one go."""
    edges = outlines.edges(cam)
    best = (-np.inf, 0.0, 0.0, 0.0, 0.0)
    lm_dirs = [lm.direction(cam, outlines) for lm in landmarks]
    if not landmarks and headings is None:
        for hfov in hfovs:
            f1 = _f1_all_headings(edges, photo_x, hfov)
            i = int(np.argmax(f1))
            if f1[i] > best[0]:
                best = (float(f1[i]), float(i * BIN_DEG), float(hfov), float(f1[i]), 0.0)
        return best
    for hfov in hfovs:
        if landmarks:
            hs = [lm_dirs[0] - angle_from_x(landmarks[0].x, hfov)]
        else:
            hs = headings
        for hd in hs:
            lm_err = 0.0
            if landmarks:
                xs = [x_from_angle(_wrap(dr - hd), hfov) for dr in lm_dirs]
                lm_err = float(np.mean([abs(x - lm.x) for x, lm in zip(xs, landmarks)]))
            rel = _wrap(edges - hd)
            map_x = x_from_angle(rel[np.abs(rel) < hfov / 2], hfov)
            f1 = edge_f1(photo_x, map_x)
            score = f1 - lm_err / lm_scale
            if score > best[0]:
                best = (score, float(hd % 360), float(hfov), f1, lm_err)
    return best


def resect(buildings: Buildings, photo_x, lat: float, lon: float, radius_m: float = 300.0,
           landmarks: list[Landmark] | None = None, hfovs=None, step_m: float = 4.0,
           min_height: float = 20.0, top: int = 5, window_min_height: float | None = None,
           log=None) -> list[dict]:
    """Stage 1: the spots (within `radius_m` of lat, lon) whose plan-view outline edges best
    line up with `photo_x`, plus any landmark directions. Returns the best `top` distinct
    spots, each refined, with lat, lon, heading, hfov, the edge match (0-1) and the mean
    landmark error (share of the photo width). With `window_min_height`, only spots on the
    outline of buildings at least that tall are tried: a view from a high window."""
    landmarks = landmarks or []
    photo_x = np.asarray(photo_x, float)
    hfovs = list(hfovs) if hfovs is not None else list(np.arange(45, 96, 5.0))
    outlines = Outlines(buildings, lat, lon, radius_m + 900, min_height)
    c = to_xy(lat, lon)
    k = int(radius_m // step_m)
    if window_min_height is not None:
        sel, _ = buildings._select(lat, lon, radius_m)
        sel = sel[buildings.heights[sel] >= window_min_height]
        cells = np.unique(np.floor(buildings.xy[sel] / step_m).astype(int), axis=0, return_index=True)[1]
        grid = [buildings.xy[i] for i in sel[np.sort(cells)]]  # facade points, about one per step
    else:
        grid = [c + (i * step_m, j * step_m) for i in range(-k, k + 1) for j in range(-k, k + 1)
                if (i * i + j * j) * step_m * step_m <= radius_m * radius_m]
    headings = None  # every heading, scored at once
    lm_scale = 0.05  # a landmark 5% of the width off costs as much as the whole edge match
    scored = []
    for n, cam in enumerate(grid):
        s = _score_spot(outlines, cam, photo_x, landmarks, hfovs, headings, lm_scale)
        scored.append((s[0], n, s))
        if log and n % 2000 == 0:
            log(f"  {n}/{len(grid)} spots")
    scored.sort(key=lambda r: -r[0])
    out: list[dict] = []
    for _, n, s in scored:
        if any(np.hypot(*(grid[n] - to_xy(o["lat"], o["lon"]))) < 4 * step_m for o in out):
            continue
        if window_min_height is not None:  # stay on the facade: refine heading and view only
            cam = np.asarray(grid[n], float)
            best = _score_spot(outlines, cam, photo_x, landmarks, np.arange(s[2] - 4, s[2] + 4.1, 1.0),
                               None if not landmarks else np.arange(s[1] - 2, s[1] + 2.01, 0.25), lm_scale)
        else:
            cam, best = _refine(outlines, grid[n], s, photo_x, landmarks, step_m, lm_scale)
        la, lo = to_latlon(cam)
        out.append({"lat": round(la, 6), "lon": round(lo, 6), "heading": round(best[1], 1), "hfov": round(best[2], 1),
                    "edge_match": round(best[3], 3), "landmark_error": round(best[4], 4), "score": round(best[0], 3)})
        if len(out) == 3 * top:
            break
    out = sorted(out, key=lambda o: -o["score"])[:top]
    all_scores = np.array([r[0] for r in scored])
    for o in out:
        o["typical_score"] = round(float(np.median(all_scores)), 3)
    return out


def _refine(outlines, cam, best, photo_x, landmarks, step_m, lm_scale):
    """Local search around a grid winner: half-size spot steps, finer heading and hfov."""
    cam = np.asarray(cam, float)
    for scale in (0.5, 0.25):
        moved = True
        while moved:
            moved = False
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)):
                cand = cam + np.array([dx, dy]) * step_m * scale
                hfovs = np.arange(best[2] - 4, best[2] + 4.1, 1.0)
                headings = np.arange(best[1] - 2, best[1] + 2.01, 0.25)
                s = _score_spot(outlines, cand, photo_x, landmarks, hfovs, headings, lm_scale)
                if s[0] > best[0] + 1e-9:
                    best, cam, moved = s, cand, (dx, dy) != (0, 0)
    return cam, best


# --------------------------------------------------------------------------- stage 2


def fit_height(buildings: Buildings, photo, lat: float, lon: float, heading: float, hfov: float,
               eyes=None, max_tilt: float = 15.0) -> dict | None:
    """Stage 2: with the spot and heading fixed, the eye height (and camera tilt) that best
    puts the map's building tops on the photo's skyline. None without a usable skyline."""
    sky = photo_skyline(photo, hfov)
    if sky is None:
        return None
    az, el = sky
    eyes = np.arange(1.5, 150, 1.5) if eyes is None else np.asarray(eyes, float)
    sel, p = buildings._select(lat, lon, 1500)
    rel = buildings.xy[sel] - p
    d = np.hypot(rel[:, 0], rel[:, 1])
    hs = buildings.heights[sel]
    own = set(buildings.ids[sel][d < OWN_BUILDING_M].tolist())
    if own:  # the photo is taken from this building: the eye is at most at its roof
        roof = float(buildings.heights[sel][d < OWN_BUILDING_M].max())
        eyes = eyes[eyes <= roof + 2.0]
    keep = (d > OWN_BUILDING_M) & ~np.isin(buildings.ids[sel], list(own))
    rel, d, hs = rel[keep], d[keep], hs[keep]
    a = _wrap(_bearing(rel) - heading)
    cols = np.round(a / 0.5).astype(int)
    want = np.round(az / 0.5).astype(int)
    rows = []
    for eye in eyes:
        e = np.degrees(np.arctan2(hs - eye, d))
        top = {}
        m = np.isin(cols, want)
        for c_, v in zip(cols[m], e[m]):
            if v > top.get(c_, -90):
                top[c_] = v
        mp = np.array([top.get(c_, 0.0) for c_ in want])  # open sky: the horizon
        tilt = float(np.clip(np.median(mp - el), -max_tilt, max_tilt))
        rows.append((float(np.median(np.abs(mp - el - tilt))), float(eye), tilt))
    rows.sort()
    err, eye, tilt = rows[0]
    near = [r for r in rows if r[0] <= err + 0.25]
    return {"reliable": err < 1.5, "eye_m": eye, "floor": max(1, int(round((eye - EYE_ABOVE_FLOOR_M) / FLOOR_M)) + 1), "tilt_deg": round(tilt, 2),
            "error_deg": round(err, 3), "eye_range_m": [min(r[1] for r in near), max(r[1] for r in near)]}
