"""Map matching: compare what the photo shows with what the map says is at each spot.

A person locating "school with a field, next to a carpark with solar panels, newer HDB
blocks" thinks: where does that combination exist? This module does the same:

1. `MapGrid`: Singapore cut into ~55 m cells; for each cell, which features are within a
   short walk (school, sports pitch, multi-storey carpark, MRT track, expressway, canal,
   mall, HDB blocks of each era, landed houses, ...). Built from OpenStreetMap (ODbL) and
   HDB's property dataset (data.gov.sg, Singapore Open Data Licence) for block ages.
2. `MapMatcher`: logistic regressions that predict, from the photo's StreetCLIP embedding,
   which features are nearby. They are trained on the reference index, whose photos have
   known locations, so the labels come free from the grid.
3. Every cell is scored by how well its features agree with the predictions (naive Bayes,
   tempered by `lam`), added to the area classifier's log-probability, and the best cell
   wins. `lam` and `alpha` are tuned on held-out capture sessions.
"""

from __future__ import annotations

import numpy as np
import requests

from .roads import HEADERS, OVERPASS_URLS

STEP = 0.0005  # grid cell, degrees (~55 m)
BOUNDS = (1.15, 103.59, 1.48, 104.10)  # lat_min, lon_min, lat_max, lon_max
M_PER_DEG = 111_000.0

# (name, Overpass statements, radius in metres). Area features are filled, others are points.
FEATURES = [
    ("school", 'nwr["amenity"~"^(school|college)$"]', 150),
    ("sports_pitch", 'nwr["leisure"~"^(pitch|sports_centre|stadium|track)$"]', 150),
    ("playground", 'nwr["leisure"="playground"]', 100),
    ("multistorey_carpark", 'nwr["amenity"="parking"]["parking"="multi-storey"]', 150),
    ("park", 'nwr["leisure"~"^(park|garden|nature_reserve)$"]', 150),
    ("forest", 'nwr["natural"="wood"];nwr["landuse"="forest"]', 100),
    ("water", 'way["waterway"~"^(canal|river)$"];nwr["natural"="water"]', 150),
    ("rail", 'way["railway"~"^(subway|light_rail|rail)$"]["tunnel"!="yes"]', 200),
    ("expressway", 'way["highway"="motorway"]', 200),
    ("mall", 'nwr["shop"="mall"]', 200),
    ("hawker", 'nwr["amenity"~"^(food_court|marketplace)$"]', 150),
    ("worship", 'nwr["amenity"="place_of_worship"]', 150),
    ("tower_block", 'way["building"]["building:levels"~"^([1-9][0-9])$"]', 120),
    ("landed_house", 'way["building"~"^(house|detached|semidetached_house|terrace|bungalow)$"]', 120),
    ("industrial", 'nwr["landuse"="industrial"]', 100),
    ("construction", 'nwr["landuse"="construction"]', 100),
]
HDB_ERAS = [("hdb_pre1985", 0, 1984), ("hdb_1985_1999", 1985, 1999), ("hdb_2000_2014", 2000, 2014), ("hdb_2015on", 2015, 9999)]
HDB_RADIUS = 150
FEATURE_NAMES = [f[0] for f in FEATURES] + [e[0] for e in HDB_ERAS]
LAND_RADIUS = 250  # cells further than this from any named road are sea or forest interior

HDB_DATASET = "d_17f5382f26140b1fdae0ba2ef6239d2f"  # HDB Property Information
HDB_URL = "https://data.gov.sg/api/action/datastore_search"
# OSM spells street words out; HDB abbreviates them.
STREET_ABBR = {
    "AVENUE": "AVE", "STREET": "ST", "ROAD": "RD", "DRIVE": "DR", "CRESCENT": "CRES", "CENTRAL": "CTRL",
    "NORTH": "NTH", "SOUTH": "STH", "CLOSE": "CL", "PLACE": "PL", "TERRACE": "TER", "LORONG": "LOR",
    "JALAN": "JLN", "BUKIT": "BT", "UPPER": "UPP", "KAMPONG": "KG", "TANJONG": "TG", "HEIGHTS": "HTS",
    "GARDENS": "GDNS", "PARK": "PK", "COMMONWEALTH": "C'WEALTH", "MARKET": "MKT", "SAINT": "ST.",
}


def _overpass(query: str, log=print) -> dict:
    errors = []
    for url in OVERPASS_URLS:
        try:
            r = requests.post(url, data={"data": query}, headers=HEADERS, timeout=900)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            errors.append(f"{url}: {exc}")
            log(f"  {errors[-1]}")
    raise RuntimeError("Overpass failed: " + "; ".join(errors))


def _query(statements: str, out: str = "geom") -> str:
    body = "".join(f"{s.strip()}(area.sg);" for s in statements.split(";") if s.strip())
    return f'[out:json][timeout:600];area["ISO3166-1"="SG"][admin_level=2]->.sg;({body});out {out};'


def _shapes(element: dict) -> list[np.ndarray]:
    """(lat, lon) arrays: one per ring/line of a way or relation, or a single point."""
    if "geometry" in element:
        return [np.array([[p["lat"], p["lon"]] for p in element["geometry"]])]
    if "members" in element:
        return [np.array([[p["lat"], p["lon"]] for p in m["geometry"]]) for m in element["members"] if m.get("geometry")]
    if "lat" in element:
        return [np.array([[element["lat"], element["lon"]]])]
    if "center" in element:
        return [np.array([[element["center"]["lat"], element["center"]["lon"]]])]
    return []


def normalise_street(name: str) -> str:
    return " ".join(STREET_ABBR.get(w, w) for w in name.upper().replace(".", "").split())


def fetch_hdb_years(log=print) -> dict[tuple[str, str], int]:
    """(block, street) -> year completed, from HDB's property dataset."""
    out, offset = {}, 0
    while True:
        r = requests.get(HDB_URL, params={"resource_id": HDB_DATASET, "limit": 5000, "offset": offset},
                         headers=HEADERS, timeout=120)
        r.raise_for_status()
        recs = r.json()["result"]["records"]
        for rec in recs:
            try:
                out[(rec["blk_no"].strip().upper(), normalise_street(rec["street"]))] = int(rec["year_completed"])
            except (KeyError, ValueError):
                pass
        offset += len(recs)
        if not recs or offset >= r.json()["result"].get("total", 0):
            break
    log(f"  HDB dataset: {len(out)} blocks")
    return out


class MapGrid:
    """Boolean feature presence for every grid cell (rows = latitude, cols = longitude)."""

    def __init__(self, features: np.ndarray, land: np.ndarray, names=FEATURE_NAMES):
        self.features = features  # (n_features, rows, cols) bool
        self.land = land  # (rows, cols) bool
        self.names = list(names)

    @property
    def shape(self):
        return self.land.shape

    @staticmethod
    def rc(lat, lon):
        return (np.floor((np.asarray(lat) - BOUNDS[0]) / STEP).astype(int),
                np.floor((np.asarray(lon) - BOUNDS[1]) / STEP).astype(int))

    @staticmethod
    def centre(r, c):
        return BOUNDS[0] + (np.asarray(r) + 0.5) * STEP, BOUNDS[1] + (np.asarray(c) + 0.5) * STEP

    def at(self, lat, lon) -> np.ndarray:
        """(n, n_features) features at the given points (all False outside the grid)."""
        r, c = self.rc(lat, lon)
        ok = (r >= 0) & (r < self.shape[0]) & (c >= 0) & (c < self.shape[1])
        out = np.zeros((len(r), len(self.names)), dtype=bool)
        out[ok] = self.features[:, r[ok], c[ok]].T
        return out

    @classmethod
    def build(cls, roads, log=print) -> "MapGrid":
        rows = int(np.ceil((BOUNDS[2] - BOUNDS[0]) / STEP))
        cols = int(np.ceil((BOUNDS[3] - BOUNDS[1]) / STEP))
        layers = []
        for name, statements, radius in FEATURES:
            data = _overpass(_query(statements), log)
            seed = np.zeros((rows, cols), dtype=bool)
            n = 0
            for el in data.get("elements", []):
                for shape in _shapes(el):
                    _rasterise(seed, shape)
                    n += 1
            layers.append(_dilate(seed, radius))
            log(f"  {name}: {n} shapes, near {layers[-1].mean():.1%} of the grid")
        layers += cls._hdb_layers(rows, cols, log)
        road_seed = np.zeros((rows, cols), dtype=bool)
        for lines in roads.roads.values():
            for line in lines:
                _rasterise(road_seed, np.asarray(line), fill=False)
        land = _dilate(road_seed, LAND_RADIUS)
        log(f"  land cells: {land.sum()}")
        return cls(np.stack(layers), land)

    @staticmethod
    def _hdb_layers(rows, cols, log) -> list[np.ndarray]:
        layers = [np.zeros((rows, cols), dtype=bool) for _ in HDB_ERAS]
        try:
            years = fetch_hdb_years(log)
            data = _overpass(_query('way["building"]["addr:housenumber"]["addr:street"]', out="center tags"), log)
        except Exception as exc:  # optional: the other features still work without block ages
            log(f"  ::warning::HDB block ages unavailable ({exc}); skipping era features")
            return layers
        matched = 0
        for el in data.get("elements", []):
            t = el.get("tags", {})
            year = years.get((t["addr:housenumber"].strip().upper(), normalise_street(t["addr:street"])))
            if year is None or "center" not in el:
                continue
            for layer, (_, lo, hi) in zip(layers, HDB_ERAS):
                if lo <= year <= hi:
                    _rasterise(layer, np.array([[el["center"]["lat"], el["center"]["lon"]]]))
            matched += 1
        log(f"  HDB blocks placed on the map: {matched} of {len(years)}")
        return [_dilate(layer, HDB_RADIUS) for layer in layers]

    def save(self, path) -> None:
        np.savez_compressed(path, features=np.packbits(self.features, axis=-1), land=np.packbits(self.land, axis=-1),
                            cols=np.asarray(self.land.shape[1]), names=np.asarray(self.names))

    @classmethod
    def load(cls, path) -> "MapGrid":
        d = np.load(path)
        cols = int(d["cols"])
        return cls(np.unpackbits(d["features"], axis=-1, count=cols).astype(bool),
                   np.unpackbits(d["land"], axis=-1, count=cols).astype(bool), [str(n) for n in d["names"]])


def _rasterise(grid: np.ndarray, shape: np.ndarray, fill: bool = True) -> None:
    """Mark the cells a point, line or (closed, if `fill`) polygon covers."""
    if len(shape) == 0:
        return
    # Densify so consecutive points are at most half a cell apart.
    pts = [shape[:1]]
    for a, b in zip(shape[:-1], shape[1:]):
        n = max(1, int(np.ceil(np.abs(b - a).max() / (STEP / 2))))
        pts.append(a + (b - a) * (np.arange(1, n + 1)[:, None] / n))
    pts = np.concatenate(pts)
    r, c = MapGrid.rc(pts[:, 0], pts[:, 1])
    ok = (r >= 0) & (r < grid.shape[0]) & (c >= 0) & (c < grid.shape[1])
    grid[r[ok], c[ok]] = True
    closed = len(shape) > 3 and np.allclose(shape[0], shape[-1])
    if fill and closed and ok.any():
        r0, r1, c0, c1 = r[ok].min(), r[ok].max(), c[ok].min(), c[ok].max()
        rr, cc = np.mgrid[r0 : r1 + 1, c0 : c1 + 1]
        lat, lon = MapGrid.centre(rr.ravel(), cc.ravel())
        inside = _point_in_polygon(lat, lon, shape)
        grid[rr.ravel()[inside], cc.ravel()[inside]] = True


def _point_in_polygon(lat: np.ndarray, lon: np.ndarray, poly: np.ndarray) -> np.ndarray:
    inside = np.zeros(len(lat), dtype=bool)
    y, x = poly[:, 0], poly[:, 1]
    for i in range(len(poly) - 1):
        y1, x1, y2, x2 = y[i], x[i], y[i + 1], x[i + 1]
        crosses = (y1 > lat) != (y2 > lat)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (lon < xi)
    return inside


def _dilate(seed: np.ndarray, radius_m: float) -> np.ndarray:
    """Cells within `radius_m` of any marked cell."""
    k = int(np.ceil(radius_m / (STEP * M_PER_DEG)))
    out = seed.copy()
    rows, cols = seed.shape
    for dr in range(-k, k + 1):
        for dc in range(-k, k + 1):
            if (dr * dr + dc * dc) * (STEP * M_PER_DEG) ** 2 > radius_m**2 or (dr == 0 and dc == 0):
                continue
            src = seed[max(0, -dr) : rows - max(0, dr), max(0, -dc) : cols - max(0, dc)]
            out[max(0, dr) : rows - max(0, -dr), max(0, dc) : cols - max(0, -dc)] |= src
    return out


# --------------------------------------------------------------------------- matcher


class MapMatcher:
    """Predicts nearby map features from an embedding, then scores grid cells."""

    def __init__(self, grid: MapGrid, W=None, b=None, mu=None, sd=None, prior=None, lam=0.5, alpha=1.0):
        self.grid = grid
        self.W, self.b, self.mu, self.sd, self.prior = W, b, mu, sd, prior
        self.lam, self.alpha = lam, alpha
        self._cells = None

    def fit(self, X: np.ndarray, Y: np.ndarray, l2: float = 1e-2, steps: int = 400, lr: float = 0.01, log=None):
        """Independent logistic regressions (one per feature), trained together with Adam."""
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-6
        Xs = ((X - self.mu) / self.sd).astype(np.float32)
        Y = Y.astype(np.float32)
        n, d = Xs.shape
        self.prior = np.clip(Y.mean(axis=0), 1e-3, 1 - 1e-3)
        W = np.zeros((d, Y.shape[1]), dtype=np.float32)
        b = np.log(self.prior / (1 - self.prior)).astype(np.float32)
        state = [np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b)]
        for t in range(1, steps + 1):
            p = _sigmoid(Xs @ W + b)
            g = (p - Y) / n
            grads = (Xs.T @ g + l2 * W, g.sum(axis=0))
            for i, (param, grad) in enumerate(zip((W, b), grads)):
                m, v = state[2 * i], state[2 * i + 1]
                m *= 0.9
                m += 0.1 * grad
                v *= 0.999
                v += 0.001 * grad**2
                param -= lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
            if log and t % 100 == 0:
                loss = -(Y * np.log(p + 1e-9) + (1 - Y) * np.log(1 - p + 1e-9)).mean()
                log(f"  map features: step {t}, loss {loss:.4f}")
        self.W, self.b = W, b
        return self

    def feature_probs(self, X: np.ndarray) -> np.ndarray:
        Xs = ((np.atleast_2d(X) - self.mu) / self.sd).astype(np.float32)
        return _sigmoid(Xs @ self.W + self.b)

    @property
    def cells(self):
        """(row, col, features) of every land cell, cached."""
        if self._cells is None:
            r, c = np.nonzero(self.grid.land)
            self._cells = (r, c, self.grid.features[:, r, c].T.astype(np.float32))
        return self._cells

    def map_loglik(self, probs: np.ndarray) -> np.ndarray:
        """(n_photos, n_land_cells) naive-Bayes log-likelihood ratio of each cell."""
        p = np.clip(probs, 0.02, 0.98)
        pos = np.log(p / self.prior)  # feature present at the cell
        neg = np.log((1 - p) / (1 - self.prior))  # feature absent
        _, _, F = self.cells
        return (pos - neg) @ F.T + neg.sum(axis=1, keepdims=True)

    def cell_geocell_logp(self, geocell, X: np.ndarray) -> np.ndarray:
        """(n_photos, n_land_cells) area classifier log-probability of each fine cell."""
        if getattr(self, "_geo_idx", (None,))[0] is not geocell:
            r, c, _ = self.cells
            lat, lon = MapGrid.centre(r, c)
            keys = np.floor(geocell.centroids / geocell.cell_deg + 1e-9).astype(np.int64)
            lookup = {tuple(k): i for i, k in enumerate(keys)}
            cell_keys = np.floor(np.stack([lat, lon], axis=1) / geocell.cell_deg).astype(np.int64)
            self._geo_idx = (geocell, np.array([lookup.get(tuple(k), -1) for k in cell_keys]))
        idx = self._geo_idx[1]
        logp = np.log(geocell.predict_proba(X) + 1e-9)
        floor = logp.min(axis=1, keepdims=True) - 2.0  # cells with no training photos: unlikely, not impossible
        return np.where(idx[None, :] >= 0, logp[:, np.maximum(idx, 0)], floor).astype(np.float32)

    def scores(self, X: np.ndarray, geocell=None, lam=None, alpha=None) -> np.ndarray:
        """(n_photos, n_land_cells) combined score; higher = more likely."""
        lam = self.lam if lam is None else lam
        alpha = self.alpha if alpha is None else alpha
        X = np.atleast_2d(X)
        score = lam * self.map_loglik(self.feature_probs(X))
        if geocell is not None:
            score += alpha * self.cell_geocell_logp(geocell, X)
        return score

    def best(self, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Best cell per row of `score`: (coords (n, 2), confidence = posterior mass within ~300 m)."""
        r, c, _ = self.cells
        best = score.argmax(axis=1)
        post = np.exp(score - score.max(axis=1, keepdims=True))
        post /= post.sum(axis=1, keepdims=True)
        near = (np.abs(r[None, :] - r[best][:, None]) <= 5) & (np.abs(c[None, :] - c[best][:, None]) <= 5)
        lat, lon = MapGrid.centre(r[best], c[best])
        return np.stack([lat, lon], axis=1), (post * near).sum(axis=1)

    def locate(self, X: np.ndarray, geocell=None) -> tuple[np.ndarray, np.ndarray]:
        return self.best(self.scores(X, geocell))

    def save(self, path) -> None:
        """One file with the trained weights and the map grid they score."""
        g = self.grid
        np.savez_compressed(path, W=self.W, b=self.b, mu=self.mu, sd=self.sd, prior=self.prior,
                            lam=np.asarray(self.lam), alpha=np.asarray(self.alpha), names=np.asarray(g.names),
                            features=np.packbits(g.features, axis=-1), land=np.packbits(g.land, axis=-1),
                            cols=np.asarray(g.land.shape[1]))

    @classmethod
    def load(cls, path) -> "MapMatcher":
        d = np.load(path)
        cols = int(d["cols"])
        grid = MapGrid(np.unpackbits(d["features"], axis=-1, count=cols).astype(bool),
                       np.unpackbits(d["land"], axis=-1, count=cols).astype(bool), [str(n) for n in d["names"]])
        return cls(grid, d["W"], d["b"], d["mu"], d["sd"], d["prior"], float(d["lam"]), float(d["alpha"]))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train_and_evaluate(index_path, grid: MapGrid, geocell_cell: float = 0.01, log=print) -> tuple[MapMatcher, dict]:
    """Train on the index's training sequences; tune lam/alpha on half the held-out
    sequences and report accuracy on the other half (geocell alone vs with the map)."""
    from .evaluate import _errors, is_test, metrics, split
    from .geocell import GeoCellClassifier

    data = np.load(index_path)
    X, coords, seqs = data["embeddings"].astype(np.float32), data["coords"], data["sequences"]
    Y = grid.at(coords[:, 0], coords[:, 1])
    test = split(seqs)
    # Halve the held-out sequences with a different hash (the test split already fixes crc32 % 1000).
    val = test & np.array([is_test(f"{s}#tune", str(i), 0.5) for i, s in enumerate(seqs)])
    hold = test & ~val
    log(f"Train {(~test).sum()}, tune {val.sum()}, test {hold.sum()} photos")

    matcher = MapMatcher(grid).fit(X[~test], Y[~test], log=log)
    geocell = GeoCellClassifier(cell_deg=geocell_cell).fit(X[~test], coords[~test], log=log)

    # How well each feature is predicted (AUC on held-out photos).
    probs = matcher.feature_probs(X[test])
    auc = {n: _auc(Y[test][:, j], probs[:, j]) for j, n in enumerate(grid.names)}
    log("Feature AUC: " + ", ".join(f"{n} {a:.2f}" for n, a in auc.items()))

    combos = [(lam, alpha) for lam in (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0) for alpha in (0.5, 1.0, 2.0)]

    def run(mask):
        """Errors and confidences for every (lam, alpha), in chunks to bound memory."""
        idx = np.flatnonzero(mask)
        errs = {k: [] for k in combos}
        confs = {k: [] for k in combos}
        for s in range(0, len(idx), 50):
            chunk = idx[s : s + 50]
            ll = matcher.map_loglik(matcher.feature_probs(X[chunk]))
            lp = matcher.cell_geocell_logp(geocell, X[chunk])
            for lam, alpha in combos:
                pred, conf = matcher.best(lam * ll + alpha * lp)
                errs[(lam, alpha)].append(_errors(pred, coords[chunk]))
                confs[(lam, alpha)].append(conf)
        return {k: np.concatenate(v) for k, v in errs.items()}, {k: np.concatenate(v) for k, v in confs.items()}

    val_err, _ = run(val)
    for k in combos:
        log(f"  tune lam={k[0]} alpha={k[1]}: median {np.median(val_err[k]):.2f} km")
    matcher.lam, matcher.alpha = min(combos, key=lambda k: np.median(val_err[k]))
    hold_err, hold_conf = run(hold)
    geo_only = hold_err[(0.0, 1.0)]
    combined, conf = hold_err[(matcher.lam, matcher.alpha)], hold_conf[(matcher.lam, matcher.alpha)]
    methods = {"geocell only": metrics(geo_only), f"geocell + map (lam={matcher.lam}, alpha={matcher.alpha})": metrics(combined)}
    for t in (0.3, 0.5):
        sure = conf >= t
        if sure.any():
            methods[f"geocell + map (conf>={t})"] = metrics(combined[sure]) | {"coverage": float(sure.mean())}
    # Refit on everything for release.
    matcher = MapMatcher(grid, lam=matcher.lam, alpha=matcher.alpha).fit(X, Y)
    return matcher, {"methods": methods, "auc": auc, "n_test": int(hold.sum())}


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def report_markdown(result: dict) -> str:
    from .evaluate import THRESHOLDS_KM

    cols = ["n", "median_km"] + [f"within_{t}km" for t in THRESHOLDS_KM]
    lines = ["## Map matching on held-out Mapillary sequences", "",
             "| method | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, m in result["methods"].items():
        n = f"{m['n']} ({m['coverage']:.0%})" if "coverage" in m else str(m["n"])
        lines.append(f"| {name} | {n} | {m['median_km']:.2f} | " + " | ".join(f"{m[f'within_{t}km']:.1%}" for t in THRESHOLDS_KM) + " |")
    lines += ["", "How well each map feature is predicted from the photo (AUC, 0.5 = chance):", "",
              "| feature | AUC |", "|---|---|"]
    lines += [f"| {n} | {a:.2f} |" for n, a in sorted(result["auc"].items(), key=lambda x: -np.nan_to_num(x[1]))]
    return "\n".join(lines) + "\n"


def describe(probs: np.ndarray, prior: np.ndarray, names: list[str], top: int = 4) -> list[str]:
    """The features the photo most clearly shows (relative to how common they are)."""
    lift = np.log(np.clip(probs, 1e-3, 1) / prior)
    return [f"map: {names[j].replace('_', ' ')} nearby ({probs[j]:.0%})" for j in np.argsort(-lift)[:top] if lift[j] > 0.3]
