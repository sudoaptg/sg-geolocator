import numpy as np

from sg_geolocator.mapmatch import STEP, MapGrid, MapMatcher, _dilate, _rasterise, normalise_street


def _blank():
    return np.zeros((660, 1020), dtype=bool)


def test_rasterise_fills_closed_polygons_only():
    lat, lon = 1.35, 103.80
    d = 5 * STEP
    square = np.array([[lat, lon], [lat, lon + d], [lat + d, lon + d], [lat + d, lon], [lat, lon]])
    g = _blank()
    _rasterise(g, square)
    r, c = MapGrid.rc(lat + d / 2, lon + d / 2)
    assert g[r, c]  # interior filled
    line = _blank()
    _rasterise(line, square[:3])  # open line: outline only
    assert not line[r, c] and line.sum() > 5


def test_dilate_reaches_radius():
    g = _blank()
    g[300, 500] = True
    out = _dilate(g, 150)
    assert out[300, 502] and not out[300, 505]


def test_street_abbreviations_match_hdb():
    assert normalise_street("Ang Mo Kio Avenue 3") == "ANG MO KIO AVE 3"
    assert normalise_street("Bukit Batok Street 21") == "BT BATOK ST 21"


def test_matcher_finds_the_cell_with_the_shown_features():
    rng = np.random.default_rng(0)
    rows, cols, nf = 40, 40, 6
    feats = rng.random((nf, rows, cols)) < 0.3
    land = np.ones((rows, cols), dtype=bool)
    grid = MapGrid(feats, land, [f"f{i}" for i in range(nf)])
    # Embeddings encode the cell's features plus noise.
    r = rng.integers(0, rows, 3000)
    c = rng.integers(0, cols, 3000)
    Y = feats[:, r, c].T
    X = np.hstack([Y * 2.0 + rng.normal(0, 0.5, Y.shape), rng.normal(0, 1, (3000, 10))]).astype(np.float32)
    m = MapMatcher(grid, lam=1.0).fit(X, Y)
    rr, cc, F = m.cells
    probs = m.feature_probs(X[:200])
    ll = m.map_loglik(probs)
    # The true cell's features should score near the top.
    true_idx = r[:200] * cols + c[:200]
    rank = (ll > ll[np.arange(200), true_idx][:, None]).sum(axis=1)
    assert np.median(rank) < 0.1 * rows * cols
