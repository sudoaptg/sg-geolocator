import numpy as np
from PIL import Image

from sg_geolocator.skyline import _densify, AZ_BINS, Buildings, building_height, match_cost, photo_skyline, render, search


def _city(seed=0, n=150):
    rng = np.random.default_rng(seed)
    pts, hs = [], []
    for _ in range(n):
        lat, lon = 1.35 + rng.uniform(-0.01, 0.01), 103.8 + rng.uniform(-0.01, 0.01)
        h = rng.choice([12.0, 36.0, 48.0, 60.0])
        sq = _densify(np.array([[lat, lon], [lat, lon + 3e-4], [lat + 2e-4, lon + 3e-4], [lat + 2e-4, lon], [lat, lon]]), 8.0)
        pts.append(sq)
        hs.append(np.full(len(sq), h))
    return Buildings(np.concatenate(pts), np.concatenate(hs).astype(np.float32))


def _photo_from(profile, heading, pitch=0.0, hfov=70.0, width=240, height=180):
    """Synthetic night photo: dark sky above the rendered skyline, lit buildings below."""
    img = np.full((height, width), 20, dtype=np.uint8)
    vfov = hfov * height / width
    for x in range(width):
        az = (heading + (x + 0.5 - width / 2) / width * hfov) % 360
        el = profile[int(round(az)) % AZ_BINS] + pitch
        row = int(round(height / 2 - el / vfov * height))
        img[max(0, row):, x] = 200
    return img


def test_skyline_matches_true_spot_better(tmp_path):
    b = _city()
    true = (1.35, 103.80)
    rel, hs = b.near(*true)
    prof = render(rel, hs, 25.0)
    path = tmp_path / "p.png"
    Image.fromarray(_photo_from(prof, heading=130, pitch=3)).save(path)
    sky = photo_skyline(path, hfov_deg=70)
    assert sky is not None
    good = match_cost(prof, *sky)
    other = match_cost(render(*b.near(1.356, 103.806), 25.0), *sky)
    assert good < 1.0 and other > good * 2
    grid = np.array([(1.35 + i * 0.003, 103.80 + j * 0.003) for i in range(-3, 4) for j in range(-3, 4)])
    prior = -np.hypot(grid[:, 0] - 1.356, grid[:, 1] - 103.806) * 100  # area guess favours another spot
    found = search(b, grid, prior, sky)
    best = found["spots"][0]
    assert abs(best["lat"] - 1.35) < 0.001 and abs(best["lon"] - 103.80) < 0.001


def test_no_skyline_when_frame_blocks_the_sky(tmp_path):
    path = tmp_path / "wall.png"
    Image.fromarray(np.full((120, 160), 128, dtype=np.uint8)).save(path)
    assert photo_skyline(path) is None


def test_building_height_sources():
    assert building_height({"height": "45 m"}, {}) == 45.0
    assert building_height({"building:levels": "10"}, {}) == 30.0
    assert building_height({"addr:housenumber": "123", "addr:street": "Ang Mo Kio Avenue 3"},
                           {("123", "ANG MO KIO AVE 3"): 12}) == 36.0
    assert building_height({}, {}) is None


def test_check_ranks_true_area_above_random(tmp_path):
    from sg_geolocator.skyline import check

    b = _city(n=200)
    rel, hs = b.near(1.35, 103.80)
    path = tmp_path / "p.png"
    Image.fromarray(_photo_from(render(rel, hs, 25.0), heading=40)).save(path)
    out = check(b, path, 1.35, 103.80, radius_m=40, n_random=40)
    assert out["share_of_random_spots_beaten"] > 0.8


def test_spread_candidates_one_per_block():
    from sg_geolocator.mapmatch import spread_candidates

    r = np.array([0, 1, 2, 50, 51])
    c = np.zeros(5, dtype=int)
    score = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert spread_candidates(score, r, c, 10, 0.005).tolist() == [0, 3]


def test_day_sky_with_white_clouds_and_white_blocks(tmp_path):
    """Bright clouds above a bright, window-textured block: the boundary is the block top."""
    rng = np.random.default_rng(1)
    img = np.full((180, 240), 225.0)
    img[:60] += rng.normal(0, 2, (60, 240))  # smooth cloud
    img[100:, :] = 215.0
    img[100:, ::4] = 120.0  # windows / corridors: texture
    img[100::6, :] = 120.0
    path = tmp_path / "day.png"
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)
    az, el = photo_skyline(path, hfov_deg=70)
    vfov = 70 * 180 / 240
    assert len(az) > 200
    # Within ~2 rows (the gradient blur shifts the edge up slightly; the tilt search absorbs it).
    assert np.allclose(np.median(el), (90 - 100) / 180 * vfov, atol=0.7)


def test_viewpoint_recovers_position_and_floor(tmp_path):
    from sg_geolocator.skyline import FLOOR_M, viewpoint

    b = _city(seed=3, n=500)
    true = (1.35 + 0.002, 103.80 + 0.001)  # ~250 m from the anchor
    floor = 8
    rel, hs = b.near(*true)
    path = tmp_path / "v.png"
    Image.fromarray(_photo_from(render(rel, hs, floor * FLOOR_M + 1.5), heading=200, pitch=-4)).save(path)
    v = viewpoint(b, path, 1.35, 103.80, radius_m=400)
    err_m = np.hypot((v["lat"] - true[0]) * 111_000, (v["lon"] - true[1]) * 111_000)
    assert err_m < 40, v
    assert abs(v["floor"] - floor) <= 2, v
    assert v["cost"] < v["typical_cost"]


def test_viewpoint_names_block_and_floor(tmp_path):
    from sg_geolocator.skyline import FLOOR_M, viewpoint

    rng = np.random.default_rng(5)
    pts, hs, ids, labels = [], [], [], []
    for k in range(300):
        lat, lon = 1.35 + rng.uniform(-0.008, 0.008), 103.8 + rng.uniform(-0.008, 0.008)
        ring = _densify(np.array([[lat, lon], [lat, lon + 4e-4], [lat + 1.2e-4, lon + 4e-4], [lat + 1.2e-4, lon], [lat, lon]]), 8.0)
        pts.append(ring)
        hs.append(np.full(len(ring), rng.choice([15.0, 36.0, 45.0, 60.0])))
        ids.append(np.full(len(ring), k))
        labels.append(f"Blk {100 + k} TEST STREET" if k % 3 == 0 else "")
    b = Buildings(np.concatenate(pts), np.concatenate(hs).astype(np.float32), np.concatenate(ids), labels)
    own = next(k for k in range(0, 300, 3) if hs[k][0] == 60.0
               and abs(pts[k][0][0] - 1.35) < 0.004 and abs(pts[k][0][1] - 103.8) < 0.004)
    view = pts[own][3]
    floor = 8
    rel, h = b.near(*view, exclude=own)
    anchor = (view[0] - 0.0015, view[1])  # landmark ~170 m south
    path = tmp_path / "f.png"
    Image.fromarray(_photo_from(render(rel, h, floor * FLOOR_M + 1.5), heading=180)).save(path)
    v = viewpoint(b, path, *anchor, radius_m=400)
    assert v["block"] == labels[own], v
    assert abs(v["floor"] - floor) <= 2, v
