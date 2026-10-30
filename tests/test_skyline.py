import numpy as np
from PIL import Image

from sg_geolocator.skyline import AZ_BINS, Buildings, building_height, match_cost, photo_skyline, render, rerank


def _city(seed=0, n=150):
    rng = np.random.default_rng(seed)
    pts, hs = [], []
    for _ in range(n):
        lat, lon = 1.35 + rng.uniform(-0.01, 0.01), 103.8 + rng.uniform(-0.01, 0.01)
        h = rng.choice([12.0, 36.0, 48.0, 60.0])
        sq = np.array([[lat, lon], [lat, lon + 3e-4], [lat + 2e-4, lon + 3e-4], [lat + 2e-4, lon]])
        pts.append(sq)
        hs.append(np.full(4, h))
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
    scores, costs = rerank(b, np.array([true, (1.356, 103.806)]), np.zeros(2), sky)
    assert scores[0] > scores[1]


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
