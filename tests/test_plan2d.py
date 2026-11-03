import numpy as np
from PIL import Image

from sg_geolocator import plan2d as P
from sg_geolocator.skyline import Buildings, _densify, render

from test_skyline import _city, _photo_from


def _blocks():
    """A few rectangular blocks around (1.35, 103.80), each with a different size and spacing."""
    specs = [(0.0006, 0.0004, 3e-4, 2e-4, 60), (0.0009, -0.0005, 2e-4, 4e-4, 45), (-0.0002, 0.0010, 4e-4, 2e-4, 80),
             (0.0013, 0.0004, 2e-4, 2e-4, 36), (-0.0010, -0.0008, 3e-4, 3e-4, 50), (0.0004, -0.0012, 2e-4, 3e-4, 70),
             (-0.0007, 0.0003, 1.5e-4, 3e-4, 30), (0.0016, -0.0010, 3e-4, 2e-4, 90)]
    pts, hs, ids = [], [], []
    for k, (dlat, dlon, w, h, ht) in enumerate(specs):
        lat, lon = 1.35 + dlat, 103.80 + dlon
        sq = _densify(np.array([[lat, lon], [lat, lon + w], [lat + h, lon + w], [lat + h, lon], [lat, lon]]), 4.0)
        pts.append(sq), hs.append(np.full(len(sq), ht)), ids.append(np.full(len(sq), k))
    return Buildings(np.concatenate(pts), np.concatenate(hs).astype(np.float32), np.concatenate(ids), [""] * len(specs))


def _photo_x(b, lat, lon, heading, hfov):
    o = P.Outlines(b, lat, lon, 2000, min_height=0)
    rel = P._wrap(o.edges(P.to_xy(lat, lon)) - heading)
    return np.sort(P.x_from_angle(rel[np.abs(rel) < hfov / 2], hfov))


def test_resect_finds_the_spot_from_edges_alone():
    b = _blocks()
    true = (1.35, 103.80)
    px = _photo_x(b, *true, heading=40, hfov=70)
    assert len(px) >= 4
    best = P.resect(b, px, 1.35, 103.80, radius_m=60, hfovs=[70], step_m=4, min_height=0, top=3)[0]
    off = np.hypot(*(P.to_xy(best["lat"], best["lon"]) - P.to_xy(*true)))
    assert off < 10 and abs(P._wrap(best["heading"] - 40)) < 3 and best["edge_match"] > 0.9


def test_landmark_pins_the_heading():
    b = _blocks()
    true = (1.3501, 103.8001)
    px = _photo_x(b, *true, heading=200, hfov=60)
    lm_lat, lm_lon = 1.3492, 103.7993  # a point we pretend was recognised in the photo
    a = P._wrap(P._bearing(P.to_xy(lm_lat, lm_lon) - P.to_xy(*true)) - 200)
    lm = P.Landmark(float(P.x_from_angle(a, 60)), lm_lat, lm_lon)
    best = P.resect(b, px, 1.35, 103.80, radius_m=60, landmarks=[lm], hfovs=[60], step_m=4, min_height=0)[0]
    assert np.hypot(*(P.to_xy(best["lat"], best["lon"]) - P.to_xy(*true))) < 10
    assert best["landmark_error"] < 0.01


def test_photo_edges_finds_building_sides(tmp_path):
    img = np.full((300, 600), 25, np.uint8)  # night sky
    for x0, x1, top in ((90, 180, 60), (240, 300, 120), (400, 520, 40)):
        img[top:, x0:x1] = 170
    path = tmp_path / "e.png"
    Image.fromarray(img).save(path)
    got = P.photo_edges(path) * 600
    for want in (90, 180, 240, 300, 400, 520):
        assert np.abs(got - want).min() < 4


def test_fit_height_recovers_eye_level(tmp_path):
    b = _city()
    rel, hs = b.near(1.35, 103.80)
    path = tmp_path / "s.png"
    Image.fromarray(_photo_from(render(rel, hs, 25.0), heading=130, pitch=0, hfov=70)).save(path)
    r = P.fit_height(b, path, 1.35, 103.80, heading=130, hfov=70)
    assert r is not None and r["reliable"] and abs(r["eye_m"] - 25.0) <= 6


def test_window_mode_only_tries_building_outlines():
    b = _blocks()
    k = 2  # the 80 m block: photo from its outline
    cam = b.points[b.ids == k][3]
    px = _photo_x(b, float(cam[0]), float(cam[1]), heading=300, hfov=70)
    best = P.resect(b, px, 1.35, 103.80, radius_m=250, hfovs=[70], step_m=4, min_height=0, window_min_height=75)[0]
    on = P.to_xy(best["lat"], best["lon"])
    assert np.hypot(*(b.xy[b.ids == k] - on).T).min() < 1  # a point on the tall block's outline
    assert abs(P._wrap(best["heading"] - 300)) < 5
