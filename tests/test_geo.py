from PIL import Image

from sg_geolocator.exif import read_gps
from sg_geolocator.geo import haversine_km, in_singapore, load_landmarks, nearest_landmark


def test_haversine_known_distance():
    # Marina Bay Sands to Jewel Changi is roughly 14 km.
    d = haversine_km(1.2834, 103.8607, 1.3602, 103.9898)
    assert 13 < d < 18


def test_landmarks_inside_singapore():
    for lm in load_landmarks():
        assert in_singapore(lm["lat"], lm["lon"]), lm["name"]


def test_nearest_landmark():
    name, dist = nearest_landmark(1.2868, 103.8545)
    assert name == "Merlion Park"
    assert dist < 0.01


def test_read_gps_roundtrip(tmp_path):
    path = tmp_path / "gps.jpg"
    img = Image.new("RGB", (8, 8))
    exif = img.getexif()
    exif.get_ifd(0x8825).update({1: "N", 2: (1.0, 17.0, 0.24), 3: "E", 4: (103.0, 51.0, 38.52)})
    img.save(path, exif=exif)

    lat, lon = read_gps(path)
    assert abs(lat - 1.2834) < 1e-3
    assert abs(lon - 103.8607) < 1e-3


def test_read_gps_missing(tmp_path):
    path = tmp_path / "plain.jpg"
    Image.new("RGB", (8, 8)).save(path)
    assert read_gps(path) is None
