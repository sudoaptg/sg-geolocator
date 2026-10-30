"""OpenStreetMap features for Singapore, read from Geofabrik's extract.

The public Overpass servers are often overloaded (502/504 for long stretches), so the
map matcher and skyline data are built from the regional extract instead: one plain file
download, read locally with pyosmium. Data © OpenStreetMap contributors, ODbL.
"""

from __future__ import annotations

import gzip
import pickle
import time
from pathlib import Path

import numpy as np
import requests

from .geo import SG_BOUNDS
from .roads import CACHE_PATH, HEADERS

EXTRACT_URL = "https://download.geofabrik.de/asia/malaysia-singapore-brunei-latest.osm.pbf"


def _levels(tags) -> float:
    try:
        return float(tags.get("building:levels", "0"))
    except ValueError:
        return 0.0


# name -> tag test (tags behave like a dict)
FEATURE_TESTS = {
    "school": lambda t: t.get("amenity") in ("school", "college"),
    "sports_pitch": lambda t: t.get("leisure") in ("pitch", "sports_centre", "stadium", "track"),
    "playground": lambda t: t.get("leisure") == "playground",
    "multistorey_carpark": lambda t: t.get("amenity") == "parking" and t.get("parking") == "multi-storey",
    "park": lambda t: t.get("leisure") in ("park", "garden", "nature_reserve"),
    "forest": lambda t: t.get("natural") == "wood" or t.get("landuse") == "forest",
    "water": lambda t: t.get("waterway") in ("canal", "river") or t.get("natural") == "water",
    "rail": lambda t: t.get("railway") in ("subway", "light_rail", "rail") and t.get("tunnel") != "yes",
    "expressway": lambda t: t.get("highway") == "motorway",
    "mall": lambda t: t.get("shop") == "mall",
    "hawker": lambda t: t.get("amenity") in ("food_court", "marketplace"),
    "worship": lambda t: t.get("amenity") == "place_of_worship",
    "tower_block": lambda t: "building" in t and _levels(t) >= 10,
    "landed_house": lambda t: t.get("building") in ("house", "detached", "semidetached_house", "terrace", "bungalow"),
    "industrial": lambda t: t.get("landuse") == "industrial",
    "construction": lambda t: t.get("landuse") == "construction",
}
BUILDING_TAGS = ("height", "building:levels", "addr:housenumber", "addr:street")


def _in_sg(lat: float, lon: float) -> bool:
    b = SG_BOUNDS
    return b["lat_min"] <= lat <= b["lat_max"] and b["lon_min"] <= lon <= b["lon_max"]


def extract(pbf: str | Path, log=print) -> dict:
    """{"features": {name: [(lat, lon) arrays]}, "buildings": [(ring array, tags dict)]}."""
    import osmium

    features: dict[str, list[np.ndarray]] = {n: [] for n in FEATURE_TESTS}
    buildings: list[tuple[np.ndarray, dict]] = []

    def coords(nodes):
        try:
            return np.array([[n.lat, n.lon] for n in nodes])
        except osmium.InvalidLocationError:
            return None

    class Handler(osmium.SimpleHandler):
        def node(self, n):
            if not n.tags or not _in_sg(n.location.lat, n.location.lon):
                return
            for name, test in FEATURE_TESTS.items():
                if test(n.tags):
                    features[name].append(np.array([[n.location.lat, n.location.lon]]))

        def way(self, w):
            if not w.tags or len(w.nodes) == 0:
                return
            matches = [name for name, test in FEATURE_TESTS.items() if test(w.tags)]
            is_building = "building" in w.tags and any(k in w.tags for k in BUILDING_TAGS)
            if not matches and not is_building:
                return
            pts = coords(w.nodes)
            if pts is None or not _in_sg(*pts[0]):
                return
            for name in matches:
                features[name].append(pts)
            if is_building:
                buildings.append((pts, {k: w.tags[k] for k in BUILDING_TAGS if k in w.tags}))

        def area(self, a):
            if a.from_way():  # closed ways were handled in way()
                return
            matches = [name for name, test in FEATURE_TESTS.items() if test(a.tags)]
            if not matches:
                return
            for ring in a.outer_rings():
                pts = coords(ring)
                if pts is None or not _in_sg(*pts[0]):
                    continue
                for name in matches:
                    features[name].append(pts)

    start = time.time()
    Handler().apply_file(str(pbf), locations=True)
    log(f"  read {pbf} in {time.time() - start:.0f}s: "
        + ", ".join(f"{n} {len(v)}" for n, v in features.items()) + f", buildings {len(buildings)}")
    return {"features": features, "buildings": buildings}


def load(log=print) -> dict:
    """Singapore features from the Geofabrik extract (downloaded and parsed once, cached)."""
    cache = CACHE_PATH.parent / "osm_features.pkl.gz"
    if cache.exists():
        with gzip.open(cache, "rb") as f:
            return pickle.load(f)
    pbf = CACHE_PATH.parent / "region.osm.pbf"
    pbf.parent.mkdir(parents=True, exist_ok=True)
    if not pbf.exists():
        log(f"Downloading {EXTRACT_URL} ...")
        with requests.get(EXTRACT_URL, headers=HEADERS, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(pbf, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        log(f"  {pbf.stat().st_size / 1e6:.0f} MB")
    data = extract(pbf, log)
    with gzip.open(cache, "wb") as f:
        pickle.dump(data, f)
    pbf.unlink()  # large; the parsed features are what's kept
    return data
