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
# Named places whose signs are worth reading: schools, malls, community clubs, hawker
# centres, places of worship, ... (the text-clue stage matches OCR text against these).
NAMED_TESTS = {
    "amenity": {"school", "college", "university", "kindergarten", "community_centre", "hospital", "clinic",
                "place_of_worship", "marketplace", "food_court", "library", "police", "fire_station",
                "theatre", "cinema", "post_office"},
    "shop": {"mall", "supermarket", "department_store"},
    "leisure": {"sports_centre", "stadium", "park", "swimming_pool"},
    "tourism": {"attraction", "museum", "hotel"},
    "building": {"school", "hospital", "church", "temple", "mosque", "commercial", "retail", "apartments"},
}


def _named(tags) -> list[str]:
    """The place's names worth matching on signs: the main name plus its English and Chinese
    names (Chinatown and estate signs are often in Chinese, e.g. People's Park Complex is
    signed 珍珠坊)."""
    if not any(tags.get(k) in v for k, v in NAMED_TESTS.items()):
        return []
    return list(dict.fromkeys(n for n in (tags.get(k) for k in ("name", "name:en", "name:zh", "name:zh-Hans")) if n))


def _in_sg(lat: float, lon: float) -> bool:
    b = SG_BOUNDS
    return b["lat_min"] <= lat <= b["lat_max"] and b["lon_min"] <= lon <= b["lon_max"]


def extract(pbf: str | Path, log=print) -> dict:
    """{"features": {name: [(lat, lon) arrays]}, "buildings": [(ring array, tags dict)]}."""
    import osmium

    features: dict[str, list[np.ndarray]] = {n: [] for n in FEATURE_TESTS}
    buildings: list[tuple[np.ndarray, dict]] = []
    names: list[tuple[str, float, float]] = []  # (name, lat, lon of the centre)

    def coords(nodes):
        try:
            return np.array([[n.lat, n.lon] for n in nodes])
        except osmium.InvalidLocationError:
            return None

    class Handler(osmium.SimpleHandler):
        def node(self, n):
            if not n.tags or not _in_sg(n.location.lat, n.location.lon):
                return
            for nm in _named(n.tags):
                names.append((nm, n.location.lat, n.location.lon))
            for name, test in FEATURE_TESTS.items():
                if test(n.tags):
                    features[name].append(np.array([[n.location.lat, n.location.lon]]))

        def way(self, w):
            if not w.tags or len(w.nodes) == 0:
                return
            matches = [name for name, test in FEATURE_TESTS.items() if test(w.tags)]
            is_building = "building" in w.tags and any(k in w.tags for k in BUILDING_TAGS)
            named = _named(w.tags)
            if not matches and not is_building and not named:
                return
            pts = coords(w.nodes)
            if pts is None or not _in_sg(*pts[0]):
                return
            for nm in named:
                names.append((nm, *pts.mean(axis=0)))
            for name in matches:
                features[name].append(pts)
            if is_building:
                buildings.append((pts, {k: w.tags[k] for k in BUILDING_TAGS if k in w.tags}))

        def area(self, a):
            if a.from_way():  # closed ways were handled in way()
                return
            matches = [name for name, test in FEATURE_TESTS.items() if test(a.tags)]
            named = _named(a.tags)
            if not matches and not named:
                return
            for ring in a.outer_rings():
                pts = coords(ring)
                if pts is None or not _in_sg(*pts[0]):
                    continue
                for name in matches:
                    features[name].append(pts)
                for nm in named:
                    names.append((nm, *pts.mean(axis=0)))
                named = []  # one entry per relation

    start = time.time()
    Handler().apply_file(str(pbf), locations=True)
    log(f"  read {pbf} in {time.time() - start:.0f}s: "
        + ", ".join(f"{n} {len(v)}" for n, v in features.items()) + f", buildings {len(buildings)}, named places {len(names)}")
    return {"features": features, "buildings": buildings, "names": names}


def load(log=print) -> dict:
    """Singapore features from the Geofabrik extract (downloaded and parsed once, cached)."""
    cache = CACHE_PATH.parent / "osm_features_v3.pkl.gz"  # v3: named places with Chinese names
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
