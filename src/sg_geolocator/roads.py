"""Singapore road network from OpenStreetMap: fuzzy road-name matching and junctions.

Street signs are the most common readable clue, but OCR garbles them ("Yo Chu Kang Rd",
"Serhawang Way") and a road name alone only says "somewhere along this road", which for
a long road is kilometres. With the road shapes we can
- correct misread names against every named road in Singapore, and
- when two road names are read in the same photo (junction signs usually show both),
  place the photo where those roads meet: typically within tens of metres.

Data: OpenStreetMap contributors, ODbL, via the Overpass API (downloaded once, cached).
"""

from __future__ import annotations

import difflib
import gzip
import json
import os
import re
from pathlib import Path

import numpy as np

from .geo import haversine_km

# Public Overpass servers, tried in order (override with OVERPASS_URL).
OVERPASS_URLS = [os.environ["OVERPASS_URL"]] if os.environ.get("OVERPASS_URL") else [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
# overpass-api.de answers 406 to requests without an identifying User-Agent.
HEADERS = {
    "User-Agent": "sg-geolocator/0.1 (+https://github.com/sudoaptg/sg-geolocator)",
    "Accept": "application/json",
}
OVERPASS_QUERY = """
[out:json][timeout:300];
area["ISO3166-1"="SG"][admin_level=2]->.sg;
way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|service|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|pedestrian)$"]["name"](area.sg);
out geom;
"""
CACHE_PATH = Path(os.environ.get("SG_GEOLOCATOR_CACHE", Path.home() / ".cache" / "sg_geolocator")) / "roads.json.gz"

MATCH_CUTOFF = 0.82
JUNCTION_KM = 0.05  # roads closer than this are treated as meeting


def _digits(name: str) -> list[str]:
    return re.findall(r"\d+", name)


class Roads:
    def __init__(self, roads: dict[str, list[list[list[float]]]]):
        # name (upper case) -> list of polylines -> [lat, lon] points
        self.roads = roads
        self.names = sorted(roads)
        self._points: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ loading

    @classmethod
    def from_overpass_json(cls, data: dict) -> "Roads":
        roads: dict[str, list] = {}
        for el in data.get("elements", []):
            name = el.get("tags", {}).get("name")
            geom = el.get("geometry")
            if not name or not geom:
                continue
            line = [[round(p["lat"], 5), round(p["lon"], 5)] for p in geom]
            roads.setdefault(name.upper(), []).append(line)
        return cls(roads)

    @classmethod
    def load(cls, path: str | Path | None = None, log=None) -> "Roads":
        path = Path(path) if path else CACHE_PATH
        if not path.exists():
            import requests

            if log:
                log("Downloading Singapore road network from OpenStreetMap (one-off)...")
            errors = []
            for url in OVERPASS_URLS:
                try:
                    r = requests.post(url, data={"data": OVERPASS_QUERY}, headers=HEADERS, timeout=600)
                    r.raise_for_status()
                    roads = cls.from_overpass_json(r.json())
                    if roads.roads:
                        break
                    errors.append(f"{url}: no roads in response")
                except (requests.RequestException, ValueError) as exc:
                    errors.append(f"{url}: {exc}")
                    if log:
                        log(f"  {errors[-1]}")
            else:
                raise RuntimeError("Could not download the road network: " + "; ".join(errors))
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wt") as f:
                json.dump(roads.roads, f)
            return roads
        with gzip.open(path, "rt") as f:
            return cls(json.load(f))

    # ------------------------------------------------------------------ queries

    def match(self, name: str) -> str | None:
        """Canonical road name for an OCR'd (and abbreviation-expanded) name, or None.

        Numbers must agree exactly: 'TAMPINES AVENUE 5' must never match 'AVENUE 3'."""
        name = name.upper().strip()
        if name in self.roads:
            return name
        candidates = [n for n in self.names if _digits(n) == _digits(name)]
        best = difflib.get_close_matches(name, candidates, n=1, cutoff=MATCH_CUTOFF)
        return best[0] if best else None

    def points(self, name: str) -> np.ndarray:
        if name not in self._points:
            self._points[name] = np.array([p for line in self.roads[name] for p in line], dtype=np.float64)
        return self._points[name]

    def point_on(self, name: str, prior: tuple[float, float] | None = None) -> tuple[float, float]:
        """A point on the road: the one nearest `prior` if given, else the middle vertex."""
        pts = self.points(name)
        if prior is None:
            centre = pts.mean(axis=0)
            return tuple(pts[np.argmin(_km(pts, centre))])
        return tuple(pts[np.argmin(_km(pts, np.asarray(prior)))])

    def junctions(self, a: str, b: str, max_km: float = JUNCTION_KM) -> list[tuple[float, float]]:
        """Places where roads a and b meet (one point per separate meeting place)."""
        pa, pb = self.points(a), self.points(b)
        # Only compare points inside the other road's bounding box (plus margin), in chunks,
        # so two long expressways don't build a huge distance matrix.
        margin = max_km / 111.0 * 1.5
        lo, hi = pb.min(axis=0) - margin, pb.max(axis=0) + margin
        pa = pa[np.all((pa >= lo) & (pa <= hi), axis=1)]
        close: list[tuple[float, int, int]] = []
        for start in range(0, len(pa), 512):
            d = _km(pa[start : start + 512, None, :], pb[None, :, :])
            for i, j in zip(*np.nonzero(d <= max_km)):
                close.append((float(d[i, j]), start + int(i), int(j)))
        found: list[tuple[float, float]] = []
        for _, i, j in sorted(close):
            p = (float((pa[i][0] + pb[j][0]) / 2), float((pa[i][1] + pb[j][1]) / 2))
            if all(haversine_km(*p, *q) > 0.2 for q in found):  # same junction, nearby vertices
                found.append(p)
        return found


def _km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return np.sqrt((d[..., 0] * 111.0) ** 2 + (d[..., 1] * 111.0 * np.cos(np.radians(1.35))) ** 2)
