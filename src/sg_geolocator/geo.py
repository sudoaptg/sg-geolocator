"""Geographic helpers and the shared prediction type."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path

# Rough bounding box of Singapore (main island plus offshore islands). It also clips
# bits of Johor and the Riau islands; use a Boundary when that matters.
SG_BOUNDS = {"lat_min": 1.15, "lat_max": 1.48, "lon_min": 103.59, "lon_max": 104.10}


@dataclass
class Prediction:
    lat: float
    lon: float
    confidence: float
    source: str  # "exif", "text", "landmark" or "retrieval"
    label: str | None = None  # landmark name or nearest known area
    evidence: list[str] = field(default_factory=list)  # e.g. which signs were read

    def as_dict(self) -> dict:
        out = {
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "label": self.label,
            "map": f"https://www.google.com/maps?q={self.lat:.6f},{self.lon:.6f}",
        }
        if self.evidence:
            out["evidence"] = self.evidence
        return out


def in_singapore(lat: float, lon: float) -> bool:
    b = SG_BOUNDS
    return b["lat_min"] <= lat <= b["lat_max"] and b["lon_min"] <= lon <= b["lon_max"]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


class Boundary:
    """Land boundary from a GeoJSON (Multi)Polygon, e.g. geoBoundaries SGP ADM0."""

    def __init__(self, polygons: list[list[list[tuple[float, float]]]]):
        # polygons -> rings (first is the outer ring, rest are holes) -> (lon, lat) points
        self.polygons = polygons

    @classmethod
    def from_geojson(cls, path: str | Path) -> "Boundary":
        data = json.loads(Path(path).read_text())
        geoms = [f["geometry"] for f in data["features"]] if "features" in data else [data]
        polygons = []
        for g in geoms:
            coords = g["coordinates"]
            polys = [coords] if g["type"] == "Polygon" else coords
            for poly in polys:
                polygons.append([[(float(x), float(y)) for x, y, *_ in ring] for ring in poly])
        return cls(polygons)

    def contains(self, lat: float, lon: float) -> bool:
        for rings in self.polygons:
            if _in_ring(lon, lat, rings[0]) and not any(_in_ring(lon, lat, h) for h in rings[1:]):
                return True
        return False

    def intersects_box(self, min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> bool:
        """Conservative test: true if any boundary vertex is in the box or a box corner/centre is on land."""
        for rings in self.polygons:
            for x, y in rings[0]:
                if min_lon <= x <= max_lon and min_lat <= y <= max_lat:
                    return True
        corners = [
            (min_lat, min_lon), (min_lat, max_lon), (max_lat, min_lon), (max_lat, max_lon),
            ((min_lat + max_lat) / 2, (min_lon + max_lon) / 2),
        ]
        return any(self.contains(lat, lon) for lat, lon in corners)


def _in_ring(x: float, y: float, ring: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


@lru_cache(maxsize=1)
def load_landmarks() -> list[dict]:
    text = resources.files("sg_geolocator").joinpath("data/landmarks.json").read_text()
    return json.loads(text)


def nearest_landmark(lat: float, lon: float) -> tuple[str, float]:
    """Return the closest known landmark and its distance in km."""
    best = min(load_landmarks(), key=lambda lm: haversine_km(lat, lon, lm["lat"], lm["lon"]))
    return best["name"], haversine_km(lat, lon, best["lat"], best["lon"])
