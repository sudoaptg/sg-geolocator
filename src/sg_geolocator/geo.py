"""Geographic helpers and the shared prediction type."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

# Rough bounding box of Singapore (main island plus offshore islands).
SG_BOUNDS = {"lat_min": 1.15, "lat_max": 1.48, "lon_min": 103.59, "lon_max": 104.10}


@dataclass
class Prediction:
    lat: float
    lon: float
    confidence: float
    source: str  # "exif", "landmark" or "retrieval"
    label: str | None = None  # landmark name or nearest known area

    def as_dict(self) -> dict:
        return {
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "label": self.label,
        }


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


@lru_cache(maxsize=1)
def load_landmarks() -> list[dict]:
    text = resources.files("sg_geolocator").joinpath("data/landmarks.json").read_text()
    return json.loads(text)


def nearest_landmark(lat: float, lon: float) -> tuple[str, float]:
    """Return the closest known landmark and its distance in km."""
    best = min(load_landmarks(), key=lambda lm: haversine_km(lat, lon, lm["lat"], lm["lon"]))
    return best["name"], haversine_km(lat, lon, best["lat"], best["lon"])
