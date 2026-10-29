"""End-to-end pipeline: EXIF -> text clues -> landmark -> retrieval."""

from __future__ import annotations

from pathlib import Path

from .exif import read_gps
from .geo import Prediction, in_singapore, nearest_landmark

LANDMARK_THRESHOLD = 0.5
CLUE_THRESHOLD = 0.5  # even a road name alone usually beats the km-scale visual guess


class Geolocator:
    def __init__(self, index_path: str | Path | None = None, use_exif: bool = True, use_text: bool = True):
        self.index_path = index_path
        self.use_exif = use_exif
        self.use_text = use_text
        self._embedder = None
        self._landmarks = None
        self._index = None
        self._clues = None

    def _load_models(self) -> None:
        if self._embedder is not None:
            return
        from .embedder import ClipEmbedder
        from .landmarks import LandmarkRecognizer
        from .retrieval import RetrievalIndex

        self._embedder = ClipEmbedder()
        self._landmarks = LandmarkRecognizer(self._embedder)
        if self.index_path:
            self._index = RetrievalIndex(self.index_path)
        if self.use_text:
            from .clues import ClueLocator

            self._clues = ClueLocator()

    def visual(self, path: str | Path) -> Prediction | None:
        """Best guess from how the scene looks: a known landmark, else similar reference photos."""
        self._load_models()
        emb = self._embedder.embed_images([path])[0]
        landmark = self._landmarks.predict(emb)
        if landmark and landmark.confidence >= LANDMARK_THRESHOLD:
            return landmark
        if self._index is not None:
            return self._index.predict(emb)
        return landmark  # low-confidence landmark guess, or None

    def predict(self, path: str | Path) -> Prediction | None:
        if self.use_exif:
            gps = read_gps(path)
            if gps and in_singapore(*gps):
                label, _ = nearest_landmark(*gps)
                return Prediction(gps[0], gps[1], 1.0, "exif", f"near {label}")

        visual = self.visual(path)
        if self._clues is not None:
            prior = (visual.lat, visual.lon) if visual else None
            found = self._clues.locate(path, prior)
            if found and found.confidence >= CLUE_THRESHOLD:
                label, _ = nearest_landmark(found.lat, found.lon)
                return Prediction(found.lat, found.lon, found.confidence, "text", f"near {label}", found.evidence)
        return visual
