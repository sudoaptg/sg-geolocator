"""End-to-end pipeline: EXIF -> landmark -> retrieval."""

from __future__ import annotations

from pathlib import Path

from .exif import read_gps
from .geo import Prediction, in_singapore, nearest_landmark

LANDMARK_THRESHOLD = 0.5


class Geolocator:
    def __init__(self, index_path: str | Path | None = None, use_exif: bool = True):
        self.index_path = index_path
        self.use_exif = use_exif
        self._embedder = None
        self._landmarks = None
        self._index = None

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

    def predict(self, path: str | Path) -> Prediction | None:
        if self.use_exif:
            gps = read_gps(path)
            if gps and in_singapore(*gps):
                label, _ = nearest_landmark(*gps)
                return Prediction(gps[0], gps[1], 1.0, "exif", f"near {label}")

        self._load_models()
        emb = self._embedder.embed_images([path])[0]

        landmark = self._landmarks.predict(emb)
        if landmark and landmark.confidence >= LANDMARK_THRESHOLD:
            return landmark
        if self._index is not None:
            return self._index.predict(emb)
        return landmark  # low-confidence landmark guess, or None
