"""End-to-end pipeline: EXIF -> text clues -> landmark -> geocell / retrieval."""

from __future__ import annotations

from pathlib import Path

from .exif import read_gps
from .geo import Prediction, in_singapore, nearest_landmark

LANDMARK_THRESHOLD = 0.5
PLACE_THRESHOLD = 0.5  # share of the top matches that agree on one spot (within PLACE_RADIUS_KM)
PLACE_RADIUS_KM = 0.1
CLUE_THRESHOLD = 0.5  # even a road name alone usually beats the km-scale visual guess


class Geolocator:
    def __init__(
        self,
        index_path: str | Path | None = None,
        use_exif: bool = True,
        use_text: bool = True,
        geocell_path: str | Path | None = None,
        places_path: str | Path | None = None,
    ):
        self.index_path = index_path
        self.geocell_path = geocell_path
        self.places_path = places_path
        self.use_exif = use_exif
        self.use_text = use_text
        self._embedder = None
        self._landmarks = None
        self._index = None
        self._geocell = None
        self._places = None
        self._place_embedder = None
        self._clues = None

    def _load_models(self) -> None:
        if self._embedder is not None:
            return
        from .embedder import DEFAULT_MODEL, make_embedder
        from .geocell import GeoCellClassifier
        from .landmarks import LandmarkRecognizer
        from .retrieval import RetrievalIndex

        model = DEFAULT_MODEL
        if self.index_path:
            self._index = RetrievalIndex(self.index_path)
            model = self._index.model  # query photos must be embedded with the index's model
        if self.geocell_path:
            self._geocell = GeoCellClassifier.load(self.geocell_path)
            if self._geocell.model and self._geocell.model != model:
                raise ValueError(f"geocell was trained on {self._geocell.model} embeddings, index uses {model}")
        self._embedder = make_embedder(model)
        self._landmarks = LandmarkRecognizer(self._embedder)
        if self.places_path:
            self._places = RetrievalIndex(self.places_path)
            self._place_embedder = make_embedder(self._places.model)
        if self.use_text:
            from .clues import ClueLocator

            self._clues = ClueLocator()

    def visual(self, path: str | Path) -> Prediction | None:
        """Best guess from how the scene looks: a known landmark, else an exact-spot match
        against the dense place index, else the area classifier, else similar photos."""
        self._load_models()
        emb = self._embedder.embed_images([path])[0]
        landmark = self._landmarks.predict(emb)
        if landmark and landmark.confidence >= LANDMARK_THRESHOLD:
            return landmark
        if self._places is not None:
            place_emb = self._place_embedder.embed_images([path])[0]
            match = self._places.predict(place_emb, k=10, method="cluster", radius_km=PLACE_RADIUS_KM)
            if match.confidence >= PLACE_THRESHOLD:
                match.source = "place-match"
                return match
        if self._geocell is not None:
            (lat, lon), conf = (x[0] for x in self._geocell.predict(emb[None, :]))
            label, _ = nearest_landmark(lat, lon)
            return Prediction(float(lat), float(lon), float(conf), "geocell", f"near {label}")
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
