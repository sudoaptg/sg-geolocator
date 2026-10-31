"""End-to-end pipeline: EXIF -> text clues -> landmark -> geocell / retrieval."""

from __future__ import annotations

from pathlib import Path

from .exif import read_gps
from .geo import Prediction, in_singapore, nearest_landmark

LANDMARK_THRESHOLD = 0.5
PLACE_THRESHOLD = 0.5  # share of the top matches that agree on one spot (within PLACE_RADIUS_KM)
PLACE_RADIUS_KM = 0.1
# On held-out sequences (190k-photo index): k=50 answers 19% of photos, 88% within 50 m and
# 95% within 1 km; k=10 answers 30% but only 80% within 1 km. Prefer being right over
# answering often: a wrong "exact spot" is worse than falling back to the area guess.
PLACE_K = 50
CLUE_THRESHOLD = 0.5  # even a road name alone usually beats the km-scale visual guess
SKYLINE_BLOCK_DEG = 0.005
VIEWPOINT_RADIUS_M = 600.0  # a sign readable in a photo is rarely further away than this
VIEWPOINT_DISTINCT = 0.8  # use the skyline viewpoint only if it fits clearly better than a typical spot  # skyline candidates: the best map-match cell of every ~550 m block


class Geolocator:
    def __init__(
        self,
        index_path: str | Path | None = None,
        use_exif: bool = True,
        use_text: bool = True,
        geocell_path: str | Path | None = None,
        places_path: str | Path | None = None,
        map_path: str | Path | None = None,
        buildings_path: str | Path | None = None,
    ):
        self.index_path = index_path
        self.geocell_path = geocell_path
        self.places_path = places_path
        self.map_path = map_path
        self.buildings_path = buildings_path
        self._map = None
        self._buildings = None
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
        if self.map_path:
            from .mapmatch import MapMatcher

            self._map = MapMatcher.load(self.map_path)
            if self._geocell is not None and self._map.mu.shape != self._geocell.mu.shape:
                raise ValueError("map matcher and area classifier were trained on different embedding models")
        if self.buildings_path:
            from .skyline import Buildings

            self._buildings = Buildings.load(self.buildings_path)
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
            match = self._places.predict(place_emb, k=PLACE_K, method="cluster", radius_km=PLACE_RADIUS_KM)
            if match.confidence >= PLACE_THRESHOLD:
                match.source = "place-match"
                return match
        if self._geocell is not None and self._map is not None and self._map.lam > 0:
            return self._map_match(path, emb)
        if self._geocell is not None:
            (lat, lon), conf = (x[0] for x in self._geocell.predict(emb[None, :]))
            label, _ = nearest_landmark(lat, lon)
            return Prediction(float(lat), float(lon), float(conf), "geocell", f"near {label}")
        if self._index is not None:
            return self._index.predict(emb)
        return landmark  # low-confidence landmark guess, or None

    def _map_match(self, path, emb) -> Prediction:
        """Area classifier + map features; skyline matches are added as hints."""
        import numpy as np

        from .mapmatch import MapGrid, describe, spread_candidates

        score = self._map.scores(emb[None, :], self._geocell)
        (lat, lon), conf = (x[0] for x in self._map.best(score))
        seen = describe(self._map.feature_probs(emb[None, :])[0], self._map.prior, self._map.grid.names)
        if self._buildings is not None:
            from .skyline import search

            r, c, _ = self._map.cells
            top = spread_candidates(score[0], r, c, len(r), SKYLINE_BLOCK_DEG)
            cand = np.stack(MapGrid.centre(r[top], c[top]), axis=1)
            found = search(self._buildings, cand, score[0, top], path)
            if found is not None:  # hints only: the skyline alone picked wrong estates in testing
                for spot in found["spots"][:3]:
                    near, _ = nearest_landmark(spot["lat"], spot["lon"])
                    seen.append(f"skyline also matches near {spot['lat']:.4f}, {spot['lon']:.4f} ({near}): "
                                f"{spot['cost']:.2f}° vs typical {found['median_cost']:.2f}°")
        label, _ = nearest_landmark(lat, lon)
        return Prediction(float(lat), float(lon), float(conf), "map-match", f"near {label}", seen)

    def _viewpoint(self, path, pred: Prediction) -> Prediction:
        """Once a sign places the photo near a landmark, find the exact viewpoint and floor
        by matching the skyline against buildings within a few hundred metres of it."""
        if self._buildings is None:
            return pred
        from .geo import haversine_km
        from .skyline import viewpoint

        v = viewpoint(self._buildings, path, pred.lat, pred.lon, radius_m=VIEWPOINT_RADIUS_M)
        if v is None or v["cost"] > VIEWPOINT_DISTINCT * v["typical_cost"]:
            return pred  # no clear skyline, or no spot fits clearly better than the rest
        dist = haversine_km(pred.lat, pred.lon, v["lat"], v["lon"]) * 1000
        lo, hi = v["floor_range"]
        floors = f"floor ~{v['floor']}" + (f" (fits {lo}-{hi})" if hi > lo else "")
        pred.evidence.append(
            f"viewpoint from skyline: {dist:.0f} m from the landmark, {floors}, facing {v['heading']:.0f}° "
            f"(outline error {v['cost']:.2f}° vs typical {v['typical_cost']:.2f}°; near-equal spots within "
            f"{v['spread_m']:.0f} m)")
        return Prediction(v["lat"], v["lon"], pred.confidence, "text+skyline", pred.label, pred.evidence)

    def predict(self, path: str | Path) -> Prediction | None:
        if self.use_exif:
            gps = read_gps(path)
            if gps and in_singapore(*gps):
                label, _ = nearest_landmark(*gps)
                return Prediction(gps[0], gps[1], 1.0, "exif", f"near {label}")

        visual = self.visual(path)
        if self._clues is not None:
            prior = (visual.lat, visual.lon) if visual else None
            texts = self._clues.reader.read(path)
            found = self._clues.locate_text(texts, prior)
            if found and found.confidence >= CLUE_THRESHOLD:
                label, _ = nearest_landmark(found.lat, found.lon)
                pred = Prediction(found.lat, found.lon, found.confidence, "text", f"near {label}", found.evidence)
                return self._viewpoint(path, pred)
            if visual is not None and texts:  # show what was read, so misses can be diagnosed
                visual.evidence.append("text read, no match: " + ", ".join(repr(t) for t in texts[:8]))
        return visual
