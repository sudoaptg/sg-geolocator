"""Zero-shot landmark recognition with CLIP text prompts."""

from __future__ import annotations

import numpy as np

from .embedder import ClipEmbedder
from .geo import Prediction, load_landmarks

# Generic Singapore scenes compete with the landmark prompts, so an ordinary street
# photo isn't forced onto the nearest-looking landmark.
NEGATIVE_PROMPTS = [
    "a residential HDB housing estate in Singapore",
    "an ordinary street in Singapore",
    "a road with cars and trees",
    "the interior of a building",
    "a park with trees and grass",
    "an industrial estate with warehouses",
]


class LandmarkRecognizer:
    def __init__(self, embedder: ClipEmbedder):
        self.embedder = embedder
        self.landmarks = load_landmarks()
        prompts = [lm["prompt"] for lm in self.landmarks] + NEGATIVE_PROMPTS
        self.text_emb = embedder.embed_texts(prompts)

    def scores(self, image_emb: np.ndarray) -> np.ndarray:
        """Softmax probabilities over landmarks followed by the negative prompts."""
        logits = self.embedder.logit_scale * (self.text_emb @ image_emb)
        logits -= logits.max()
        p = np.exp(logits)
        return p / p.sum()

    def predict(self, image_emb: np.ndarray) -> Prediction | None:
        probs = self.scores(image_emb)
        idx = int(np.argmax(probs))
        if idx >= len(self.landmarks):
            return None  # a generic scene won
        lm = self.landmarks[idx]
        return Prediction(lm["lat"], lm["lon"], float(probs[idx]), "landmark", lm["name"])
