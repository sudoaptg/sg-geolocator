"""Nearest-neighbour geolocation against a gallery of geotagged reference photos."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .embedder import ClipEmbedder
from .exif import read_gps
from .geo import Prediction, in_singapore, nearest_landmark

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}


def build_index(
    folder: str | Path, embedder: ClipEmbedder, out: str | Path, batch_size: int = 32
) -> int:
    """Embed every geotagged Singapore photo under `folder` and save an .npz index."""
    paths, coords = [], []
    for p in sorted(Path(folder).rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        gps = read_gps(p)
        if gps and in_singapore(*gps):
            paths.append(p)
            coords.append(gps)
    if not paths:
        raise ValueError(f"No geotagged Singapore photos found in {folder}")

    embs = [embedder.embed_images(paths[i : i + batch_size]) for i in range(0, len(paths), batch_size)]
    np.savez(
        out,
        embeddings=np.concatenate(embs).astype(np.float32),
        coords=np.asarray(coords, dtype=np.float64),
        paths=np.asarray([str(p) for p in paths]),
    )
    return len(paths)


class RetrievalIndex:
    def __init__(self, path: str | Path):
        data = np.load(path)
        self.embeddings = data["embeddings"]
        self.coords = data["coords"]

    def predict(self, image_emb: np.ndarray, k: int = 10, temperature: float = 0.05) -> Prediction:
        sims = self.embeddings @ image_emb
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        weights = np.exp((sims[top] - sims[top].max()) / temperature)
        weights /= weights.sum()
        lat, lon = (weights[:, None] * self.coords[top]).sum(axis=0)

        # Confidence: how tightly the neighbours agree (1.0 when all within ~0 km).
        spread_km = float(np.sqrt((weights * ((self.coords[top] - [lat, lon]) ** 2).sum(axis=1)).sum()) * 111)
        confidence = 1.0 / (1.0 + spread_km)
        label, _ = nearest_landmark(lat, lon)
        return Prediction(float(lat), float(lon), confidence, "retrieval", f"near {label}")
