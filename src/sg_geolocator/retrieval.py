"""Nearest-neighbour geolocation against a gallery of geotagged reference photos."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .embedder import DEFAULT_MODEL, ClipEmbedder
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
    return save_index(paths, coords, [""] * len(paths), embedder, out, batch_size)


def build_index_from_csv(
    csv_path: str | Path,
    embedder: ClipEmbedder,
    out: str | Path,
    batch_size: int = 32,
    log=print,
    shard: tuple[int, int] | None = None,
    limit: int | None = None,
) -> int:
    """Embed photos listed in a metadata.csv (columns: path, lat, lon, sequence).

    `shard=(i, n)` embeds only the i-th of n contiguous slices, so a slow model can be
    spread over parallel jobs; `merge_indexes` joins the parts back in order."""
    base = Path(csv_path).parent / "images"
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path} lists no photos")
    rows = rows[:limit] if limit else rows
    start = 0
    if shard:
        i, n = shard
        start, stop = len(rows) * i // n, len(rows) * (i + 1) // n
        rows = rows[start:stop]
    paths = [base / r["path"] for r in rows]
    coords = [(float(r["lat"]), float(r["lon"])) for r in rows]
    sequences = [r.get("sequence", "") for r in rows]
    return save_index(paths, coords, sequences, embedder, out, batch_size, log, offset=start)


def save_index(paths, coords, sequences, embedder, out, batch_size=32, log=None, offset=0) -> int:
    embs = []
    for i in range(0, len(paths), batch_size):
        embs.append(embedder.embed_images(paths[i : i + batch_size]))
        if log and (i // batch_size) % 50 == 0:
            log(f"  embedded {i + len(embs[-1])}/{len(paths)}")
    np.savez(
        out,
        embeddings=np.concatenate(embs).astype(np.float32),
        coords=np.asarray(coords, dtype=np.float64),
        sequences=np.asarray(sequences),
        paths=np.asarray([str(Path(p).name) for p in paths]),
        model=np.asarray(getattr(embedder, "model_name", DEFAULT_MODEL)),
        offset=np.asarray(offset),
    )
    return len(paths)


def merge_indexes(parts: list[str | Path], out: str | Path) -> int:
    """Join shard indexes (from build_index_from_csv with `shard`) in their original order."""
    loaded = sorted((np.load(p) for p in parts), key=lambda d: int(d["offset"]))
    models = {str(d["model"]) for d in loaded}
    if len(models) != 1:
        raise ValueError(f"Shards were built with different models: {models}")
    np.savez(
        out,
        **{k: np.concatenate([d[k] for d in loaded]) for k in ("embeddings", "coords", "sequences", "paths")},
        model=np.asarray(models.pop()),
        offset=np.asarray(0),
    )
    return sum(len(d["coords"]) for d in loaded)


def _approx_km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance in km between (lat, lon) arrays; equirectangular is exact enough at 1°N."""
    d = a - b
    return np.sqrt((d[..., 0] * 111.0) ** 2 + (d[..., 1] * 111.0 * np.cos(np.radians(1.35))) ** 2)


class RetrievalIndex:
    def __init__(self, path: str | Path | None = None, embeddings=None, coords=None, model: str = DEFAULT_MODEL):
        self.model = model
        if path is not None:
            data = np.load(path)
            embeddings, coords = data["embeddings"], data["coords"]
            # Indexes built before the model was recorded used the default model.
            self.model = str(data["model"]) if "model" in data else DEFAULT_MODEL
        self.embeddings = np.asarray(embeddings, dtype=np.float32)  # dense indexes are stored as float16
        self.coords = np.asarray(coords)

    def predict(
        self,
        image_emb: np.ndarray,
        k: int = 10,
        temperature: float = 0.05,
        method: str = "cluster",
        radius_km: float = 1.0,
    ) -> Prediction:
        """Locate from the top-k most similar gallery photos.

        method="mean" averages all k neighbours' coordinates, which drifts towards the middle
        of the island when they disagree. method="cluster" keeps only the neighbours within
        `radius_km` of the best-supported neighbour, i.e. the densest agreeing group."""
        sims = self.embeddings @ image_emb
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        weights = np.exp((sims[top] - sims[top].max()) / temperature)
        weights /= weights.sum()
        coords = self.coords[top]

        if method == "cluster" and k > 1:
            near = _approx_km(coords[:, None, :], coords[None, :, :]) <= radius_km
            keep = near[int(np.argmax(near @ weights))]
            # Confidence: share of the neighbours' weight that agrees on this spot.
            confidence = float(weights[keep].sum())
            coords, weights = coords[keep], weights[keep] / weights[keep].sum()
            lat, lon = (weights[:, None] * coords).sum(axis=0)
        else:
            lat, lon = (weights[:, None] * coords).sum(axis=0)
            # Confidence: how tightly the neighbours agree (1.0 when all within ~0 km).
            spread_km = float(np.sqrt((weights * _approx_km(coords, np.array([lat, lon])) ** 2).sum()))
            confidence = 1.0 / (1.0 + spread_km)

        label, _ = nearest_landmark(lat, lon)
        return Prediction(float(lat), float(lon), confidence, "retrieval", f"near {label}")
