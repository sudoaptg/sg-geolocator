"""Measure retrieval accuracy on held-out photos.

Photos are split by Mapillary *sequence* (one continuous capture session), not at random:
neighbouring frames of the same drive are near-duplicates, so a random split would let the
model "find" a test photo by matching the frame taken one second earlier.
"""

from __future__ import annotations

import zlib

import numpy as np

from .geo import haversine_km
from .retrieval import RetrievalIndex

THRESHOLDS_KM = (0.5, 1, 5, 10)


def is_test(sequence: str, key: str, test_fraction: float) -> bool:
    return (zlib.crc32(f"{sequence or key}".encode()) % 1000) < test_fraction * 1000


def split(sequences: np.ndarray, test_fraction: float = 0.1) -> np.ndarray:
    """Boolean mask of test rows; photos without a sequence are split individually."""
    return np.array([is_test(s, str(i), test_fraction) for i, s in enumerate(sequences)])


def metrics(errors_km: np.ndarray) -> dict:
    out = {
        "n": int(len(errors_km)),
        "median_km": float(np.median(errors_km)),
        "mean_km": float(np.mean(errors_km)),
    }
    for t in THRESHOLDS_KM:
        out[f"within_{t}km"] = float(np.mean(errors_km <= t))
    return out


def evaluate_index(index_path: str, test_fraction: float = 0.1, k: int = 10) -> dict:
    data = np.load(index_path)
    emb, coords, seqs = data["embeddings"], data["coords"], data["sequences"]
    test = split(seqs, test_fraction)
    if test.sum() == 0 or (~test).sum() == 0:
        raise ValueError("Not enough photos to split into gallery and test sets")

    gallery = RetrievalIndex(embeddings=emb[~test], coords=coords[~test])
    errs, base_errs = [], []
    centre = coords[~test].mean(axis=0)  # baseline: always guess the gallery's centre
    for e, (lat, lon) in zip(emb[test], coords[test]):
        p = gallery.predict(e, k=k)
        errs.append(haversine_km(p.lat, p.lon, lat, lon))
        base_errs.append(haversine_km(centre[0], centre[1], lat, lon))

    return {
        "gallery_size": int((~test).sum()),
        "retrieval": metrics(np.array(errs)),
        "baseline_centre": metrics(np.array(base_errs)),
    }


def to_markdown(result: dict) -> str:
    cols = ["n", "median_km", "mean_km"] + [f"within_{t}km" for t in THRESHOLDS_KM]
    lines = [
        "## Retrieval accuracy (held-out Mapillary sequences)",
        "",
        f"Gallery size: {result['gallery_size']} photos",
        "",
        "| method | " + " | ".join(cols) + " |",
        "|---" * (len(cols) + 1) + "|",
    ]
    for name in ("retrieval", "baseline_centre"):
        m = result[name]
        cells = [str(m["n"]), f"{m['median_km']:.2f}", f"{m['mean_km']:.2f}"]
        cells += [f"{m[f'within_{t}km']:.1%}" for t in THRESHOLDS_KM]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "`baseline_centre` always guesses the middle of the gallery; retrieval must beat it."]
    return "\n".join(lines) + "\n"
