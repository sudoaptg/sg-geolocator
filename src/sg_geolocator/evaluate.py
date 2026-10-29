"""Measure geolocation accuracy on held-out photos.

Photos are split by Mapillary *sequence* (one continuous capture session), not at random:
neighbouring frames of the same drive are near-duplicates, so a random split would let the
model "find" a test photo by matching the frame taken one second earlier.
"""

from __future__ import annotations

import zlib

import numpy as np

from .geo import haversine_km
from .geocell import GeoCellClassifier
from .retrieval import RetrievalIndex

THRESHOLDS_KM = (0.5, 1, 5, 10)

# (name, k, aggregation) for the retrieval variants compared on every evaluation.
KNN_VARIANTS = [
    ("knn10_mean", 10, "mean"),
    ("top1", 1, "mean"),
    ("knn10_cluster", 10, "cluster"),
    ("knn50_cluster", 50, "cluster"),
]
GEOCELL_SIZES_DEG = (0.01, 0.02)  # ~1.1 km and ~2.2 km cells


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


def _errors(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.array([haversine_km(a, b, c, d) for (a, b), (c, d) in zip(pred, truth)])


def evaluate_index(index_path: str, test_fraction: float = 0.1, geocell: bool = True, log=None) -> dict:
    data = np.load(index_path)
    emb, coords, seqs = data["embeddings"], data["coords"], data["sequences"]
    test = split(seqs, test_fraction)
    if test.sum() == 0 or (~test).sum() == 0:
        raise ValueError("Not enough photos to split into gallery and test sets")

    gallery = RetrievalIndex(embeddings=emb[~test], coords=coords[~test])
    truth = coords[test]
    methods: dict[str, dict] = {}

    centre = coords[~test].mean(axis=0)  # baseline: always guess the gallery's centre
    methods["baseline_centre"] = metrics(_errors(np.tile(centre, (len(truth), 1)), truth))

    for name, k, how in KNN_VARIANTS:
        preds = [gallery.predict(e, k=k, method=how) for e in emb[test]]
        methods[name] = metrics(_errors(np.array([[p.lat, p.lon] for p in preds]), truth))

    if geocell:
        for size in GEOCELL_SIZES_DEG:
            clf = GeoCellClassifier(cell_deg=size).fit(emb[~test], coords[~test], log=log)
            pred, _ = clf.predict(emb[test])
            methods[f"geocell_{size}deg"] = metrics(_errors(pred, truth))

    best = min((m for m in methods if m != "baseline_centre"), key=lambda m: methods[m]["median_km"])
    return {"gallery_size": int((~test).sum()), "methods": methods, "best": best}


def to_markdown(result: dict) -> str:
    cols = ["n", "median_km", "mean_km"] + [f"within_{t}km" for t in THRESHOLDS_KM]
    lines = [
        "## Accuracy on held-out Mapillary sequences",
        "",
        f"Gallery size: {result['gallery_size']} photos. Best method: **{result['best']}**",
        "",
        "| method | " + " | ".join(cols) + " |",
        "|---" * (len(cols) + 1) + "|",
    ]
    for name, m in result["methods"].items():
        cells = [str(m["n"]), f"{m['median_km']:.2f}", f"{m['mean_km']:.2f}"]
        cells += [f"{m[f'within_{t}km']:.1%}" for t in THRESHOLDS_KM]
        label = f"**{name}**" if name == result["best"] else name
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "`baseline_centre` always guesses the middle of the gallery; every method must beat it.",
        "`knnK_mean` averages the K most similar photos; `knnK_cluster` keeps only the densest",
        "agreeing group of them; `geocell_*` is a linear classifier over grid cells of that size.",
    ]
    return "\n".join(lines) + "\n"
