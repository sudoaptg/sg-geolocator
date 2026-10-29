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


TEXT_THRESHOLDS_KM = (0.1, 0.5, 1, 5)


def evaluate_text(
    metadata_csv: str,
    test_fraction: float = 0.1,
    limit: int | None = None,
    hires_token: str | None = None,
    log=print,
) -> dict:
    """Run the text-clue stage on held-out photos and score the ones it could place.

    The cached gallery photos are 512 px thumbnails, too small to read most signs; with a
    Mapillary token the 2048 px version of each test photo is fetched instead, which is
    closer to a real phone photo."""
    import csv
    import tempfile
    from pathlib import Path

    from .clues import ClueLocator

    base = Path(metadata_csv).parent / "images"
    with open(metadata_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if is_test(r.get("sequence", ""), r["id"], test_fraction)]
    rows = rows[:limit] if limit else rows
    locator = ClueLocator()
    tmp = Path(tempfile.mkdtemp())
    records = []
    for i, r in enumerate(rows, 1):
        path = base / r["path"]
        if hires_token:
            path = _fetch_hires(r["id"], hires_token, tmp) or path
        texts = locator.reader.read(path)
        found = locator.locate_text(texts) if texts else None
        rec = {"id": r["id"], "texts": texts, "lat": float(r["lat"]), "lon": float(r["lon"])}
        if found:
            rec |= {"error_km": haversine_km(found.lat, found.lon, rec["lat"], rec["lon"]),
                    "kind": found.clues[0].kind, "evidence": found.evidence, "confidence": found.confidence}
        records.append(rec)
        if i % 100 == 0:
            log(f"  {i}/{len(rows)} photos, {sum('error_km' in x for x in records)} placed from text")

    placed = [x for x in records if "error_km" in x]
    errs = np.array([x["error_km"] for x in placed])
    by_kind = {}
    for x in placed:
        by_kind.setdefault(x["kind"], []).append(x["error_km"])
    return {
        "n": len(records),
        "with_text": sum(bool(x["texts"]) for x in records),
        "placed": len(placed),
        "metrics": {
            "median_km": float(np.median(errs)) if len(errs) else None,
            **{f"within_{t}km": float(np.mean(errs <= t)) if len(errs) else None for t in TEXT_THRESHOLDS_KM},
        },
        "by_kind": {k: {"n": len(v), "median_km": float(np.median(v)), "within_0.5km": float(np.mean(np.array(v) <= 0.5))}
                    for k, v in sorted(by_kind.items())},
        "examples": placed[:25],
        "hires": bool(hires_token),
    }


def _fetch_hires(image_id: str, token: str, out_dir):
    import requests

    try:
        r = requests.get(f"https://graph.mapillary.com/{image_id}", params={"fields": "thumb_2048_url"},
                         headers={"Authorization": f"OAuth {token}"}, timeout=30)
        url = r.json().get("thumb_2048_url")
        if not url:
            return None
        img = requests.get(url, timeout=60)
        img.raise_for_status()
        path = out_dir / f"{image_id}.jpg"
        path.write_bytes(img.content)
        return path
    except (requests.RequestException, ValueError):
        return None


def text_to_markdown(res: dict) -> str:
    m = res["metrics"]
    pct = lambda v: "n/a" if v is None else f"{v:.1%}"  # noqa: E731
    median = "n/a" if m["median_km"] is None else f"{m['median_km']:.2f}"
    lines = [
        "## Text clues on held-out Mapillary photos",
        "",
        f"Photos: {res['n']} ({'2048 px from Mapillary' if res['hires'] else '512 px cached thumbnails'}). "
        f"Any text read: {res['with_text']} ({res['with_text'] / max(res['n'], 1):.0%}). "
        f"**Placed from text: {res['placed']} ({res['placed'] / max(res['n'], 1):.0%})**",
        "",
        "Accuracy of the photos placed from text:",
        "",
        "| median_km | " + " | ".join(f"within_{t}km" for t in TEXT_THRESHOLDS_KM) + " |",
        "|---" * (len(TEXT_THRESHOLDS_KM) + 1) + "|",
        f"| {median} | " + " | ".join(pct(m[f"within_{t}km"]) for t in TEXT_THRESHOLDS_KM) + " |",
        "",
        "| clue used | photos | median_km | within 0.5 km |",
        "|---|---|---|---|",
    ]
    for kind, v in res["by_kind"].items():
        lines.append(f"| {kind} | {v['n']} | {v['median_km']:.2f} | {v['within_0.5km']:.0%} |")
    lines += ["", "<details><summary>Examples</summary>", ""]
    for x in res["examples"]:
        lines.append(f"- `{x['id']}` off by **{x['error_km']:.2f} km** — {'; '.join(x['evidence'])} — read: {x['texts'][:6]}")
    lines += ["", "</details>"]
    return "\n".join(lines) + "\n"
