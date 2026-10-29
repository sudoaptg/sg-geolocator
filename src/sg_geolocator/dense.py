"""Dense reference index for exact-place matching (visual place recognition).

Matching a photo to the *same spot* needs reference photos every few tens of metres,
i.e. hundreds of thousands of them: far more than fit on a CI runner's disk. So each
shard job searches its own share of the map tiles, then downloads, embeds and deletes
photos in batches, keeping only the compact descriptors. `merge_indexes` joins shards.
"""

from __future__ import annotations

import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

from .geo import Boundary
from .mapillary import _download, iter_tiles, search_tiles, select

DENSE_SUBCELL_DEG = 0.00015  # ~17 m grid
DENSE_PER_SUBCELL = 2  # from different capture sessions, so views differ


def shard_tiles(boundary: Boundary | None, shard: int, n: int) -> list:
    """Every n-th tile, so each shard gets a mix of dense city and sparse outskirts."""
    return list(iter_tiles(boundary))[shard::n]


def build_dense_shard(
    token: str,
    embedder,
    out: str | Path,
    shard: int,
    n: int,
    boundary: Boundary | None = None,
    subcell_deg: float = DENSE_SUBCELL_DEG,
    per_subcell: int = DENSE_PER_SUBCELL,
    limit: int | None = None,
    batch_size: int = 64,
    workers: int = 16,
    log=print,
) -> int:
    seen = search_tiles(token, shard_tiles(boundary, shard, n), boundary, log)
    chosen = select(list(seen.values()), subcell_deg, per_subcell)
    if limit:
        chosen = chosen[:limit]
    log(f"Shard {shard}/{n}: embedding {len(chosen)} photos (from {len(seen)} candidates)")

    embs, coords, seqs, ids = [], [], [], []
    session = requests.Session()
    # One pooled connection per download thread (the default pool of 10 discards the rest).
    session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers))
    tmp = Path(tempfile.mkdtemp())
    try:
        with ThreadPoolExecutor(workers) as pool:
            for start in range(0, len(chosen), batch_size):
                recs = chosen[start : start + batch_size]
                got = [r for r in pool.map(lambda rec: _download(session, rec, tmp), recs) if r]
                if got:
                    embs.append(embedder.embed_images([tmp / r["path"] for r in got]).astype(np.float16))
                    coords += [(r["lat"], r["lon"]) for r in got]
                    seqs += [r["sequence"] for r in got]
                    ids += [r["id"] for r in got]
                for f in tmp.iterdir():
                    f.unlink()
                if (start // batch_size) % 20 == 0:
                    log(f"  {start + len(recs)}/{len(chosen)} photos")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if not embs:
        raise RuntimeError(f"Shard {shard}: no photos could be downloaded")
    np.savez(
        out,
        embeddings=np.concatenate(embs),
        coords=np.asarray(coords, dtype=np.float64),
        sequences=np.asarray(seqs),
        paths=np.asarray(ids),
        model=np.asarray(embedder.model_name),
        offset=np.asarray(shard),
    )
    return len(ids)
