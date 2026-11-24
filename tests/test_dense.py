import numpy as np
from PIL import Image

import sg_geolocator.dense as dense
from sg_geolocator.evaluate import evaluate_index, to_markdown
from sg_geolocator.retrieval import merge_indexes


class FakePlaceEmbedder:
    model_name = "eigenplaces/ResNet50/512"

    def embed_images(self, paths):
        a = np.array([[float(Image.open(p).getpixel((0, 0))[0]) + 1.0, 1.0, 0.5] for p in paths])
        return a / np.linalg.norm(a, axis=1, keepdims=True)


def _fake_mapillary(monkeypatch, n_photos=50):
    recs = {str(i): {"id": str(i), "lat": 1.30 + i * 0.001, "lon": 103.80, "sequence": f"s{i}",
                     "captured_at": "", "url": "u"} for i in range(n_photos)}
    monkeypatch.setattr(dense, "search_tiles", lambda token, tiles, boundary, log: dict(recs))

    def fake_download(session, rec, out_dir):
        if rec["id"] == "7":
            return None  # a failed download must just be skipped
        Image.new("RGB", (16, 12), (int(rec["id"]) * 4, 0, 0)).save(out_dir / f"{rec['id']}.jpg")
        return {**{k: rec[k] for k in ("id", "lat", "lon", "sequence", "captured_at")}, "path": f"{rec['id']}.jpg"}

    monkeypatch.setattr(dense, "_download", fake_download)


def test_dense_shard_embeds_in_batches_and_keeps_only_descriptors(tmp_path, monkeypatch):
    _fake_mapillary(monkeypatch)
    n = dense.build_dense_shard("tok", FakePlaceEmbedder(), tmp_path / "p.npz", shard=1, n=2,
                                subcell_deg=0.0005, per_subcell=1, batch_size=8, log=lambda *_: None)
    d = np.load(tmp_path / "p.npz")
    assert n == 49 and d["embeddings"].shape == (49, 3) and d["embeddings"].dtype == np.float16
    assert str(d["model"]) == "eigenplaces/ResNet50/512" and int(d["offset"]) == 1
    assert "7" not in set(d["paths"])


def test_dense_shards_merge(tmp_path, monkeypatch):
    _fake_mapillary(monkeypatch, n_photos=10)
    parts = []
    for i in (1, 0):
        p = tmp_path / f"part{i}.npz"
        dense.build_dense_shard("tok", FakePlaceEmbedder(), p, shard=i, n=2, subcell_deg=0.0005,
                                per_subcell=1, limit=3, log=lambda *_: None)
        parts.append(p)
    assert merge_indexes(parts, tmp_path / "places.npz") == 6


def test_evaluate_dense_options(tmp_path):
    # Every place photographed twice, from two capture sessions, a few metres apart.
    rng = np.random.default_rng(0)
    n = 400
    coords = np.column_stack([rng.uniform(1.25, 1.45, n), rng.uniform(103.65, 104.0, n)])
    place = rng.normal(size=(n, 16))
    emb = np.concatenate([place, place + rng.normal(scale=0.05, size=place.shape)])
    emb = (emb / np.linalg.norm(emb, axis=1, keepdims=True)).astype(np.float16)
    np.savez(tmp_path / "d.npz", embeddings=emb, coords=np.concatenate([coords, coords + 0.00002]),
             sequences=np.array([f"a{i}" for i in range(n)] + [f"b{i}" for i in range(n)]),
             paths=np.array(["x"] * 2 * n), model=np.asarray("eigenplaces/ResNet50/512"))

    res = evaluate_index(str(tmp_path / "d.npz"), geocell=False, radius_km=0.1, max_test=30)
    m = res["methods"]
    # Some places have both photos in the test split (no twin in the gallery), so not 100%.
    assert m["top1"]["n"] == 30 and m["top1"]["within_0.05km"] >= 0.8
    assert not any(k.startswith("geocell") for k in m)
    sure = [k for k in m if "conf>=" in k]
    assert sure and 0 < m[sure[0]]["coverage"] <= 1
    assert res["model"] == "eigenplaces/ResNet50/512"
    assert "of queries" in to_markdown(res)


def test_hub_repos_are_pre_trusted_once(tmp_path):
    from types import SimpleNamespace

    from sg_geolocator.embedder import _trust_hub_repos

    fake_torch = SimpleNamespace(hub=SimpleNamespace(get_dir=lambda: str(tmp_path / "hub")))
    _trust_hub_repos(fake_torch, ["gmberton_eigenplaces", "gmberton_cosplace"])
    _trust_hub_repos(fake_torch, ["gmberton_cosplace"])
    assert (tmp_path / "hub" / "trusted_list").read_text().split() == ["gmberton_eigenplaces", "gmberton_cosplace"]


def test_pca_merge_keeps_neighbours_and_projects_queries(tmp_path):
    from sg_geolocator.retrieval import RetrievalIndex

    # 300 places in 64-d whose variation lives in 8 directions; two shards.
    rng = np.random.default_rng(1)
    basis = np.linalg.qr(rng.normal(size=(64, 8)))[0]
    place = rng.normal(size=(300, 8)) @ basis.T + 0.3
    emb = place / np.linalg.norm(place, axis=1, keepdims=True)
    coords = np.column_stack([np.linspace(1.25, 1.45, 300), np.full(300, 103.8)])
    parts = []
    for i, sl in enumerate((slice(0, 150), slice(150, 300))):
        p = tmp_path / f"part{i}.npz"
        np.savez(p, embeddings=emb[sl].astype(np.float16), coords=coords[sl], sequences=np.array(["s"] * 150),
                 paths=np.array(["x"] * 150), model=np.asarray("megaloc"), offset=np.asarray(i))
        parts.append(p)
    assert merge_indexes(parts, tmp_path / "m.npz", pca_dims=8, log=lambda *_: None) == 300
    idx = RetrievalIndex(tmp_path / "m.npz")
    assert idx.embeddings.shape == (300, 8) and idx.pca is not None
    query = emb[123] + rng.normal(scale=0.01, size=64)  # a new photo of place 123, full-size descriptor
    q = idx.project(query / np.linalg.norm(query))
    assert q.shape == (8,) and abs(np.linalg.norm(q) - 1) < 1e-5
    assert int(np.argmax(idx.embeddings @ q)) == 123


def test_megaloc_is_picked_by_name(monkeypatch):
    import sg_geolocator.embedder as E

    monkeypatch.setattr(E.MegaLocEmbedder, "__init__", lambda self, name=E.MEGALOC: setattr(self, "model_name", name))
    assert isinstance(E.make_embedder("megaloc"), E.MegaLocEmbedder)
