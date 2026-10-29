import numpy as np

from sg_geolocator.evaluate import evaluate_index, split, to_markdown
from sg_geolocator.geo import Boundary
from sg_geolocator.mapillary import select, to_record

SQUARE = Boundary([[[(103.6, 1.2), (104.0, 1.2), (104.0, 1.45), (103.6, 1.45), (103.6, 1.2)]]])


def test_boundary_contains_and_box():
    assert SQUARE.contains(1.3, 103.8)
    assert not SQUARE.contains(1.47, 103.8)
    assert SQUARE.intersects_box(103.95, 1.40, 104.05, 1.50)  # straddles the edge
    assert not SQUARE.intersects_box(104.05, 1.30, 104.06, 1.31)


def test_boundary_from_geojson(tmp_path):
    p = tmp_path / "b.geojson"
    p.write_text(
        '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},'
        '"geometry":{"type":"MultiPolygon","coordinates":[[[[103.6,1.2],[104.0,1.2],[104.0,1.45],[103.6,1.2]]]]}}]}'
    )
    b = Boundary.from_geojson(p)
    assert b.contains(1.25, 103.9)
    assert not b.contains(1.4, 103.65)


def test_to_record_skips_panoramas_and_missing_urls():
    base = {"id": 1, "geometry": {"coordinates": [103.8, 1.3]}, "sequence": "s", "thumb_1024_url": "u"}
    assert to_record(base)["lat"] == 1.3
    assert to_record(base | {"is_pano": True}) is None
    assert to_record({k: v for k, v in base.items() if k != "thumb_1024_url"}) is None


def test_select_thins_per_cell_and_sequence():
    recs = [{"id": str(i), "lat": 1.3001, "lon": 103.8001, "sequence": "a" if i < 5 else "b"} for i in range(10)]
    chosen = select(recs, subcell_deg=0.002, per_subcell=3)
    assert len(chosen) == 2  # one per sequence in the single cell
    assert {r["sequence"] for r in chosen} == {"a", "b"}


def test_split_keeps_sequences_together():
    seqs = np.array([f"seq{i // 5}" for i in range(1000)])
    test = split(seqs, 0.1)
    for s in np.unique(seqs):
        assert len(set(test[seqs == s])) == 1
    assert 0.03 < test.mean() < 0.2


def test_evaluate_index_beats_baseline(tmp_path):
    # Synthetic index where an embedding encodes its location, so retrieval should be accurate.
    rng = np.random.default_rng(0)
    n = 600
    coords = np.column_stack([rng.uniform(1.25, 1.45, n), rng.uniform(103.65, 104.0, n)])
    feats = np.column_stack([coords - coords.mean(axis=0), rng.normal(scale=1e-3, size=(n, 6))])
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    # Duplicate each photo into its own sequence of two near-identical frames.
    emb = np.repeat(feats, 2, axis=0).astype(np.float32)
    path = tmp_path / "i.npz"
    np.savez(path, embeddings=emb, coords=np.repeat(coords, 2, axis=0),
             sequences=np.repeat([f"s{i}" for i in range(n)], 2), paths=np.array(["x"] * 2 * n))

    result = evaluate_index(str(path))
    m = result["methods"]
    assert m["baseline_centre"]["median_km"] > 5
    for name in ("knn10_cluster", "top1", "geocell_0.02deg"):
        assert m[name]["n"] > 0
        assert m[name]["median_km"] < m["baseline_centre"]["median_km"], name
    assert result["best"] in m
    assert "| baseline_centre |" in to_markdown(result)


def test_cluster_ignores_outlier_neighbours():
    from sg_geolocator.retrieval import RetrievalIndex

    # Three near-identical matches at A, two slightly weaker at far-away B and C.
    emb = np.array([[1, 0], [1, 0.01], [1, 0.02], [0.9, 0.3], [0.9, -0.3]], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    coords = np.array([[1.30, 103.80], [1.301, 103.801], [1.299, 103.80], [1.44, 103.99], [1.25, 103.62]])
    idx = RetrievalIndex(embeddings=emb, coords=coords)
    q = np.array([1.0, 0.0], dtype=np.float32)

    cluster = idx.predict(q, k=5, temperature=1.0, method="cluster")
    mean = idx.predict(q, k=5, temperature=1.0, method="mean")
    assert abs(cluster.lat - 1.30) < 0.002 and abs(cluster.lon - 103.80) < 0.002
    assert abs(mean.lat - 1.30) > abs(cluster.lat - 1.30)  # the mean gets dragged away
    assert 0.5 < cluster.confidence <= 1.0


def test_geocell_learns_location_direction():
    from sg_geolocator.geocell import GeoCellClassifier

    rng = np.random.default_rng(0)
    coords = np.column_stack([rng.uniform(1.25, 1.45, 2000), rng.uniform(103.65, 104.0, 2000)])
    X = np.column_stack([coords - coords.mean(axis=0), rng.normal(scale=0.01, size=(2000, 4))])
    clf = GeoCellClassifier(cell_deg=0.05, steps=300, lr=0.05).fit(X[:1800], coords[:1800])
    pred, conf = clf.predict(X[1800:])
    err = np.abs(pred - coords[1800:]).max(axis=1)
    assert np.median(err) < 0.05 and conf.shape == (200,)


def test_search_splits_failing_tile_and_skips_smallest(monkeypatch):
    from sg_geolocator import mapillary as m

    client = m.MapillaryClient("tok", log=lambda *_: None)
    calls = []

    def fake_get(params, attempts=4):
        calls.append(params["bbox"])
        x0 = float(params["bbox"].split(",")[0])
        if len(calls) == 1 or x0 == 103.8:  # whole tile fails, and its west-most chain keeps failing
            raise m.RequestFailed("HTTP 500")
        return [{"id": params["bbox"]}]

    monkeypatch.setattr(client, "_get", fake_get)
    out = client.search((103.8, 1.3, 103.81, 1.31))
    assert out and client.skipped_tiles > 0


def test_as_array_handles_both_transformers_return_types():
    from types import SimpleNamespace

    from sg_geolocator.embedder import _as_array

    class FakeTensor:
        def __init__(self, a):
            self.a = a

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    a = np.ones((2, 4))
    assert _as_array(FakeTensor(a)) is a  # transformers 4.x
    assert _as_array(SimpleNamespace(pooler_output=FakeTensor(a))) is a  # transformers 5.x


def test_reduce_data_error_is_not_retried(monkeypatch):
    from sg_geolocator import mapillary as m

    client = m.MapillaryClient("tok", log=lambda *_: None)
    attempts = []

    class Resp:
        status_code = 500
        text = '{"error":{"code":1,"message":"Please reduce the amount of data you\'re asking for"}}'

    monkeypatch.setattr(client.session, "get", lambda *a, **k: attempts.append(1) or Resp())
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    try:
        client._get({"bbox": "0,0,1,1"})
    except m.RequestFailed:
        pass
    assert len(attempts) == 1


def test_sharded_build_merges_back_in_order(tmp_path):
    import csv

    from PIL import Image

    from sg_geolocator.retrieval import RetrievalIndex, build_index_from_csv, merge_indexes

    (tmp_path / "images").mkdir()
    rows = []
    for i in range(23):
        Image.new("RGB", (8, 8), (i * 10, 0, 0)).save(tmp_path / "images" / f"{i}.jpg")
        rows.append({"id": str(i), "lat": 1.3 + i / 1000, "lon": 103.8, "sequence": f"s{i}", "captured_at": "", "path": f"{i}.jpg"})
    with open(tmp_path / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    class FakeEmb:
        model_name = "fake/clip"

        def embed_images(self, paths):
            a = np.array([[np.asarray(Image.open(p))[0, 0, 0] + 1.0, 1.0] for p in paths])
            return a / np.linalg.norm(a, axis=1, keepdims=True)

    parts = []
    for i in (2, 0, 1):  # shards may finish in any order
        part = tmp_path / f"part{i}.npz"
        build_index_from_csv(tmp_path / "metadata.csv", FakeEmb(), part, batch_size=4, log=None, shard=(i, 3))
        parts.append(part)
    assert merge_indexes(parts, tmp_path / "index.npz") == 23
    merged = np.load(tmp_path / "index.npz")
    assert list(merged["paths"]) == [f"{i}.jpg" for i in range(23)]
    assert np.allclose(merged["coords"][:, 0], [1.3 + i / 1000 for i in range(23)])
    assert RetrievalIndex(tmp_path / "index.npz").model == "fake/clip"


def test_geocell_save_load_roundtrip(tmp_path):
    from sg_geolocator.geocell import GeoCellClassifier

    rng = np.random.default_rng(1)
    coords = np.column_stack([rng.uniform(1.25, 1.45, 300), rng.uniform(103.65, 104.0, 300)])
    X = np.column_stack([coords - coords.mean(axis=0), rng.normal(scale=0.01, size=(300, 3))])
    clf = GeoCellClassifier(cell_deg=0.05, steps=50).fit(X, coords)
    clf.save(tmp_path / "g.npz", model="geolocal/StreetCLIP")
    back = GeoCellClassifier.load(tmp_path / "g.npz")
    assert back.model == "geolocal/StreetCLIP"
    assert np.allclose(back.predict(X[:5])[0], clf.predict(X[:5])[0])


def test_index_without_model_field_defaults_to_base_clip(tmp_path):
    from sg_geolocator.embedder import DEFAULT_MODEL
    from sg_geolocator.retrieval import RetrievalIndex

    np.savez(tmp_path / "old.npz", embeddings=np.eye(2, dtype=np.float32), coords=np.zeros((2, 2)))
    assert RetrievalIndex(tmp_path / "old.npz").model == DEFAULT_MODEL
