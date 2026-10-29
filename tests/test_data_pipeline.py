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
    assert result["retrieval"]["n"] > 0
    assert result["baseline_centre"]["median_km"] > 5
    assert "| retrieval |" in to_markdown(result)


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
