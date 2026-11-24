"""Look at outside data sources from a machine with open internet (a GitHub runner) and
save raw copies of what is useful into ./out for publishing as a release.

    python scripts/probe_sources.py hdb heights ocr megaloc

Each part prints a short report and never stops the others.
"""

from __future__ import annotations

import gzip
import io
import json
import re
import sys
import time
import traceback
from pathlib import Path

import requests

OUT = Path("out")
SG = (103.59, 1.15, 104.10, 1.48)  # west, south, east, north
UA = {"User-Agent": "sg-geolocator-data-probe/0.1"}
DATAGOV = "https://api-open.data.gov.sg/v1/public/api/datasets/{}/{}"


def say(*a):
    print(*a, flush=True)


def get(url, **kw):
    kw.setdefault("timeout", 120)
    kw.setdefault("headers", UA)
    return requests.get(url, **kw)


# --------------------------------------------------------------------------- data.gov.sg


def datagov_file(dataset: str) -> bytes:
    """A data.gov.sg dataset's file (poll-download, initiating the export when needed)."""
    meta = get(DATAGOV.format(dataset, "metadata")).json()
    say(f"  metadata {dataset}: {json.dumps(meta)[:600]}")
    for attempt in range(12):
        r = get(DATAGOV.format(dataset, "poll-download")).json()
        data = r.get("data") or {}
        if r.get("code") == 0 and data.get("url"):
            return get(data["url"], timeout=900).content
        say(f"  poll {attempt}: {json.dumps(r)[:300]}")
        for method in (requests.get, requests.post):
            try:
                method(DATAGOV.format(dataset, "initiate-download"), headers=UA, timeout=60)
                break
            except requests.RequestException:
                pass
        time.sleep(5 + 5 * attempt)
    raise RuntimeError(f"no download for {dataset}")


def hdb():
    """HDB Existing Building (block outlines) and HDB Property Information (storeys, year)."""
    for name, ds in (("hdb_existing_building.geojson", "d_16b157c52ed637edd6ba1232e026258d"),
                     ("hdb_property_information.csv", "d_17f5382f26140b1fdae0ba2ef6239d2f")):
        try:
            raw = datagov_file(ds)
            say(f"  {name}: {len(raw) / 1e6:.1f} MB")
            if name.endswith(".geojson"):
                g = json.loads(raw)
                feats = g.get("features", [])
                say(f"  {len(feats)} features; first properties:")
                for f in feats[:3]:
                    say("   ", json.dumps(f.get("properties"))[:700], f.get("geometry", {}).get("type"))
            else:
                say("  " + raw[:600].decode("utf-8", "replace").replace("\n", "\n  "))
            with gzip.open(OUT / (name + ".gz"), "wb") as f:
                f.write(raw)
        except Exception:
            traceback.print_exc()


# --------------------------------------------------------------------------- heights


def _hf_tree(repo: str, path: str = "") -> list[dict]:
    out, url = [], f"https://huggingface.co/api/datasets/{repo}/tree/main/{path}".rstrip("/")
    params = {"recursive": "true", "expand": "false"}
    while url:
        r = get(url, params=params)
        r.raise_for_status()
        out += r.json()
        nxt = r.links.get("next", {}).get("url")
        url, params = nxt, None
    return out


def _tiles_for_sg(names):
    """Names that look like they cover lon 100-105 / lat 0-5 (Singapore)."""
    pats = [r"e100[_-]?n0?5", r"e100[_-]?n0?0", r"e10[0-4]", r"E10[0-4]", r"100E", r"N0?0[_-]?E10", r"n0?1[_-]?e10"]
    return [n for n in names if any(re.search(p, n, re.I) for p in pats)]


SG_TILE = "e100_n05_e105_n00"  # lon 100-105, lat 0-5


def _download(url: str, path: Path) -> Path:
    with requests.get(url, headers=UA, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)
    say(f"  downloaded {url.split('/')[-1]}: {path.stat().st_size / 1e6:.0f} MB")
    return path


def _feature_id(feat: dict):
    props = feat.get("properties") or {}
    for k in ("id", "ID", "Id", "fid", "building_id", "osm_id", "source_id"):
        if k in props:
            return str(props[k])
    return str(feat.get("id")) if feat.get("id") is not None else None


def _sg_polygons(path: Path, label: str) -> dict:
    """id -> (ring as [lat, lon] rows, source) for features inside Singapore's box."""
    import ijson

    out, n = {}, 0
    with open(path, "rb") as f:
        for feat in ijson.items(f, "features.item", use_float=True):
            n += 1
            if n <= 2:
                say(f"    {label} feature: {json.dumps(feat)[:700]}")
            rings = _rings(feat.get("geometry") or {})
            if not rings:
                continue
            ring = rings[0]
            x, y = ring[0][0], ring[0][1]
            if not (SG[0] <= x <= SG[2] and SG[1] <= y <= SG[3]):
                continue
            fid = _feature_id(feat)
            if fid is not None:
                out[fid] = ([[round(p[1], 6), round(p[0], 6)] for p in ring], label)
    say(f"  {label}: {n} features in the tile, {len(out)} in Singapore's box")
    return out


def gba():
    """GlobalBuildingAtlas: LoD1 heights (id -> height) joined with the outlines of the
    Singapore tile (Part II polygons in vitrobot/GBA.LoD1, OSM-derived Part I in
    zhu-xlab/GBA.ODbLPolygon). Writes out/gba_sg.json.gz: [[ring lat/lon], height, source] rows."""
    import ijson

    polys = {}
    for repo in ("vitrobot/GBA.LoD1", "zhu-xlab/GBA.ODbLPolygon"):
        try:
            files = _hf_tree(repo)
        except Exception as exc:
            say(f"  HF {repo}: {exc}")
            continue
        names = {f["path"]: f.get("size") for f in files if f.get("type") == "file"}
        hits = {n: sz for n, sz in names.items() if SG_TILE in n}
        say(f"  HF {repo}: {len(names)} files; Singapore tile: {hits}")
        for n in hits:
            if n.startswith("LoD1/"):
                continue
            path = _download(f"https://huggingface.co/datasets/{repo}/resolve/main/{n}", Path("gba_poly.json"))
            polys.update(_sg_polygons(path, repo.split("/")[0] + ":" + n.split("/")[0]))
            path.unlink()
    lod1 = next((n for n in ("LoD1/oceania/" + SG_TILE + ".json",)), None)
    path = _download(f"https://huggingface.co/datasets/vitrobot/GBA.LoD1/resolve/main/{lod1}", Path("gba_lod1.json"))
    heights, n, sgp = {}, 0, 0
    with open(path, "rb") as f:
        for key, val in ijson.kvitems(f, "", use_float=True):
            n += 1
            if key.endswith("SGP"):
                sgp += 1
            if key in polys:
                heights[key] = val.get("height")
    path.unlink()
    say(f"  LoD1: {n} heights in the tile ({sgp} with a Singapore id); matched to {len(heights)} of {len(polys)} outlines")
    unmatched = [k for k in polys if k not in heights][:5]
    say(f"  sample unmatched outline ids: {unmatched}; sample matched: {list(heights)[:3]}")
    out = [[polys[k][0], h, polys[k][1]] for k, h in heights.items() if h is not None]
    if out:
        hs = sorted(o[1] for o in out)
        say(f"  height percentiles 25/50/75/95/99: {[round(hs[int(len(hs) * q)], 1) for q in (.25, .5, .75, .95, .99)]}")
        with gzip.open(OUT / "gba_sg.json.gz", "wt") as f:
            json.dump(out, f)


def _rings(geom):
    if geom["type"] == "Polygon":
        return geom["coordinates"]
    if geom["type"] == "MultiPolygon":
        return [r for poly in geom["coordinates"] for r in poly]
    return []


def globfp():
    for rec in ("11397014", "15487006"):
        try:
            r = _retry_json(f"https://zenodo.org/api/records/{rec}")
            say(f"  zenodo {rec}: {r.get('metadata', {}).get('title')} | version {r.get('metadata', {}).get('version')}")
            files = r.get("files", [])
            say(f"  {len(files)} files: {[(f['key'], f['size']) for f in files][:40]}")
            for f in files:
                if f["key"].lower().endswith(".txt"):
                    txt = _retry_text(f["links"]["self"])
                    lines = txt.splitlines()
                    say(f"  {f['key']}: {len(lines)} lines; first: {lines[:5]}")
                    say(f"  lines naming Asia/E10x: {[x for x in lines if re.search(r'asia|e10[0-5]|103', x, re.I)][:20]}")
        except Exception:
            traceback.print_exc()


def _retry_text(url, tries=5):
    for k in range(tries):
        try:
            r = get(url, timeout=180)
            if r.status_code == 200:
                return r.text
            say(f"  {url}: HTTP {r.status_code}")
        except requests.RequestException as exc:
            say(f"  {url}: {exc}")
        time.sleep(20 * (k + 1))
    return ""


def _retry_json(url, tries=5):
    return json.loads(_retry_text(url, tries) or "{}")


def msbuildings():
    try:
        csv = get("https://bfppub.blob.core.windows.net/$web/2026-11-12/dataset-links.csv", timeout=300).text
        rows = [x for x in csv.splitlines() if x.lower().startswith("singapore") or ",singapore," in x.lower()]
        say(f"  MS dataset-links: {len(csv.splitlines())} rows; Singapore rows: {rows[:10]}")
        heights, n = [], 0
        for row in rows[:6]:
            url = [c for c in row.split(",") if c.startswith("http")][0]
            raw = gzip.decompress(get(url, timeout=300).content).decode()
            for line in raw.splitlines():
                f = json.loads(line)
                n += 1
                h = (f.get("properties") or {}).get("height")
                if h is not None and h > 0:
                    heights.append(h)
        say(f"  MS Singapore: {n} footprints, {len(heights)} with a height; "
            f"heights over 30 m: {sum(h > 30 for h in heights)}")
        if rows:
            (OUT / "ms_buildings_sg.txt").write_text("\n".join(rows))
    except Exception:
        traceback.print_exc()


def ghsl():
    """GHS-BUILT-H (average net building height, 100 m, 2018): the tile file names."""
    try:
        url = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_H_GLOBE_R2023A/"
        say("  GHSL: " + re.sub(r"<[^>]+>", " ", get(url).text)[:800])
    except Exception:
        traceback.print_exc()


def heights():
    for part in (gba,):
        say(f"\n== {part.__name__}")
        part()


# --------------------------------------------------------------------------- OCR / MegaLoc


def ocr():
    from rapidocr import RapidOCR

    photos = sorted(Path("benchmarks/photos").glob("*")) + sorted(Path("benchmarks/initial").glob("*"))
    from rapidocr import LangDet, LangRec, ModelType, OCRVersion

    v5 = {"Det.ocr_version": OCRVersion.PPOCRV5, "Rec.ocr_version": OCRVersion.PPOCRV5,
          "Det.lang_type": LangDet.CH, "Rec.lang_type": LangRec.CH}
    for label, params in (("PP-OCRv5 server", {**v5, "Det.model_type": ModelType.SERVER, "Rec.model_type": ModelType.SERVER}),
                          ("PP-OCRv5 mobile", {**v5, "Det.model_type": ModelType.MOBILE, "Rec.model_type": ModelType.MOBILE}),
                          ("PP-OCRv6 medium", {"Det.model_type": ModelType.MEDIUM, "Rec.model_type": ModelType.MEDIUM})):
        try:
            eng = RapidOCR(params=params)
            for p in photos:
                from PIL import Image
                import numpy as np

                img = Image.open(p).convert("RGB")
                if max(img.size) < 1600:
                    s = 1600 / max(img.size)
                    img = img.resize((round(img.width * s), round(img.height * s)))
                t = time.time()
                res = eng(np.asarray(img)[:, :, ::-1])
                items = []
                for box, txt, sc in zip(res.boxes if res.boxes is not None else [], res.txts or [], res.scores or []):
                    b = np.asarray(box)
                    items.append(f"{txt!r}@{b[:, 0].mean() / img.width:.2f},{b[:, 1].mean() / img.height:.2f}"
                                 f"({sc:.2f},{(b[:, 1].max() - b[:, 1].min()) / max(1, b[:, 0].max() - b[:, 0].min()):.1f}h/w)")
                say(f"  [{label}] {p.name} {time.time() - t:.1f}s: {' '.join(items)[:1200]}")
            # Stacked digits on the Bishan block (x ~ 0.82 across): narrow upscaled column crops.
            img = Image.open("benchmarks/photos/bishan-221.webp").convert("RGB")
            W, H = img.size
            for x0, x1 in ((0.78, 0.87), (0.80, 0.85), (0.76, 0.90)):
                crop = img.crop((int(W * x0), 0, int(W * x1), int(H * 0.6)))
                crop = crop.resize((crop.width * 4, crop.height * 4), Image.LANCZOS)
                res = eng(np.asarray(crop)[:, :, ::-1])
                say(f"  [{label}] bishan column {x0}-{x1}: {list(zip(res.txts or [], [round(float(v), 2) for v in (res.scores or [])]))}")
        except Exception:
            traceback.print_exc()


def megaloc():
    try:
        import torch
        import torchvision.transforms as T
        from PIL import Image

        t = time.time()
        model = torch.hub.load("gmberton/MegaLoc", "get_trained_model", trust_repo=True, skip_validation=True).eval()
        say(f"  MegaLoc loaded in {time.time() - t:.0f}s")
        tf = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        T.Resize([322, 322], antialias=True)])
        photos = sorted(Path("benchmarks/photos").glob("*"))[:6]
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in photos])
        with torch.inference_mode():
            t = time.time()
            d = model(x)
        say(f"  {tuple(d.shape)} in {time.time() - t:.1f}s for {len(photos)} images; norms {d.norm(dim=1)}")
        say(f"  similarity matrix:\n{(d @ d.T).numpy().round(2)}")
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name in sys.argv[1:]:
        say(f"\n######## {name}")
        globals()[name]()
