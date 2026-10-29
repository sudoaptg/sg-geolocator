"""Collect geotagged street-level reference photos of Singapore from Mapillary.

Needs a Mapillary access token (https://www.mapillary.com/dashboard/developers).
Imagery is CC BY-SA 4.0: credit Mapillary and its contributors when you publish results.
"""

from __future__ import annotations

import csv
import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

from .geo import SG_BOUNDS, Boundary

API_URL = "https://graph.mapillary.com/images"
FIELDS = "id,geometry,sequence,captured_at,is_pano,thumb_1024_url"
LIMIT = 2000  # max results the search endpoint returns per request
CELL_DEG = 0.01  # ~1.1 km search tiles; small tiles keep requests under the bbox size cap
MAX_SPLIT_DEPTH = 3
SAVE_MAX_SIDE = 512  # CLIP only sees 224 px, so keep files small

CSV_FIELDS = ["id", "lat", "lon", "sequence", "captured_at", "path"]


class RequestFailed(Exception):
    pass


class MapillaryClient:
    def __init__(self, token: str, pause: float = 0.05, log=print):
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"OAuth {token}"
        self.pause = pause
        self.log = log
        self.skipped_tiles = 0

    def _get(self, params: dict, attempts: int = 4) -> list[dict]:
        last = ""
        for attempt in range(attempts):
            try:
                r = self.session.get(API_URL, params=params, timeout=90)
            except requests.RequestException as exc:
                last = type(exc).__name__
            else:
                if r.status_code == 200:
                    time.sleep(self.pause)
                    return r.json().get("data", [])
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                if r.status_code in (401, 403):
                    raise RuntimeError(f"Mapillary rejected the token ({last})")
                if r.status_code == 400 or "reduce the amount of data" in r.text:
                    break  # retrying won't help, but a smaller bbox will
                if r.status_code == 429:
                    time.sleep(60)  # rate limited: back off hard
                    continue
            time.sleep(2 ** (attempt + 1))
        raise RequestFailed(last)

    def search(self, bbox: tuple[float, float, float, float], depth: int = 0) -> list[dict]:
        """Images in bbox (min_lon, min_lat, max_lon, max_lat).

        Tiles that hit the result cap, or that the API keeps failing on (dense areas can
        time out), are split into quadrants. A tile that still fails at the smallest size
        is skipped with a warning instead of aborting the whole run."""
        params = {"bbox": ",".join(f"{v:.6f}" for v in bbox), "fields": FIELDS, "limit": LIMIT}
        try:
            data = self._get(params)
        except RequestFailed as exc:
            if depth >= MAX_SPLIT_DEPTH:
                self.skipped_tiles += 1
                self.log(f"  warning: skipping tile {params['bbox']} ({exc})")
                return []
            self.log(f"  tile {params['bbox']} failed ({exc}); splitting")
            data = None
        if data is not None and (len(data) < LIMIT or depth >= MAX_SPLIT_DEPTH):
            return data
        x0, y0, x1, y1 = bbox
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        out: list[dict] = []
        for q in ((x0, y0, xm, ym), (xm, y0, x1, ym), (x0, ym, xm, y1), (xm, ym, x1, y1)):
            out.extend(self.search(q, depth + 1))
        return out


def iter_tiles(boundary: Boundary | None, cell: float = CELL_DEG):
    """Search tiles covering Singapore; tiles entirely outside the boundary are skipped."""
    b = SG_BOUNDS
    lat = b["lat_min"]
    while lat < b["lat_max"]:
        lon = b["lon_min"]
        while lon < b["lon_max"]:
            tile = (lon, lat, lon + cell, lat + cell)
            if boundary is None or boundary.intersects_box(*tile):
                yield tile
            lon += cell
        lat += cell


def to_record(img: dict) -> dict | None:
    if img.get("is_pano") or not img.get("thumb_1024_url"):
        return None  # panoramas are distorted and look nothing like normal photos
    try:
        lon, lat = img["geometry"]["coordinates"]
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "id": str(img["id"]),
        "lat": float(lat),
        "lon": float(lon),
        "sequence": str(img.get("sequence", "")),
        "captured_at": img.get("captured_at", ""),
        "url": img["thumb_1024_url"],
    }


def select(records: list[dict], subcell_deg: float, per_subcell: int) -> list[dict]:
    """Thin dense coverage: keep at most `per_subcell` images per small grid square,
    each from a different sequence, so one long drive doesn't dominate an area."""
    buckets: dict[tuple[int, int], list[dict]] = {}
    for r in sorted(records, key=lambda r: r["id"]):
        key = (int(r["lat"] // subcell_deg), int(r["lon"] // subcell_deg))
        bucket = buckets.setdefault(key, [])
        if len(bucket) < per_subcell and all(x["sequence"] != r["sequence"] for x in bucket):
            bucket.append(r)
    return [r for bucket in buckets.values() for r in bucket]


def search_tiles(token: str, tiles: list, boundary: Boundary | None = None, log=print) -> dict[str, dict]:
    """All usable (non-panorama, on-land) images in `tiles`, keyed by image id."""
    client = MapillaryClient(token, log=log)
    log(f"Searching {len(tiles)} tiles")
    seen: dict[str, dict] = {}
    for i, tile in enumerate(tiles, 1):
        for img in client.search(tile):
            rec = to_record(img)
            if rec and (boundary is None or boundary.contains(rec["lat"], rec["lon"])):
                seen[rec["id"]] = rec
        if i % 50 == 0:
            log(f"  {i}/{len(tiles)} tiles, {len(seen)} candidate images")
    if client.skipped_tiles:
        log(f"warning: {client.skipped_tiles} small tiles were skipped after repeated API errors")
    if not seen:
        raise RuntimeError("Mapillary returned no usable images")
    return seen


def _download(session: requests.Session, rec: dict, out_dir: Path) -> dict | None:
    path = out_dir / f"{rec['id']}.jpg"
    if not path.exists():
        try:
            r = session.get(rec["url"], timeout=60)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            img.thumbnail((SAVE_MAX_SIDE, SAVE_MAX_SIDE))
            img.save(path, quality=90)
        except Exception:
            return None
    return {k: rec[k] for k in CSV_FIELDS if k != "path"} | {"path": path.name}


def collect(
    token: str,
    out_dir: str | Path,
    boundary: Boundary | None = None,
    subcell_deg: float = 0.002,
    per_subcell: int = 2,
    max_images: int | None = None,
    workers: int = 16,
    log=print,
) -> Path:
    """Search, thin, download and write `metadata.csv`. Returns the CSV path."""
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    seen = search_tiles(token, list(iter_tiles(boundary)), boundary, log)
    chosen = select(list(seen.values()), subcell_deg, per_subcell)
    if max_images and len(chosen) > max_images:
        step = len(chosen) / max_images
        chosen = [chosen[int(i * step)] for i in range(max_images)]
    log(f"Downloading {len(chosen)} images (from {len(seen)} candidates)")

    dl = requests.Session()  # thumbnail URLs are pre-signed; no auth header
    dl.mount("https://", requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers))
    with ThreadPoolExecutor(workers) as pool:
        rows = [r for r in pool.map(lambda rec: _download(dl, rec, img_dir), chosen) if r]

    csv_path = out_dir / "metadata.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    log(f"Saved {len(rows)} images -> {csv_path}")
    return csv_path
