"""Geotagged Wikimedia Commons photos of Singapore, as extra place-recognition references.

Mapillary photos are taken from cars and bikes on roads. Commons photos are taken by
people on foot, from balconies and from across fields, and are often of exactly the
buildings people recognise: schools, MRT stations, malls, community centres, HDB blocks.
They are added to the dense place index so a photo of a building can match a Commons
photo of the same building. Only descriptors and page ids are kept; photos are deleted
after embedding. Licences vary per photo (all free licences); see each file page.
"""

from __future__ import annotations

import time

import requests
from urllib3.util.retry import Retry

from .geo import SG_BOUNDS, Boundary

API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "sg-geolocator/0.1 (https://github.com/sudoaptg/sg-geolocator; research use)"}
THUMB_WIDTH = 500  # a standard thumbnail step, so Wikimedia serves it from cache
GRID_DEG = 0.01  # ~1.1 km search points
MAX_RESULTS = 500  # geosearch limit; a full page means the area is subdivided


def session(workers: int = 4) -> requests.Session:
    """Session that backs off on rate limits (429) and transient errors."""
    s = requests.Session()
    s.headers.update(HEADERS)
    retry = Retry(total=6, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504], respect_retry_after_header=True)
    s.mount("https://", requests.adapters.HTTPAdapter(max_retries=retry, pool_connections=workers, pool_maxsize=workers))
    return s


def _geosearch(s: requests.Session, lat: float, lon: float, radius_m: int) -> list[dict]:
    r = s.get(API, params={"action": "query", "list": "geosearch", "gscoord": f"{lat}|{lon}", "gsradius": radius_m,
                           "gslimit": MAX_RESULTS, "gsnamespace": 6, "format": "json"}, timeout=60)
    r.raise_for_status()
    return r.json().get("query", {}).get("geosearch", [])


def search(boundary: Boundary | None = None, log=print, s: requests.Session | None = None) -> dict[int, dict]:
    """pageid -> {title, lat, lon} for every geotagged file in Singapore."""
    s = s or session()
    found: dict[int, dict] = {}
    todo = []
    b = SG_BOUNDS
    lat = b["lat_min"] + GRID_DEG / 2
    while lat < b["lat_max"]:
        lon = b["lon_min"] + GRID_DEG / 2
        while lon < b["lon_max"]:
            h = GRID_DEG / 2
            if boundary is None or boundary.intersects_box(lon - h, lat - h, lon + h, lat + h):
                todo.append((lat, lon, GRID_DEG))
            lon += GRID_DEG
        lat += GRID_DEG
    done = 0
    while todo:
        lat, lon, size = todo.pop()
        # Circle covering the square cell.
        hits = _geosearch(s, lat, lon, int(size * 111_000 * 0.71) + 1)
        if len(hits) >= MAX_RESULTS and size > 0.001:
            q = size / 4
            todo += [(lat + dy, lon + dx, size / 2) for dy in (-q, q) for dx in (-q, q)]
            continue
        for h in hits:
            if boundary is None or boundary.contains(h["lat"], h["lon"]):
                found[h["pageid"]] = {"title": h["title"], "lat": h["lat"], "lon": h["lon"]}
        done += 1
        if done % 100 == 0:
            log(f"  searched {done} areas, {len(found)} files ({len(todo)} areas left)")
        time.sleep(0.05)
    log(f"Commons: {len(found)} geotagged files in Singapore")
    return found


def records(found: dict[int, dict], log=print, s: requests.Session | None = None) -> list[dict]:
    """Thumbnail URL and uploader for each file; non-photos (maps, diagrams, PDFs) dropped."""
    s = s or session()
    ids = sorted(found)
    out = []
    for i in range(0, len(ids), 50):
        batch = ids[i : i + 50]
        r = s.get(API, params={"action": "query", "pageids": "|".join(map(str, batch)), "prop": "imageinfo",
                               "iiprop": "url|user|mime|timestamp", "iiurlwidth": THUMB_WIDTH, "format": "json"},
                  timeout=60)
        r.raise_for_status()
        for pid, page in r.json().get("query", {}).get("pages", {}).items():
            info = (page.get("imageinfo") or [{}])[0]
            if info.get("mime") != "image/jpeg" or not info.get("thumburl"):
                continue
            meta = found[int(pid)]
            out.append({"id": f"commons:{pid}", "lat": meta["lat"], "lon": meta["lon"],
                        # Hold out by uploader: one person's photos of a place are near-duplicates.
                        "sequence": f"commons-user:{info.get('user', '')}",
                        "captured_at": info.get("timestamp", ""), "url": info["thumburl"]})
        if (i // 50) % 50 == 0:
            log(f"  file info {i + len(batch)}/{len(ids)}")
        time.sleep(0.05)
    log(f"Commons: {len(out)} photos (JPEG) to embed")
    return out
