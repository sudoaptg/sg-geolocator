"""Locate a photo from text in it: postal codes, bus stop codes, block numbers, street and
MRT station names, looked up in Singapore gazetteers.

Singapore is unusually well indexed: every postal code belongs to one building and every
bus stop has a unique 5-digit code, so a single readable sign can pin a photo to within
tens of metres. Text is read with RapidOCR (free, runs on CPU) and resolved with:

- the public bus stop list (code -> lng, lat, name, road), derived from LTA DataMall
- the OneMap search API (Singapore Land Authority) for postal codes, addresses, roads and
  stations; works without a key, and uses ONEMAP_TOKEN from the environment if set
- the OpenStreetMap road network (see roads.py) to correct misread road names and to
  place a photo at the junction of two roads read in it
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .geo import haversine_km, in_singapore

ONEMAP_SEARCH = "https://www.onemap.gov.sg/api/common/elastic/search"
BUS_STOPS_URL = "https://data.busrouter.sg/v1/stops.min.json"
CACHE_DIR = Path(os.environ.get("SG_GEOLOCATOR_CACHE", Path.home() / ".cache" / "sg_geolocator"))

# How much to trust each kind of clue once it resolves to a place.
CONFIDENCE = {
    "postal": 0.95,
    "bus_stop": 0.9,
    "block": 0.9,
    "station": 0.75,
    "place": 0.8,  # a named school, mall, CC, ...: the sign is usually on or next to it
    "junction": 0.85,  # two road names read together, and those roads meet here
    "street": 0.5,  # roads can be kilometres long; we only get a point on them
}

ABBREVIATIONS = {
    "AVE": "AVENUE", "ST": "STREET", "RD": "ROAD", "DR": "DRIVE", "CRES": "CRESCENT",
    "LN": "LANE", "PL": "PLACE", "TER": "TERRACE", "CL": "CLOSE", "CTRL": "CENTRAL",
    "BT": "BUKIT", "JLN": "JALAN", "LOR": "LORONG", "KG": "KAMPONG", "UPP": "UPPER",
    "NTH": "NORTH", "STH": "SOUTH", "BLVD": "BOULEVARD", "PK": "PARK", "HTS": "HEIGHTS",
}
STREET_WORDS = (
    "ROAD|STREET|AVENUE|DRIVE|CRESCENT|LANE|PLACE|TERRACE|CLOSE|WAY|LINK|BOULEVARD|CENTRAL|"
    "WALK|RISE|GROVE|VIEW|HILL|GARDENS?|PARK|HEIGHTS|RING|LOOP|GREEN|VALE|"
    "AVE|ST|RD|DR|CRES|LN|PL|TER|CL|BLVD|CTRL"
)
STREET_PREFIXES = "JALAN|JLN|LORONG|LOR|UPPER|UPP|LOWER|BUKIT|BT|KAMPONG|KG|TANJONG"

POSTAL_RE = re.compile(r"(?:SINGAPORE|SPORE|S)\s*[\(\[]?\s*(\d{6})\b")
BARE_POSTAL_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")
BUS_RE = re.compile(r"(?<!\d)(\d{5})(?!\d)")
BLOCK_RE = re.compile(r"\b(?:BLK|BLOCK)\.?\s*(\d{1,4}[A-Z]?)\b")
STREET_RE = re.compile(
    rf"\b((?:(?:{STREET_PREFIXES})\s+)?[A-Z][A-Z']+(?:\s+[A-Z][A-Z']+)*?\s+(?:{STREET_WORDS})\b(?:\s+\d{{1,2}}[A-Z]?\b)?)"
    rf"|\b((?:{STREET_PREFIXES})\s+[A-Z][A-Z']+(?:\s+[A-Z][A-Z']+)?(?:\s+\d{{1,2}}\b)?)"
)
STATION_RE = re.compile(r"\b([A-Z][A-Z' ]{2,40}?)\s+(?:MRT|LRT)\b|\b([A-Z][A-Z' ]{2,40}?)\s+STATION\b")
VALID_POSTAL_SECTORS = {f"{i:02d}" for i in range(1, 83)}


@dataclass
class Clue:
    kind: str
    text: str
    lat: float
    lon: float
    confidence: float
    detail: str = ""


@dataclass
class ClueResult:
    lat: float
    lon: float
    confidence: float
    clues: list[Clue] = field(default_factory=list)
    text: list[str] = field(default_factory=list)

    @property
    def evidence(self) -> list[str]:
        return [f"{c.kind}: '{c.text}' -> {c.detail}" for c in self.clues]


# --------------------------------------------------------------------------- text reading


class TextReader:
    """RapidOCR wrapper. Small images are upscaled, since signs in a photo are often tiny.
    Large photos are also read in overlapping tiles: some RapidOCR versions shrink big
    images before detecting text, which loses distant signs (a school's name board a few
    pixels tall in a 4096 px photo)."""

    TILE = 1280
    STEP = 1024

    def __init__(self, min_score: float = 0.5):
        from rapidocr_onnxruntime import RapidOCR

        self.engine = RapidOCR()
        self.min_score = min_score

    def _run(self, img) -> list[str]:
        import numpy as np

        result, _ = self.engine(np.asarray(img)[:, :, ::-1])  # RapidOCR expects BGR
        return [text for _, text, score in (result or []) if float(score) >= self.min_score]

    def read(self, path: str | Path) -> list[str]:
        from PIL import Image

        img = Image.open(path).convert("RGB")
        if max(img.size) < 1600:
            scale = 1600 / max(img.size)
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        texts = self._run(img)
        if max(img.size) > self.TILE * 1.5:
            for top in _starts(img.height, self.TILE, self.STEP):
                for left in _starts(img.width, self.TILE, self.STEP):
                    texts += self._run(img.crop((left, top, left + self.TILE, top + self.TILE)))
        return list(dict.fromkeys(texts))


def _starts(size: int, tile: int, step: int) -> list[int]:
    """Tile offsets covering `size`, the last tile flush with the edge."""
    if size <= tile:
        return [0]
    out = list(range(0, size - tile, step))
    return out + [size - tile]


# --------------------------------------------------------------------------- gazetteers


class BusStops:
    def __init__(self, stops: dict[str, list]):
        self.stops = stops

    @classmethod
    def load(cls, path: str | Path | None = None) -> "BusStops":
        path = Path(path) if path else CACHE_DIR / "bus_stops.json"
        if not path.exists():
            import requests

            path.parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(BUS_STOPS_URL, timeout=60)
            r.raise_for_status()
            path.write_text(r.text)
        return cls(json.loads(path.read_text()))

    def lookup(self, code: str) -> tuple[float, float, str] | None:
        s = self.stops.get(code)
        if not s:
            return None
        lon, lat, name, road = s[:4]
        return float(lat), float(lon), f"bus stop {code} {name}, {road}"


class OneMap:
    """OneMap search with an on-disk cache, so repeated runs don't re-query."""

    def __init__(self, token: str | None = None, cache_path: str | Path | None = None, offline: bool = False):
        self.token = token or os.environ.get("ONEMAP_TOKEN")
        self.cache_path = Path(cache_path) if cache_path else CACHE_DIR / "onemap_cache.json"
        self.offline = offline
        try:
            self.cache = json.loads(self.cache_path.read_text())
        except (OSError, ValueError):
            self.cache = {}
        self._session = None

    def search(self, query: str, pages: int = 1) -> list[dict]:
        key = f"{query.upper()}|{pages}"
        if key in self.cache:
            return self.cache[key]
        if self.offline:
            return []
        import requests

        if self._session is None:
            self._session = requests.Session()
            if self.token:
                self._session.headers["Authorization"] = self.token
        results: list[dict] = []
        for page in range(1, pages + 1):
            params = {"searchVal": query, "returnGeom": "Y", "getAddrDetails": "Y", "pageNum": page}
            try:
                r = self._session.get(ONEMAP_SEARCH, params=params, timeout=20)
                data = r.json()
            except (requests.RequestException, ValueError):
                return results  # don't cache failures
            results.extend(data.get("results", []))
            if page >= int(data.get("totalNumPages") or 0):
                break
        self.cache[key] = results
        return results

    def save(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache))
        except OSError:
            pass


def _hits(results: list[dict], keep, label) -> list[tuple[float, float, str]]:
    """(lat, lon, description) for OneMap results passing `keep`, inside Singapore."""
    out = []
    for r in results:
        c = _coords(r)
        if c and keep(r):
            out.append((c[0], c[1], label(r)))
    return out


def _coords(result: dict) -> tuple[float, float] | None:
    try:
        lat = float(result["LATITUDE"])
        lon = float(result.get("LONGITUDE", result.get("LONGTITUDE")))
    except (KeyError, TypeError, ValueError):
        return None
    return (lat, lon) if in_singapore(lat, lon) else None


# --------------------------------------------------------------------------- clue extraction


def normalize_street(name: str) -> str:
    words = [ABBREVIATIONS.get(w, w) for w in name.upper().replace(".", " ").split()]
    return " ".join(words)


def extract(texts: list[str]) -> dict[str, list[str]]:
    """Candidate clue strings by kind, from OCR lines (one line per detected text box)."""
    # OCR often drops the space between a word and a number ("AVE5", "BUSSTOP83139", "BLK123").
    # Split letter->digit only, so block suffixes like "123A" stay intact.
    texts = [re.sub(r"(?<=[A-Z])(?=\d)", " ", t.upper()) for t in texts]
    joined = " | ".join(texts)
    found: dict[str, list[str]] = {k: [] for k in ("postal", "bare_postal", "bus_stop", "block", "street", "station")}

    def add(kind, value):
        if value and value not in found[kind]:
            found[kind].append(value)

    for m in POSTAL_RE.finditer(joined):
        if m.group(1)[:2] in VALID_POSTAL_SECTORS:
            add("postal", m.group(1))
    for m in BARE_POSTAL_RE.finditer(joined):
        if m.group(1)[:2] in VALID_POSTAL_SECTORS and m.group(1) not in found["postal"]:
            add("bare_postal", m.group(1))
    for m in BUS_RE.finditer(joined):
        add("bus_stop", m.group(1))
    for m in BLOCK_RE.finditer(joined):
        add("block", m.group(1))
    for t in texts:  # streets and stations: per line, so names don't run across signs
        line = re.sub(r"[^A-Z0-9' ]", " ", t.upper())
        for m in STREET_RE.finditer(line):
            add("street", normalize_street((m.group(1) or m.group(2)).strip()))
        for m in STATION_RE.finditer(line):
            add("station", (m.group(1) or m.group(2)).strip())
    return found


# --------------------------------------------------------------------------- resolution


def _closest(cands: list[tuple[float, float, str]], prior: tuple[float, float] | None):
    if not cands:
        return None
    if prior is None:
        return cands[0]
    return min(cands, key=lambda c: haversine_km(c[0], c[1], *prior))


class ClueLocator:
    def __init__(
        self,
        reader: TextReader | None = None,
        onemap: OneMap | None = None,
        bus_stops: BusStops | None = None,
        roads=None,
        names=None,
    ):
        """`roads`: a Roads instance, None to download/load it on first use, or False to
        resolve street names with OneMap only. `names`: a PlaceNames gazetteer, None to load
        it from the cache if present, or False to skip named places."""
        self._reader = reader
        self.onemap = onemap or OneMap()
        self._bus = bus_stops
        self._roads = roads
        self._names = names

    @property
    def names(self):
        if self._names is None:
            from .placenames import PlaceNames

            path = CACHE_DIR / "place_names.json.gz"
            self._names = PlaceNames.load(path) if path.exists() else False
        return self._names or None

    @property
    def reader(self) -> TextReader:
        if self._reader is None:
            self._reader = TextReader()
        return self._reader

    @property
    def bus(self) -> BusStops | None:
        if self._bus is None:
            try:
                self._bus = BusStops.load()
            except Exception:
                self._bus = BusStops({})
        return self._bus

    @property
    def roads(self):
        if self._roads is None:
            from .roads import Roads

            try:
                self._roads = Roads.load()
            except Exception:
                self._roads = False  # offline: fall back to OneMap street lookups
        return self._roads or None

    def _places(self, texts: list[str], prior) -> list[Clue]:
        """Named schools, malls, CCs ... read on signs. OSM supplies the (fuzzy-matched)
        official name; OneMap, when it knows that name, the current location (OSM can lag
        behind a school's move)."""
        if not self.names:
            return []
        out = []
        for line in _place_lines(texts):
            matches = self.names.match(line)
            if not matches:
                continue
            from .placenames import squash

            cands = []
            for name, lat, lon, _ in matches:
                # OneMap knows current addresses; its name may be longer ("APSN TANGLIN SCHOOL").
                hits = _hits(self.onemap.search(name), lambda r, n=squash(name): n in squash(r.get("SEARCHVAL", "")),
                             lambda r: f"{r.get('SEARCHVAL', '')}, {r.get('ADDRESS', '')}")
                cands += hits or [(lat, lon, f"{name} (OpenStreetMap)")]
            cands = list({(round(c[0], 3), round(c[1], 3)): c for c in cands}.values())
            best = _closest(cands, prior)
            detail = best[2] if len(cands) == 1 else f"{best[2]} (closest to the visual guess of {len(cands)} places with this name)"
            out.append(Clue("place", line, best[0], best[1], CONFIDENCE["place"], detail))
        # The same place read in several boxes counts once.
        return list({(c.lat, c.lon): c for c in out}.values())

    def locate(self, path: str | Path, prior: tuple[float, float] | None = None) -> ClueResult | None:
        return self.locate_text(self.reader.read(path), prior)

    def locate_text(self, texts: list[str], prior: tuple[float, float] | None = None) -> ClueResult | None:
        """Resolve OCR lines to a location. `prior` (a rough visual guess) breaks ties
        between same-named places, e.g. the many 'Blk 123's."""
        found = extract(texts)
        clues: list[Clue] = []

        for code in found["postal"] + found["bare_postal"]:
            exact = _hits(self.onemap.search(code), lambda r: r.get("POSTAL") == code, lambda r: r.get("ADDRESS", ""))
            if exact:
                conf = CONFIDENCE["postal"] if code in found["postal"] else CONFIDENCE["postal"] - 0.15
                lat, lon, addr = exact[0]
                clues.append(Clue("postal", code, lat, lon, conf, addr))

        for code in found["bus_stop"]:
            hit = self.bus.lookup(code) if self.bus else None
            if hit:
                clues.append(Clue("bus_stop", code, hit[0], hit[1], CONFIDENCE["bus_stop"], hit[2]))

        streets = found["street"]
        roads = self.roads if streets else None
        if roads:
            # Correct OCR misreads against real road names, dropping text that isn't a road.
            streets = list(dict.fromkeys(m for m in (roads.match(s) for s in streets) if m))

        for blk in found["block"]:
            for street in streets:
                hits = _hits(self.onemap.search(f"{blk} {street}"),
                             lambda r: r.get("BLK_NO", "").upper() == blk, lambda r: r.get("ADDRESS", ""))
                best = _closest(hits, prior)
                if best:
                    clues.append(Clue("block", f"BLK {blk} {street}", best[0], best[1], CONFIDENCE["block"], best[2]))

        for name in found["station"]:
            results = self.onemap.search(f"{name} MRT STATION") + self.onemap.search(f"{name} LRT STATION")
            prefixes = (f"{name} MRT STATION", f"{name} LRT STATION")
            hits = _hits(results, lambda r: r.get("SEARCHVAL", "").upper().startswith(prefixes),
                         lambda r: r.get("SEARCHVAL", ""))
            best = _closest(hits, prior)
            if best:
                clues.append(Clue("station", name, best[0], best[1], CONFIDENCE["station"], best[2]))

        clues.extend(self._places(texts, prior))

        if roads:
            junctions = []
            for i, a in enumerate(streets):
                for b in streets[i + 1 :]:
                    best = _closest([(lat, lon, "") for lat, lon in roads.junctions(a, b)], prior)
                    if best:
                        junctions.append(Clue("junction", f"{a} x {b}", best[0], best[1], CONFIDENCE["junction"],
                                              f"junction of {a} and {b}"))
            clues.extend(junctions)
            # A lone road name only says "somewhere on this road": pick the point nearest a
            # junction we found, else nearest the visual guess.
            anchor = (junctions[0].lat, junctions[0].lon) if junctions else prior
            for street in streets:
                lat, lon = roads.point_on(street, anchor)
                clues.append(Clue("street", street, lat, lon, CONFIDENCE["street"], street))
        else:
            for street in streets:
                hits = _hits(self.onemap.search(street), lambda r: r.get("ROAD_NAME", "").upper() == street,
                             lambda r: r.get("ROAD_NAME", ""))
                best = _closest(hits, prior)
                if best:
                    clues.append(Clue("street", street, best[0], best[1], CONFIDENCE["street"], best[2]))

        self.onemap.save()
        if not clues:
            return None
        return combine(clues, texts)


def _place_lines(texts: list[str]) -> list[str]:
    """Each OCR line, plus neighbouring lines joined (a name and 'SCHOOL' can be read as
    separate boxes)."""
    lines = [t for t in texts if t.strip()]
    return list(dict.fromkeys(lines + [f"{a} {b}" for a, b in zip(lines, lines[1:])]))


def combine(clues: list[Clue], texts: list[str] | None = None, agree_km: float = 1.0) -> ClueResult:
    """Pick the location best supported by the clues.

    Each clue votes for its own location with its confidence plus the confidence of every
    other clue within `agree_km` of it, so a bus stop on the street whose name was also
    read beats a lone stray number."""
    def support(c: Clue) -> float:
        return c.confidence + sum(o.confidence for o in clues if o is not c and haversine_km(c.lat, c.lon, o.lat, o.lon) <= agree_km)

    best = max(clues, key=lambda c: (support(c), c.confidence))
    agreeing = [c for c in clues if haversine_km(best.lat, best.lon, c.lat, c.lon) <= agree_km]
    # Agreement between independent clues raises confidence: 1 - product of doubts.
    doubt = 1.0
    for c in agreeing:
        doubt *= 1 - c.confidence
    agreeing.sort(key=lambda c: -c.confidence)
    return ClueResult(best.lat, best.lon, round(1 - doubt, 3), agreeing, list(texts or []))
