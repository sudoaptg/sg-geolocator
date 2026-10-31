"""Named places (schools, malls, community clubs, places of worship, ...) for sign reading.

OCR often garbles a name ("TANGUNSCHOOL" for "TANGLIN SCHOOL"), so text is matched
fuzzily against every named place in OpenStreetMap: spaces and punctuation are ignored,
the generic word (SCHOOL, MALL, ...) must appear, and the distinctive rest of the name
must also be close, so "GREEN SCHOOL" can't become "GREENRIDGE SCHOOL".
"""

from __future__ import annotations

import difflib
import gzip
import json
import re
from pathlib import Path

KEYWORDS = (
    "PRIMARYSCHOOL", "SECONDARYSCHOOL", "SCHOOL", "COLLEGE", "POLYCLINIC", "CLINIC", "HOSPITAL", "CENTRE",
    "CENTER", "MALL", "PLAZA", "CLUB", "CHURCH", "TEMPLE", "MOSQUE", "MASJID", "MARKET", "LIBRARY",
    "STADIUM", "HAWKER", "KINDERGARTEN", "PRESCHOOL", "MUSEUM", "HOTEL", "INSTITUTION",
)
MIN_LEN = 8  # squashed OCR text shorter than this is too ambiguous
FULL_CUTOFF = 0.85  # whole name similarity
CORE_CUTOFF = 0.7  # similarity of the distinctive part (the name minus the keyword)


# Abbreviations used on signs, expanded before matching.
ABBREVIATIONS = {"PRI": "PRIMARY", "SEC": "SECONDARY", "SCH": "SCHOOL", "CTR": "CENTRE", "CC": "COMMUNITY CLUB",
                 "JC": "JUNIOR COLLEGE", "POLY": "POLYTECHNIC", "MKT": "MARKET", "HOSP": "HOSPITAL"}


def squash(text: str) -> str:
    words = re.sub(r"[^A-Z0-9 ]", " ", text.upper()).split()
    s = "".join(ABBREVIATIONS.get(w, w) for w in words).replace(" ", "")
    return re.sub(r"(?<=[A-Z])0|0(?=[A-Z])", "O", s)  # OCR reads O as 0 inside words


def _keyword(s: str) -> str | None:
    return next((k for k in KEYWORDS if k in s), None)


class PlaceNames:
    def __init__(self, entries: list[tuple[str, float, float]]):
        self.entries = entries
        self._by_kw: dict[str, list[tuple[str, int]]] = {}
        for i, (name, _, _) in enumerate(entries):
            sq = squash(name)
            kw = _keyword(sq)
            if kw:
                # Also index the name without leading words ("APSN TANGLIN SCHOOL" is signed
                # "TANGLIN SCHOOL"), so a sign showing only the end of a name still matches.
                words = name.upper().split()
                for j in range(len(words)):
                    self._by_kw.setdefault(kw, []).append((squash(" ".join(words[j:])), i))

    def match(self, text: str) -> list[tuple[str, float, float, float]]:
        """(official name, lat, lon, similarity) of the places the text most likely names."""
        sq = squash(text)
        kw = _keyword(sq)
        if not kw or len(sq) < MIN_LEN or len(sq) - len(kw) < 3:
            return []
        core = sq.replace(kw, "", 1)
        best: dict[int, float] = {}
        for cand, i in self._by_kw.get(kw, []):
            if abs(len(cand) - len(sq)) > 4:
                continue
            full = difflib.SequenceMatcher(None, sq, cand).ratio()
            if full < FULL_CUTOFF:
                continue
            if difflib.SequenceMatcher(None, core, cand.replace(kw, "", 1)).ratio() < CORE_CUTOFF:
                continue
            best[i] = max(best.get(i, 0.0), full)
        if not best:
            return []
        top = max(best.values())
        return [(*self.entries[i], s) for i, s in best.items() if s >= top - 1e-9]

    @classmethod
    def from_osm(cls, osm: dict) -> "PlaceNames":
        seen, entries = set(), []
        for name, lat, lon in osm["names"]:
            key = (name, round(lat, 3), round(lon, 3))
            if key not in seen:
                seen.add(key)
                entries.append((name, float(lat), float(lon)))
        return cls(entries)

    def save(self, path) -> None:
        with gzip.open(path, "wt") as f:
            json.dump(self.entries, f)

    @classmethod
    def load(cls, path) -> "PlaceNames":
        with gzip.open(Path(path), "rt") as f:
            return cls([tuple(e) for e in json.load(f)])
