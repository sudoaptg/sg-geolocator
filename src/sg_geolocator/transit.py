"""Public transport geometry from OpenStreetMap: bus routes by service number, MRT/LRT
tracks (elevated or underground) and stations. A bus showing its service number, or an
elevated MRT viaduct in view, limits where a photo can have been taken.

Data © OpenStreetMap contributors, ODbL (read from the Geofabrik extract, see osmdata).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from .osmdata import _in_sg


def extract(pbf: str | Path, log=print) -> dict:
    """{"bus": {service: [[lat, lon], ...] per way}, "rail": [{"line", "elevated", "pts"}],
    "stations": [{"name", "lat", "lon", "network"}]}"""
    import osmium

    bus_ways: dict[int, set[str]] = {}  # way id -> service numbers using it
    rail_lines: dict[int, str] = {}  # way id -> line name
    stations = []

    class Relations(osmium.SimpleHandler):
        def relation(self, r):
            route = r.tags.get("route")
            if route == "bus" and r.tags.get("ref"):
                for m in r.members:
                    if m.type == "w":
                        bus_ways.setdefault(m.ref, set()).add(r.tags["ref"])
            elif route in ("subway", "light_rail", "train", "monorail"):
                for m in r.members:
                    if m.type == "w":
                        rail_lines.setdefault(m.ref, r.tags.get("name", r.tags.get("ref", "")))

    Relations().apply_file(str(pbf))
    log(f"  bus route ways: {len(bus_ways)}, rail route ways: {len(rail_lines)}")

    bus: dict[str, list] = {}
    rail = []

    class Ways(osmium.SimpleHandler):
        def node(self, n):
            t = n.tags
            if (t.get("railway") == "station" or t.get("public_transport") == "station") and t.get("name") \
                    and _in_sg(n.location.lat, n.location.lon):
                stations.append({"name": t.get("name"), "lat": round(n.location.lat, 6), "lon": round(n.location.lon, 6),
                                 "network": t.get("network", "")})

        def way(self, w):
            is_rail = w.tags.get("railway") in ("subway", "light_rail", "rail", "monorail")
            if w.id not in bus_ways and not is_rail:
                return
            try:
                pts = [[round(n.lat, 6), round(n.lon, 6)] for n in w.nodes]
            except osmium.InvalidLocationError:
                return
            if not pts or not _in_sg(*pts[0]):
                return
            for svc in bus_ways.get(w.id, ()):
                bus.setdefault(svc, []).append(pts)
            if is_rail:
                try:
                    layer = int(w.tags.get("layer", "0"))
                except ValueError:
                    layer = 0
                underground = w.tags.get("tunnel") == "yes" or layer < 0
                elevated = (w.tags.get("bridge") == "yes" or layer > 0) and not underground
                rail.append({"line": rail_lines.get(w.id, w.tags.get("name", "")), "elevated": elevated,
                             "underground": underground, "pts": pts})

    Ways().apply_file(str(pbf), locations=True)
    log(f"  bus services: {len(bus)}, rail ways: {len(rail)} ({sum(r['elevated'] for r in rail)} elevated), stations: {len(stations)}")
    return {"bus": bus, "rail": rail, "stations": stations}


def save(data: dict, path) -> None:
    with gzip.open(path, "wt") as f:
        json.dump(data, f, separators=(",", ":"))


def load(path) -> dict:
    with gzip.open(path, "rt") as f:
        return json.load(f)
