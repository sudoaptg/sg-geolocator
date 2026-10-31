"""Public transport and road geometry from OpenStreetMap: bus routes by service number,
MRT/LRT tracks (elevated or underground), stations, and the road network with flyovers
marked. A bus showing its service number, an elevated MRT viaduct or a road flyover in view
limits where a photo can have been taken.

Data © OpenStreetMap contributors, ODbL (read from the Geofabrik extract, see osmdata).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from .osmdata import _in_sg


ROAD_KINDS = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
              "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"}


def _layer(tags) -> int:
    try:
        return int(tags.get("layer", "0"))
    except ValueError:
        return 0


def extract(pbf: str | Path, log=print) -> dict:
    """{"bus": {service: [[lat, lon], ...] per way}, "rail": [{"line", "elevated", "pts"}],
    "stations": [{"name", "lat", "lon", "network"}],
    "roads": [{"kind", "name", "lanes", "elevated", "underground", "pts"}]}"""
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
    roads = []

    class Ways(osmium.SimpleHandler):
        def node(self, n):
            t = n.tags
            if (t.get("railway") == "station" or t.get("public_transport") == "station") and t.get("name") \
                    and _in_sg(n.location.lat, n.location.lon):
                stations.append({"name": t.get("name"), "lat": round(n.location.lat, 6), "lon": round(n.location.lon, 6),
                                 "network": t.get("network", "")})

        def way(self, w):
            is_rail = w.tags.get("railway") in ("subway", "light_rail", "rail", "monorail")
            is_road = w.tags.get("highway") in ROAD_KINDS
            if w.id not in bus_ways and not is_rail and not is_road:
                return
            try:
                pts = [[round(n.lat, 6), round(n.lon, 6)] for n in w.nodes]
            except osmium.InvalidLocationError:
                return
            if not pts or not _in_sg(*pts[0]):
                return
            for svc in bus_ways.get(w.id, ()):
                bus.setdefault(svc, []).append(pts)
            layer = _layer(w.tags)
            underground = w.tags.get("tunnel") == "yes" or layer < 0
            elevated = (w.tags.get("bridge") == "yes" or layer > 0) and not underground
            if is_road:
                roads.append({"kind": w.tags["highway"], "name": w.tags.get("name", ""), "lanes": w.tags.get("lanes", ""),
                              "elevated": elevated, "underground": underground, "pts": pts})
            if is_rail:
                rail.append({"line": rail_lines.get(w.id, w.tags.get("name", "")), "elevated": elevated,
                             "underground": underground, "pts": pts})

    Ways().apply_file(str(pbf), locations=True)
    log(f"  bus services: {len(bus)}, rail ways: {len(rail)} ({sum(r['elevated'] for r in rail)} elevated), "
        f"stations: {len(stations)}, roads: {len(roads)} ({sum(r['elevated'] for r in roads)} flyover ways)")
    return {"bus": bus, "rail": rail, "stations": stations, "roads": roads}


def save(data: dict, path) -> None:
    with gzip.open(path, "wt") as f:
        json.dump(data, f, separators=(",", ":"))


def load(path) -> dict:
    with gzip.open(path, "rt") as f:
        return json.load(f)
