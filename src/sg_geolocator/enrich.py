"""Fill gaps in the plan map from other building datasets.

OpenStreetMap has most outlines but few heights outside HDB estates: about 32,000 of the
57,000 block-sized buildings in the island export carry no height or levels, so condos,
offices and malls were drawn 12-15 m tall (or a blanket 60 m when tagged as flats). Other
open datasets carry what is missing:

- HDB Existing Building (data.gov.sg): every HDB block outline with its block number;
  HDB Property Information gives each block's storeys and completion year.
- building-height datasets estimated from satellite imagery (one value per outline).
- outlines OSM lacks (Overture's Google / Microsoft / Esri footprints).

Everything is matched to the plan's own outlines on a grid of building ids, so a dataset's
polygon lends its value to the OSM building it overlaps most.
"""

from __future__ import annotations

import numpy as np

from .plan2d import to_xy

RES = 2.0  # grid cell size in metres for matching outlines
FLOOR_M = 3.0
HDB_ROOF_M = 3.0  # lift motor rooms / water tanks above the top storey


class IdGrid:
    """Which plan building covers each cell of a grid (-1 = none)."""

    def __init__(self, items: list[dict], res: float = RES, bbox=None):
        from PIL import Image, ImageDraw

        xys = [to_xy(*np.asarray(it["pts"], float).T) for it in items]
        allxy = np.concatenate(xys) if xys else np.zeros((1, 2))
        lo, hi = (allxy.min(0) - 50, allxy.max(0) + 50) if bbox is None else bbox
        self.res, self.origin = res, np.asarray(lo, float)
        self.w, self.h = int((hi[0] - lo[0]) / res) + 1, int((hi[1] - lo[1]) / res) + 1
        img = Image.new("I", (self.w, self.h), -1)
        dr = ImageDraw.Draw(img)
        # big buildings first, so a small building inside a big one (a kiosk in a mall) keeps its own cells
        order = sorted(range(len(items)), key=lambda i: -_area(xys[i]))
        for i in order:
            if len(xys[i]) >= 3:
                dr.polygon([tuple(p) for p in (xys[i] - self.origin) / res], fill=i)
        self.ids = np.asarray(img)

    def sample(self, xy: np.ndarray) -> np.ndarray:
        """Building id under each point (-1 off the grid or outside every building)."""
        j = ((xy[:, 0] - self.origin[0]) / self.res).astype(int)
        i = ((xy[:, 1] - self.origin[1]) / self.res).astype(int)
        ok = (i >= 0) & (i < self.h) & (j >= 0) & (j < self.w)
        out = np.full(len(xy), -1)
        out[ok] = self.ids[i[ok], j[ok]]
        return out


def _area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2)


def interior_points(xy: np.ndarray, step: float = 4.0, max_points: int = 200) -> np.ndarray:
    """Points on a grid inside a polygon (its centroid when it is tiny)."""
    from .scene import inside

    lo, hi = xy.min(0), xy.max(0)
    n = max(1.0, np.sqrt(np.prod(hi - lo) / max_points))
    s = max(step, n)
    gx, gy = np.meshgrid(np.arange(lo[0] + s / 2, hi[0], s), np.arange(lo[1] + s / 2, hi[1], s))
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    if len(pts):
        pts = pts[inside(xy, pts)]
    return pts if len(pts) else xy.mean(0)[None]


def match(grid: IdGrid, xy: np.ndarray) -> tuple[int, float, float]:
    """(plan building the polygon overlaps most, share of the polygon on it, share of the
    polygon on any plan building)."""
    ids = grid.sample(interior_points(xy))
    hit = ids[ids >= 0]
    if not len(hit):
        return -1, 0.0, 0.0
    vals, counts = np.unique(hit, return_counts=True)
    k = int(np.argmax(counts))
    return int(vals[k]), counts[k] / len(ids), len(hit) / len(ids)


def has_height(item: dict) -> bool:
    for key in ("height", "building:levels"):
        try:
            if item.get(key) and float(str(item[key]).replace("m", "").strip()) > 0:
                return True
        except ValueError:
            pass
    return False


def apply_heights(plan: dict, polys: list[tuple[np.ndarray, float]], source: str, grid: IdGrid | None = None,
                  min_share: float = 0.3, log=print) -> int:
    """Give buildings without a height the estimate of the outside polygon covering most of
    them (polys: (lat/lon ring, height in m)). Stored as `height_est` / `height_src`, which
    default_height uses only when the map has no height or levels of its own."""
    items = plan["buildings"]
    grid = grid or IdGrid(items)
    votes: dict[int, list[tuple[float, float]]] = {}
    for ring, h in polys:
        if not h or h <= 0:
            continue
        xy = to_xy(*np.asarray(ring, float).T)
        i, share, _ = match(grid, xy)
        if i >= 0 and share >= min_share:
            votes.setdefault(i, []).append((float(h), _area(xy) * share))
    n = 0
    for i, v in votes.items():
        if has_height(items[i]):
            continue
        hs, ws = np.array(v).T
        order = np.argsort(hs)
        cum = np.cumsum(ws[order])
        items[i]["height_est"] = round(float(hs[order][np.searchsorted(cum, cum[-1] / 2)]), 1)  # weighted median
        items[i]["height_src"] = source
        n += 1
    log(f"  {source}: heights for {n} buildings without one")
    return n


def add_missing(plan: dict, polys: list[tuple[np.ndarray, dict]], source: str, grid: IdGrid | None = None,
                min_area: float = 300.0, max_covered: float = 0.2, log=print) -> int:
    """Add outside outlines (polys: (lat/lon ring, extra tags)) of `min_area`+ that the plan
    lacks (at most `max_covered` of them under existing buildings)."""
    items = plan["buildings"]
    grid = grid or IdGrid(items)
    new = []
    for ring, tags in polys:
        xy = to_xy(*np.asarray(ring, float).T)
        if len(xy) < 4 or _area(xy) < min_area:
            continue
        if match(grid, xy)[2] <= max_covered:
            new.append({"pts": [[round(a, 6), round(b, 6)] for a, b in ring], "building": "yes", "src": source, **tags})
    items += new
    log(f"  {source}: added {len(new)} outlines missing from the map")
    return len(new)


# --------------------------------------------------------------------------- HDB


def _norm_blk(s) -> str:
    return str(s or "").strip().upper().lstrip("0")


def apply_hdb(plan: dict, blocks: list[tuple[np.ndarray, dict]], property_rows: list[dict],
              grid: IdGrid | None = None, log=print) -> dict:
    """HDB block outlines (blocks: (lat/lon ring, {"blk": ..., "street_code": ..., "postal": ...}))
    matched to plan buildings: the block number goes onto the building (when it has none),
    the street comes from neighbours that do have it, and storeys / year from HDB Property
    Information give the height (storeys x 3 m + 3 m of roof structures) and `hdb_year`.
    Blocks missing from the map are added."""
    from .mapmatch import normalise_street

    items = plan["buildings"]
    grid = grid or IdGrid(items)
    props = {(_norm_blk(r.get("blk_no")), normalise_street(r.get("street", ""))): r for r in property_rows}
    matched = []
    for ring, info in blocks:
        xy = to_xy(*np.asarray(ring, float).T)
        i, share, covered = match(grid, xy)
        matched.append((ring, xy, info, i if share >= 0.3 else -1, covered))
    # street code -> street name, voted by buildings whose OSM address agrees on the block number
    votes: dict[str, dict[str, int]] = {}
    for _, _, info, i, _ in matched:
        if i >= 0 and info.get("street_code"):
            it = items[i]
            if _norm_blk(it.get("addr:housenumber")) == _norm_blk(info["blk"]) and it.get("addr:street"):
                st = normalise_street(it["addr:street"])
                v = votes.setdefault(info["street_code"], {})
                v[st] = v.get(st, 0) + 1
    street_of = {code: max(v, key=v.get) for code, v in votes.items()}
    stats = {"blocks": len(blocks), "matched": 0, "added": 0, "storeys": 0, "streets": len(street_of)}
    for ring, xy, info, i, covered in matched:
        blk = _norm_blk(info.get("blk"))
        street = street_of.get(info.get("street_code", ""))
        rec = props.get((blk, street)) if street else None  # a bare block number is shared by many towns
        if i < 0 and covered <= 0.2:
            items.append({"pts": [[round(a, 6), round(b, 6)] for a, b in ring], "building": "residential", "src": "hdb"})
            i = len(items) - 1
            stats["added"] += 1
        elif i < 0:
            continue
        else:
            stats["matched"] += 1
        it = items[i]
        it["hdb_blk"] = blk
        if not it.get("addr:housenumber") and blk:
            it["addr:housenumber"] = blk
        if not it.get("addr:street") and rec:
            it["addr:street"] = rec.get("street", "")
        if rec:
            try:
                storeys = int(rec.get("max_floor_lvl") or 0)
            except ValueError:
                storeys = 0
            if storeys > 0:
                it["height_hdb"] = storeys * FLOOR_M + HDB_ROOF_M
                stats["storeys"] += 1
            if rec.get("year_completed"):
                it["hdb_year"] = str(rec["year_completed"])
    log("  HDB: " + ", ".join(f"{k} {v}" for k, v in stats.items()))
    return stats


# --------------------------------------------------------------------------- loaders


def _outer_rings(geom: dict) -> list:
    if geom.get("type") == "Polygon":
        return geom["coordinates"][:1]
    if geom.get("type") == "MultiPolygon":
        return [poly[0] for poly in geom["coordinates"]]
    return []


def load_hdb_blocks(path) -> list[tuple[np.ndarray, dict]]:
    """HDB Existing Building GeoJSON (data.gov.sg): (lat/lon ring, block info) per outline."""
    import gzip
    import json

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        g = json.load(f)
    out = []
    for feat in g.get("features", []):
        p = feat.get("properties") or {}
        info = {"blk": p.get("BLK_NO"), "street_code": p.get("ST_COD"), "postal": p.get("POSTAL_COD")}
        for ring in _outer_rings(feat.get("geometry") or {}):
            out.append((np.array([[c[1], c[0]] for c in ring], float), info))
    return out


def load_hdb_property(path) -> list[dict]:
    """HDB Property Information CSV (data.gov.sg): blk_no, street, max_floor_lvl, year_completed, ..."""
    import csv
    import gzip
    import io

    raw = gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, newline="")
    with raw as f:
        return list(csv.DictReader(io.StringIO(f.read())))


def load_height_polys(path) -> list[tuple[np.ndarray, float]]:
    """Rows of [lat/lon ring, height m, source] (e.g. from the probe's GlobalBuildingAtlas extract)."""
    import gzip
    import json

    with gzip.open(path, "rt") as f:
        rows = json.load(f)
    return [(np.asarray(r[0], float), float(r[1])) for r in rows if r[1] is not None and float(r[1]) > 0]


def buildings_from_plan(plan: dict, log=print):
    """A Buildings file (footprint points + roof height + address label) from an enriched
    plan: every building whose height is known or estimated (HDB storeys, OSM tags, satellite
    estimate), not the ones that would only get a guess by type and size."""
    from .raster import default_height
    from .skyline import Buildings, _densify

    pts, hs, ids, labels = [], [], [], []
    for it in plan["buildings"]:
        if not (it.get("height_hdb") or has_height(it) or it.get("height_est")):
            continue
        ring = np.asarray(it["pts"], float)
        if len(ring) < 3:
            continue
        h = default_height(it, _area(to_xy(*ring.T)))
        dense = _densify(ring, 8.0)
        pts.append(dense)
        hs.append(np.full(len(dense), h, dtype=np.float32))
        ids.append(np.full(len(dense), len(labels), dtype=np.int32))
        hn, street = str(it.get("addr:housenumber", "")).strip(), str(it.get("addr:street", "")).strip()
        labels.append(f"Blk {hn} {street}" if hn and street else "")
    log(f"  buildings with a height: {len(labels)} ({sum(1 for x in labels if x)} with an address)")
    return Buildings(np.concatenate(pts), np.concatenate(hs), np.concatenate(ids), labels)
