"""Command-line interface for sg-geolocate."""

from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sg-geolocate")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="Estimate the location of one or more photos")
    p.add_argument("images", nargs="+")
    p.add_argument("--index", help="Retrieval index (.npz) from build-index")
    p.add_argument("--no-exif", action="store_true", help="Ignore EXIF GPS (useful for evaluation)")
    p.add_argument("--no-text", action="store_true", help="Skip reading signs/text in the photo")
    p.add_argument("--geocell", help="Area classifier (.npz) from train-geocell")
    p.add_argument("--places", help="Dense place-recognition index (.npz) for exact-spot matching")
    p.add_argument("--map", help="Map matcher (.npz) from train-map: compares the scene with map features")
    p.add_argument("--refine", action="store_true",
                   help="After the guess, run `resect --auto` around it (needs --buildings; --plan / --transit add the "
                        "render and rail checks) and report the refined spot as well")
    p.add_argument("--plan", help="Plan-view map (sgplan release), for --refine")
    p.add_argument("--transit", help="transit.json.gz, for --refine")
    p.add_argument("--buildings", help="Building heights (.npz) from build-buildings: skyline matching (with --map)")

    f = sub.add_parser("fetch-mapillary", help="Download reference photos from Mapillary")
    f.add_argument("--out", default="data/mapillary")
    f.add_argument("--boundary", help="Singapore boundary GeoJSON, to drop Johor/Riau photos")
    f.add_argument("--subcell", type=float, default=0.002, help="Thinning grid size in degrees (~220 m)")
    f.add_argument("--per-subcell", type=int, default=2)
    f.add_argument("--max-images", type=int, default=None)

    b = sub.add_parser("build-index", help="Build a retrieval index from geotagged photos")
    src = b.add_mutually_exclusive_group(required=True)
    src.add_argument("folder", nargs="?", help="Folder of photos with GPS in EXIF")
    src.add_argument("--metadata", help="metadata.csv from fetch-mapillary")
    b.add_argument("--out", default="index.npz")
    b.add_argument("--model", default=None, help="Hugging Face CLIP model id (default: CLIP ViT-B/32; e.g. geolocal/StreetCLIP)")
    b.add_argument("--shard", help="i/n: embed only the i-th of n slices (with --metadata), for parallel jobs")
    b.add_argument("--limit", type=int, help="Only the first N photos of the metadata (for quick tests)")

    mg = sub.add_parser("merge-index", help="Join shard indexes from build-index --shard")
    mg.add_argument("parts", nargs="+")
    mg.add_argument("--out", default="index.npz")

    g = sub.add_parser("train-geocell", help="Train the area classifier on all photos of an index")
    g.add_argument("index")
    g.add_argument("--cell", type=float, default=0.01, help="Cell size in degrees (0.01 is about 1.1 km)")
    g.add_argument("--out", default="geocell.npz")

    e = sub.add_parser("evaluate", help="Score retrieval on held-out photos of an index")
    e.add_argument("index")
    e.add_argument("--test-fraction", type=float, default=0.1)
    e.add_argument("--json-out")
    e.add_argument("--md-out")
    e.add_argument("--radius", type=float, default=1.0, help="Neighbour agreement radius in km (0.1 for dense indexes)")
    e.add_argument("--max-test", type=int, help="Score at most N held-out photos")
    e.add_argument("--no-geocell", action="store_true", help="Skip the area classifier (slow on large indexes)")

    d = sub.add_parser("build-dense-shard", help="Search, download and embed one shard of a dense place index")
    d.add_argument("--shard", required=True, help="i/n")
    d.add_argument("--model", default=None, help="Default: EigenPlaces ResNet50-512 place recognition")
    d.add_argument("--boundary", help="Singapore boundary GeoJSON")
    d.add_argument("--subcell", type=float, default=None, help="Thinning grid in degrees (default ~17 m)")
    d.add_argument("--per-subcell", type=int, default=None)
    d.add_argument("--limit", type=int, help="Only the first N photos of the shard (timing tests)")
    d.add_argument("--out", required=True)

    t = sub.add_parser("evaluate-text", help="Score the text-clue stage on held-out Mapillary photos")
    t.add_argument("metadata", help="metadata.csv from fetch-mapillary")
    t.add_argument("--test-fraction", type=float, default=0.1)
    t.add_argument("--limit", type=int)
    t.add_argument("--hires", action="store_true", help="Fetch 2048 px photos (needs MAPILLARY_TOKEN)")
    t.add_argument("--json-out")
    t.add_argument("--md-out")

    cs = sub.add_parser("commons-search", help="List geotagged Wikimedia Commons photos in Singapore")
    cs.add_argument("--boundary", help="Singapore boundary GeoJSON")
    cs.add_argument("--out", default="commons.json")

    cb = sub.add_parser("build-commons-shard", help="Embed one shard of the Commons photos (place recognition)")
    cb.add_argument("--records", default="commons.json")
    cb.add_argument("--shard", required=True, help="i/n")
    cb.add_argument("--model", default=None, help="Default: EigenPlaces ResNet50-512")
    cb.add_argument("--workers", type=int, default=4, help="Parallel downloads (keep low: Wikimedia rate-limits)")
    cb.add_argument("--out", required=True)

    bm = sub.add_parser("build-map", help="Grid of nearby map features (OpenStreetMap + HDB block ages)")
    bm.add_argument("--out", default="mapgrid.npz")

    ag = sub.add_parser("area-guess", help="Area classifier probabilities for each ~1 km cell (JSON)")
    ag.add_argument("images", nargs="+")
    ag.add_argument("--geocell", required=True)
    ag.add_argument("--min-prob", type=float, default=0.001, help="Leave out cells below this probability")

    bn = sub.add_parser("build-names", help="Gazetteer of named places (schools, malls, CCs...) for sign reading")
    bn.add_argument("--out", default="place_names.json.gz")

    bm = sub.add_parser("benchmark", help="Run resect on photos with a known answer and report how far off it is")
    bm.add_argument("cases", help="benchmarks/cases.json")
    bm.add_argument("--images", default="benchmarks/photos", help="Folder with the case photos")
    bm.add_argument("--buildings", required=True)
    bm.add_argument("--names", help="place_names.json.gz")
    bm.add_argument("--water", help="Water layer for cases using --render / --on-water")
    bm.add_argument("--plan", help="Plan-view map (sgplan release) for cases using --render")
    bm.add_argument("--transit", help="transit.json.gz for cases using rail filters")
    bm.add_argument("--only", action="append", default=[], help="Run only these case names")
    bm.add_argument("--out", help="Write the results as JSON here")

    ea = sub.add_parser("export-area", help="Plan-view map data for one box: all building outlines, water, green, roads")
    ea.add_argument("--bbox", required=True, help="south,west,north,east")
    ea.add_argument("--out", default="area.json.gz")
    ea.add_argument("--layers", default="buildings,water,green,roads", help="Which of buildings,water,green,roads")

    rs = sub.add_parser("resect", help="Where and which way a photo was taken: plan-view outline match first, height second")
    rs.add_argument("image")
    rs.add_argument("--buildings", required=True)
    rs.add_argument("--near", help="lat,lon or a place name to search around (default: the first landmark)")
    rs.add_argument("--radius", type=float, default=250.0, help="Search this many metres around --near")
    rs.add_argument("--landmark", action="append", default=[],
                    help="NAME[:left|right|centre]=X: something recognised in the photo and its column X (pixels, or 0-1 "
                         "across). NAME is lat,lon, a building address ('60 Robinson Road') or a place name. Repeatable.")
    rs.add_argument("--hfov", type=float, help="Horizontal field of view in degrees (default: EXIF, else 45-95 searched)")
    rs.add_argument("--step", type=float, default=4.0, help="Spot spacing in metres")
    rs.add_argument("--min-height", type=float, default=20.0, help="Ignore buildings lower than this (metres) in the outline")
    rs.add_argument("--block", help="NUMBER=X: a block number seen in the photo at column X; every block with that "
                                     "number in Singapore is tried (no --near needed)")
    rs.add_argument("--render", action="store_true",
                    help="Re-rank by render-and-compare: label the photo's pixels (sky, building, water, ...) and compare "
                         "with the map drawn from each candidate (needs the segmenter release in $SG_GEOLOCATOR_MODELS)")
    rs.add_argument("--water", help="Water layer (export-area file, e.g. the sgwater release) for --render / --on-water")
    rs.add_argument("--plan", help="Plan-view map (export-area file with buildings, water, green, e.g. the sgplan "
                                    "release): all building outlines and greenery for --render")
    rs.add_argument("--auto", action="store_true",
                    help="Work out the hints from the photo: a block number read off a building becomes --block, elevated "
                         "rail in view becomes a rail filter, mostly water in view becomes --anywhere-water (with --plan)")
    rs.add_argument("--anywhere-water", action="store_true",
                    help="With --render --plan, no area needed: photo taken at a lake/pond/river with blocks around; "
                         "every such water in Singapore is searched")
    rs.add_argument("--on-water", action="store_true", help="Photo taken at the water's edge: only try spots on mapped water")
    rs.add_argument("--elevated-tracks", type=int, metavar="N",
                    help="With --block: the photo shows N elevated rail tracks side by side; keep only blocks with at "
                         "least that many within 300 m (needs --transit)")
    rs.add_argument("--elevated-lines", type=int, metavar="N",
                    help="With --block: N different elevated rail lines are in view (e.g. 2 where two lines meet)")
    rs.add_argument("--transit", help="transit.json.gz (for --elevated-tracks / --elevated-lines)")
    rs.add_argument("--window", type=float, metavar="MIN_HEIGHT",
                    help="View from a high window: only try spots on buildings at least this tall (metres)")
    rs.add_argument("--names", help="place_names.json.gz (to resolve place names)")
    rs.add_argument("--top", type=int, default=5)

    sa = sub.add_parser("search-area", help="Find a photo's spot inside an area narrowed down by hints (bus, MRT line, area)")
    sa.add_argument("image")
    sa.add_argument("--buildings", required=True)
    sa.add_argument("--transit", help="transit.json.gz from the map release (for --bus, --line, --near-elevated)")
    sa.add_argument("--names", help="place_names.json.gz (to resolve --area names)")
    sa.add_argument("--area", action="append", default=[], help="Place or station name, or lat,lon (repeatable)")
    sa.add_argument("--radius", type=float, default=1500.0, help="Metres around each --area")
    sa.add_argument("--bus", help="Bus service number seen in the photo")
    sa.add_argument("--line", help="Part of an MRT/LRT line name in view or nearby, e.g. Thomson")
    sa.add_argument("--line-radius", type=float, default=300.0)
    sa.add_argument("--near-elevated", type=float, help="Metres: an elevated viaduct is in view, keep spots this close to one")
    sa.add_argument("--view", choices=["street", "bus", "window"], default="street")
    sa.add_argument("--step", type=float, default=30.0, help="Spacing of candidate spots, metres")

    cs = sub.add_parser("chain-sites", help="Every OneMap result for a name (e.g. a preschool chain), with HDB block age")
    cs.add_argument("query")
    cs.add_argument("--out", required=True)

    bt = sub.add_parser("build-transit", help="Bus routes, MRT/LRT tracks and stations from the OSM extract")
    bt.add_argument("--out", default="transit.json.gz")

    bb = sub.add_parser("build-buildings", help="Building footprints with heights, for skyline matching")
    bb.add_argument("--out", default="buildings.npz")

    sc = sub.add_parser("skyline-check", help="Diagnostic: skyline error near a known spot vs random spots")
    sc.add_argument("image")
    sc.add_argument("--at", required=True, help="lat,lon of the known location")
    sc.add_argument("--buildings", default="buildings.npz")
    sc.add_argument("--radius", type=float, default=300.0, help="Search this many metres around --at")

    tm = sub.add_parser("train-map", help="Train and evaluate the map matcher on an index (StreetCLIP)")
    tm.add_argument("index")
    tm.add_argument("--grid", default="mapgrid.npz")
    tm.add_argument("--out", default="mapmatch.npz")
    tm.add_argument("--json-out")
    tm.add_argument("--md-out")

    args = parser.parse_args(argv)

    if args.cmd == "predict":
        from .pipeline import Geolocator

        geo = Geolocator(index_path=args.index, use_exif=not args.no_exif, use_text=not args.no_text,
                         geocell_path=args.geocell, places_path=args.places, map_path=args.map,
                         buildings_path=args.buildings)
        for img in args.images:
            pred = geo.predict(img)
            out = {"image": img, **(pred.as_dict() if pred else {"error": "no prediction"})}
            if args.refine and pred and args.buildings:
                out["refined"] = _refine_with_resect(img, pred, args)
            print(json.dumps(out))
        return 0

    if args.cmd == "area-guess":
        from .embedder import make_embedder
        from .geocell import GeoCellClassifier

        clf = GeoCellClassifier.load(args.geocell)
        emb = make_embedder(clf.model).embed_images(args.images)
        for img, p in zip(args.images, clf.predict_proba(emb)):
            keep = [i for i in p.argsort()[::-1] if p[i] >= args.min_prob]
            cells = [{"lat": round(float(clf.centroids[i][0]), 5), "lon": round(float(clf.centroids[i][1]), 5),
                      "p": round(float(p[i]), 5)} for i in keep]
            print(json.dumps({"image": img, "cell_deg": clf.cell_deg, "cells": cells}))
        return 0

    if args.cmd == "fetch-mapillary":
        from .geo import Boundary
        from .mapillary import collect

        token = os.environ.get("MAPILLARY_TOKEN")
        if not token:
            parser.error("set the MAPILLARY_TOKEN environment variable")
        boundary = Boundary.from_geojson(args.boundary) if args.boundary else None
        collect(token, args.out, boundary, args.subcell, args.per_subcell, args.max_images)
        return 0

    if args.cmd == "build-index":
        from .embedder import DEFAULT_MODEL, make_embedder
        from .retrieval import build_index, build_index_from_csv

        embedder = make_embedder(args.model or DEFAULT_MODEL)
        if args.metadata:
            shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None
            n = build_index_from_csv(args.metadata, embedder, args.out, shard=shard, limit=args.limit)
        else:
            n = build_index(args.folder, embedder, args.out)
        print(f"Indexed {n} photos with {embedder.model_name} -> {args.out}")
        return 0

    if args.cmd == "merge-index":
        from .retrieval import merge_indexes

        print(f"Merged {merge_indexes(args.parts, args.out)} photos -> {args.out}")
        return 0

    if args.cmd == "train-geocell":
        import numpy as np

        from .geocell import GeoCellClassifier

        data = np.load(args.index)
        model = str(data["model"]) if "model" in data else ""
        clf = GeoCellClassifier(cell_deg=args.cell).fit(data["embeddings"].astype(np.float32), data["coords"], log=print)
        clf.save(args.out, model=model)
        print(f"Saved area classifier ({len(clf.centroids)} cells) -> {args.out}")
        return 0

    if args.cmd == "build-dense-shard":
        from .dense import DENSE_PER_SUBCELL, DENSE_SUBCELL_DEG, build_dense_shard
        from .embedder import EIGENPLACES, make_embedder
        from .geo import Boundary

        token = os.environ.get("MAPILLARY_TOKEN")
        if not token:
            parser.error("set the MAPILLARY_TOKEN environment variable")
        i, n = (int(x) for x in args.shard.split("/"))
        n_done = build_dense_shard(
            token, make_embedder(args.model or EIGENPLACES), args.out, i, n,
            boundary=Boundary.from_geojson(args.boundary) if args.boundary else None,
            subcell_deg=args.subcell or DENSE_SUBCELL_DEG, per_subcell=args.per_subcell or DENSE_PER_SUBCELL,
            limit=args.limit,
        )
        print(f"Shard {i}/{n}: {n_done} photos -> {args.out}")
        return 0

    if args.cmd == "commons-search":
        from . import commons
        from .geo import Boundary

        s = commons.session()
        found = commons.search(Boundary.from_geojson(args.boundary) if args.boundary else None, s=s)
        recs = commons.records(found, s=s)
        with open(args.out, "w") as fh:
            json.dump(recs, fh)
        print(f"{len(recs)} Commons photos -> {args.out}")
        return 0

    if args.cmd == "build-commons-shard":
        from . import commons
        from .dense import embed_records
        from .embedder import EIGENPLACES, make_embedder

        i, n = (int(x) for x in args.shard.split("/"))
        with open(args.records) as fh:
            recs = json.load(fh)[i::n]
        print(f"Shard {i}/{n}: {len(recs)} Commons photos")
        n_done = embed_records(recs, make_embedder(args.model or EIGENPLACES), args.out, offset=1000 + i,
                               batch_size=32, workers=args.workers, session=commons.session(args.workers))
        print(f"Shard {i}/{n}: {n_done} photos -> {args.out}")
        return 0

    if args.cmd == "build-map":
        from .mapmatch import MapGrid
        from .roads import Roads

        grid = MapGrid.build(Roads.load(log=print), log=print)
        grid.save(args.out)
        print(f"Map grid {grid.shape} with {len(grid.names)} features -> {args.out}")
        return 0

    if args.cmd == "build-names":
        from . import osmdata
        from .placenames import PlaceNames

        names = PlaceNames.from_osm(osmdata.load(log=print))
        names.save(args.out)
        print(f"{len(names.entries)} named places -> {args.out} (copy to ~/.cache/sg_geolocator/ to use)")
        return 0

    if args.cmd == "benchmark":
        import subprocess
        from pathlib import Path

        import numpy as np

        spec = json.loads(Path(args.cases).read_text())
        rows = []
        for case in spec["cases"]:
            if args.only and case["name"] not in args.only:
                continue
            img = Path(args.images) / case["image"]
            if not img.exists():
                print(f"{case['name']}: missing {img}", file=sys.stderr)
                continue
            cmd = [sys.executable, "-m", "sg_geolocator.cli", "resect", str(img), "--buildings", args.buildings,
                   *[args.transit if a == "TRANSIT" else a for a in case["args"]]]
            if args.names:
                cmd += ["--names", args.names]
            if args.water and ("--render" in case["args"] or "--on-water" in case["args"]):
                cmd += ["--water", args.water]
            if args.plan and "--render" in case["args"]:
                cmd += ["--plan", args.plan]
            import time

            t = time.time()
            res = subprocess.run(cmd, capture_output=True, text=True)
            lines = [l for l in res.stdout.splitlines() if l.startswith("{")]
            spots = json.loads(lines[-1])["spots"] if lines else []
            truth = case["truth"]
            def off(sp):
                return float(np.hypot((sp["lat"] - truth["lat"]) * 111000,
                                      (sp["lon"] - truth["lon"]) * 111000 * np.cos(np.radians(1.35))))
            dists = [off(sp) for sp in spots]
            rank = next((i + 1 for i, d_ in enumerate(dists) if d_ <= case["tolerance_m"]), None)
            row = {"name": case["name"], "error_m": round(dists[0], 1) if dists else None,
                   "solved": bool(dists) and dists[0] <= case["tolerance_m"], "rank_of_first_correct": rank,
                   "seconds": round(time.time() - t)}
            if spots and "heading" in truth and "heading" in spots[0]:
                row["heading_error"] = round(abs((spots[0]["heading"] - truth["heading"] + 180) % 360 - 180), 1)
            if res.returncode:
                row["error"] = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else f"exit {res.returncode}"
            rows.append(row)
            print(json.dumps(row), flush=True)
        n = sum(r["solved"] for r in rows)
        print(f"solved {n}/{len(rows)}", file=sys.stderr)
        if args.out:
            Path(args.out).write_text(json.dumps({"results": rows, "solved": n, "total": len(rows)}, indent=1))
        return 0

    if args.cmd == "export-area":
        import gzip

        from . import osmdata
        import requests

        pbf = osmdata.CACHE_PATH.parent / "region.osm.pbf"
        if not pbf.exists():
            pbf.parent.mkdir(parents=True, exist_ok=True)
            with requests.get(osmdata.EXTRACT_URL, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(pbf, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
        bbox = tuple(float(x) for x in args.bbox.split(","))
        data = osmdata.export_area(pbf, bbox, layers=tuple(args.layers.split(",")))
        with gzip.open(args.out, "wt") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"area -> {args.out}")
        return 0

    if args.cmd == "resect":
        from PIL import Image

        from . import plan2d
        from .skyline import Buildings, _exif_hfov

        b = Buildings.load(args.buildings)
        names = None
        if args.names:
            from .placenames import PlaceNames

            names = PlaceNames.load(args.names)
        width = Image.open(args.image).width
        if args.auto:
            _auto_hints(args, b)
        landmarks = [_parse_landmark(t, b, names, width) for t in args.landmark]
        if args.block and (args.elevated_tracks or args.elevated_lines):  # fast filter before anything else
            from . import transit as transit_mod

            tr = transit_mod.load(args.transit)
            num = args.block.rsplit("=", 1)[0]
            keep = []
            for bid in plan2d.block_candidates(b, num, min_height=0):
                c0 = b.footprint(bid)[0].mean(0)
                n = plan2d.elevated_tracks_near(tr, float(c0[0]), float(c0[1]))
                nl = plan2d.elevated_lines_near(tr, float(c0[0]), float(c0[1]))
                if n >= (args.elevated_tracks or 0) and nl >= (args.elevated_lines or 0):
                    keep.append({"block": b.labels[bid], "lat": round(float(c0[0]), 6), "lon": round(float(c0[1]), 6),
                                 "elevated_tracks": n, "elevated_lines": nl,
                                 "map": f"https://www.google.com/maps?q={c0[0]:.6f},{c0[1]:.6f}"})
            print(f"  {len(keep)} of {len(plan2d.block_candidates(b, num, min_height=0))} blocks {num} pass the rail filter",
                  file=sys.stderr)
            keep.sort(key=lambda k: (-k["elevated_lines"], -k["elevated_tracks"]))  # most rail around first
            if not (args.render or args.on_water) or len(keep) <= 1:
                print(json.dumps({"image": args.image, "block": num, "spots": keep}))
                return 0
        if args.block and (args.render or args.on_water):
            ranked = _render_block(args, b, width)
            print(json.dumps({"image": args.image, "block": args.block.split("=")[0], "spots": ranked[: args.top]}))
            return 0
        if args.block:
            num, x = args.block.rsplit("=", 1)
            x = float(x) / width if float(x) > 1 else float(x)
            hfov = args.hfov or _exif_hfov(Image.open(args.image))
            ranked = plan2d.resect_block(b, plan2d.photo_edges(args.image), num, x, args.radius,
                                         [hfov] if hfov else list(range(45, 96, 5)), args.step, args.min_height,
                                         args.window, log=lambda m: print(m, file=sys.stderr))
            for s_ in ranked[: args.top]:
                s_["map"] = f"https://www.google.com/maps?q={s_['lat']},{s_['lon']}"
            print(json.dumps({"image": args.image, "block": num, "spots": ranked[: args.top]}))
            return 0
        if args.render and args.anywhere_water:
            from . import raster, scene

            if not args.plan:
                raise SystemExit("--anywhere-water needs --plan")
            labels = scene.Segmenter().labels(args.image, width=64)
            print(f"  photo: {raster.composition(labels)}", file=sys.stderr)
            spots = raster.search_waters(raster.Plan.load(args.plan), labels, b, top=args.top,
                                         log=lambda m: print(m, file=sys.stderr))
            print(json.dumps({"image": args.image, "spots": spots}))
            return 0
        if args.render and not landmarks and args.near:
            lat, lon = _resolve_area(args.near, None, names)
            spots = _render_area(args, b, lat, lon)
            print(json.dumps({"image": args.image, "spots": spots}))
            return 0
        if args.near:
            lat, lon = _resolve_area(args.near, None, names)
        elif landmarks:
            lat, lon = landmarks[0].lat, landmarks[0].lon
        else:
            raise SystemExit("give --near or at least one --landmark")
        hfov = args.hfov or _exif_hfov(Image.open(args.image))
        hfovs = [hfov] if hfov else list(range(45, 96, 5))
        edges = plan2d.photo_edges(args.image)
        print(f"{len(edges)} vertical edges in the photo; searching {args.radius:.0f} m around {lat:.5f},{lon:.5f}",
              file=sys.stderr)
        spots = plan2d.resect(b, edges, lat, lon, args.radius, landmarks, hfovs, args.step, min_height=args.min_height,
                              top=args.top, window_min_height=args.window, log=lambda m: print(m, file=sys.stderr))
        if args.render:  # stage 2: render-and-compare around each plan-view answer (height comes with it)
            spots = _render_rerank(args, b, spots)
        else:
            for s_ in spots:  # stage 2: height from the skyline, now that the spot and heading are fixed
                s_["height"] = plan2d.fit_height(b, args.image, s_["lat"], s_["lon"], s_["heading"], s_["hfov"])
        for s_ in spots:
            s_["map"] = f"https://www.google.com/maps?q={s_['lat']},{s_['lon']}"
        print(json.dumps({"image": args.image, "photo_edges": [round(float(x), 4) for x in edges], "spots": spots}))
        return 0

    if args.cmd == "search-area":
        import numpy as np

        from . import areasearch, transit as transit_mod
        from .skyline import Buildings

        tr = transit_mod.load(args.transit) if args.transit else None
        names = None
        if args.names:
            from .placenames import PlaceNames

            names = PlaceNames.load(args.names)
        areas = [_resolve_area(a, tr, names) for a in args.area] or [None]
        spots = []
        for a in areas:
            area = (a[0], a[1], args.radius) if a else None
            spots.append(areasearch.candidate_spots(tr, area, args.bus, args.line, args.line_radius,
                                                    args.near_elevated, args.step))
        spots = np.concatenate([s for s in spots if len(s)]) if any(len(s) for s in spots) else np.zeros((0, 2))
        print(f"{len(spots)} candidate spots", file=sys.stderr)
        result = areasearch.search(Buildings.load(args.buildings), args.image, spots, args.view)
        print(json.dumps({"image": args.image, **result}))
        return 0

    if args.cmd == "chain-sites":
        from .clues import OneMap, _coords
        from .mapmatch import fetch_hdb_records, normalise_street

        om = OneMap()
        results = om.search(args.query, pages=60)
        hdb = {}
        try:
            for r in fetch_hdb_records(print):
                hdb[(r["blk_no"].strip().upper(), normalise_street(r["street"]))] = r
        except Exception as exc:
            print(f"::warning::HDB data unavailable ({exc})")
        sites, seen = [], set()
        for r in results:
            c = _coords(r)
            if not c or (r.get("SEARCHVAL"), c) in seen:
                continue
            seen.add((r.get("SEARCHVAL"), c))
            blk, road = r.get("BLK_NO", "").strip().upper(), r.get("ROAD_NAME", "")
            h = hdb.get((blk, normalise_street(road)), {})
            sites.append({"name": r.get("SEARCHVAL"), "address": r.get("ADDRESS"), "blk": blk, "road": road,
                          "lat": c[0], "lon": c[1], "year": h.get("year_completed"), "storeys": h.get("max_floor_lvl")})
        with open(args.out, "w") as fh:
            json.dump(sites, fh, indent=1)
        print(f"{len(sites)} sites for {args.query!r} -> {args.out} ({sum(1 for x in sites if x['year'])} HDB blocks)")
        return 0

    if args.cmd == "build-transit":
        from . import transit
        from .osmdata import CACHE_PATH, EXTRACT_URL
        import requests

        pbf = CACHE_PATH.parent / "region.osm.pbf"
        if not pbf.exists():
            pbf.parent.mkdir(parents=True, exist_ok=True)
            with requests.get(EXTRACT_URL, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(pbf, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
        transit.save(transit.extract(pbf), args.out)
        print(f"transit -> {args.out}")
        return 0

    if args.cmd == "build-buildings":
        from .skyline import Buildings

        b = Buildings.build(log=print)
        b.save(args.out)
        print(f"{len(b.heights)} footprint points with heights -> {args.out}")
        return 0

    if args.cmd == "skyline-check":
        from .skyline import Buildings, check

        lat, lon = (float(x) for x in args.at.split(","))
        print(json.dumps({"image": args.image, **(check(Buildings.load(args.buildings), args.image, lat, lon, args.radius)
                                                   or {"error": "no usable skyline or no buildings"})}))
        return 0

    if args.cmd == "train-map":
        from .mapmatch import MapGrid, report_markdown, train_and_evaluate

        matcher, result = train_and_evaluate(args.index, MapGrid.load(args.grid), log=print)
        matcher.save(args.out)
        md = report_markdown(result)
        print(md)
        if args.json_out:
            with open(args.json_out, "w") as fh:
                json.dump(result, fh, indent=2)
        if args.md_out:
            with open(args.md_out, "w") as fh:
                fh.write(md)
        return 0

    if args.cmd == "evaluate-text":
        from .evaluate import evaluate_text, text_to_markdown

        token = os.environ.get("MAPILLARY_TOKEN") if args.hires else None
        if args.hires and not token:
            parser.error("--hires needs the MAPILLARY_TOKEN environment variable")
        result = evaluate_text(args.metadata, args.test_fraction, args.limit, token)
        md = text_to_markdown(result)
    else:
        from .evaluate import evaluate_index, to_markdown

        result = evaluate_index(args.index, args.test_fraction, geocell=not args.no_geocell, log=print,
                                radius_km=args.radius, max_test=args.max_test)
        md = to_markdown(result)
    print(md)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2)
    if args.md_out:
        with open(args.md_out, "w") as fh:
            fh.write(md)
    return 0


def _refine_with_resect(img, pred, args) -> list:
    """Run `resect --auto` around a predict guess (or wherever the photo's own hints point)."""
    import contextlib
    import io

    argv = ["resect", img, "--buildings", args.buildings, "--auto", "--near", f"{pred.lat},{pred.lon}",
            "--radius", "600", "--step", "25", "--top", "3"]  # ~1,800 spots: minutes, not half an hour
    if args.plan:
        argv += ["--plan", args.plan, "--render"]
    if args.transit:
        argv += ["--transit", args.transit]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            main(argv)
    except SystemExit as exc:
        return [{"error": str(exc)}]
    lines = [l for l in buf.getvalue().splitlines() if l.startswith("{")]
    return json.loads(lines[-1]).get("spots", []) if lines else []


def _auto_hints(args, b):
    """Fill in --block / rail filter / --anywhere-water from what the photo shows."""
    import re

    from . import plan2d, raster, scene
    from .clues import TextReader

    seg = scene.Segmenter()
    labels = seg.labels(args.image, width=64)
    shares = seg.class_shares(args.image)
    comp = raster.composition(labels)
    found = []
    if not args.block:
        texts = TextReader().read_positions(args.image)
        nums = [(t.strip().upper(), xy) for t, xy in texts.items() if re.fullmatch(r"\d{1,4}[A-Z]?", t.strip().upper())]
        nums = [(t, xy) for t, xy in nums if plan2d.block_candidates(b, t, min_height=0)]
        if nums:
            t, (x, _) = max(nums, key=lambda n: len(n[0]))
            args.block = f"{t}={x:.4f}"
            found.append(f"block number {t} at {x:.0%} across")
    if scene.rail_in_view(shares) and args.block and not args.elevated_tracks and args.transit:
        args.elevated_tracks = scene.tracks_in_view(shares)
        found.append(f"elevated rail in view ({scene.rail_share(shares):.0%} of the photo: {args.elevated_tracks}+ tracks)")
    if not args.block and not args.near and not args.landmark and comp["water"] >= 0.2 and args.plan:
        args.render = args.anywhere_water = True
        found.append(f"mostly water ({comp['water']:.0%})")
    if args.render is False and args.plan and not scene.edges_trustworthy(labels):
        args.render = True
    print(f"  auto: {', '.join(found) or 'no hints found'}", file=sys.stderr)


def _render_setup(args, b):
    """Photo labels, the plan map, and the search ranges for render-and-compare."""
    import numpy as np

    from . import raster, scene

    labels = scene.Segmenter().labels(args.image, width=64)
    if args.plan:
        plan = raster.Plan.load(args.plan)
    else:
        plan = raster.Plan.from_buildings(b, scene.load_water(args.water) if args.water else [])
    if args.window:  # map heights are often missing or wrong for the camera's own block: allow up to 60 m
        eyes = np.arange(max(6.0, args.window * 0.5), max(61.0, args.window * 1.6), 6.0)
    elif args.on_water:
        eyes = np.array([1.0, 2.0])
    else:
        eyes = np.array([1.5, 4.0, 10.0, 20.0, 30.0, 45.0, 60.0])  # street level up to a high floor
    horizons = np.arange(0.25, 0.71, 0.03)
    comp = raster.composition(labels)
    print(f"  photo: {', '.join(f'{k} {v:.0%}' for k, v in comp.items())}", file=sys.stderr)
    return raster, plan, labels, eyes, horizons, comp


def _fits_scene(plan, comp, lat, lon, radius):
    """Cheap filter from what the photo shows: lots of water needs mapped water nearby, lots of
    greenery needs mapped greenery (parks, mangrove, wood) nearby."""
    if comp["water"] >= 0.08 and not plan.has("water", lat, lon, radius):
        return "no water nearby"
    if comp["green"] >= 0.25 and plan._pts.get("green") and not plan.has("green", lat, lon, radius):
        return "no greenery nearby"
    return None


def _render_rerank(args, b, spots):
    """Stage 2 for plan-view answers: search spot (+-2 steps), heading, eye height and horizon
    around each by how well the drawn map agrees with the labelled photo."""
    import numpy as np

    from . import plan2d

    raster, plan, labels, eyes, horizons, comp = _render_setup(args, b)
    out = []
    for s_ in spots:
        pr = raster.PlanRaster(plan.near(s_["lat"], s_["lon"], 1500), s_["lat"], s_["lon"], 1500, b)
        c = plan2d.to_xy(s_["lat"], s_["lon"])
        st = args.step
        near = [c + (dx * st, dy * st) for dx in range(-2, 3) for dy in range(-2, 3)]
        hd0 = s_["heading"]
        ag, cam, hd, hf, eye, hz = raster.search(pr, labels, near, lambda cam, hf: list(np.arange(hd0 - 9, hd0 + 9.1, 3)),
                                                 [s_["hfov"]], eyes, horizons, keep=8)
        la, lo = plan2d.to_latlon(cam)
        out.append({**s_, "lat": round(la, 6), "lon": round(lo, 6), "heading": round(hd, 1), "agreement": round(ag, 4),
                    "eye_m": eye, "horizon": round(hz, 3)})
        print(f"  render: {la:.5f},{lo:.5f} agreement {ag:.3f}", file=sys.stderr)
    return sorted(out, key=lambda o: -o["agreement"])


def _render_area(args, b, lat, lon):
    """No landmarks: search the area by render-and-compare alone (spots every `step`, every
    6 degrees of heading), after the cheap scene filter."""
    import numpy as np

    from . import plan2d

    raster, plan, labels, eyes, horizons, comp = _render_setup(args, b)
    why = _fits_scene(plan, comp, lat, lon, args.radius + 300)
    if why:
        print(f"  area rejected: {why}", file=sys.stderr)
        return []
    pr = raster.PlanRaster(plan.near(lat, lon, args.radius + 1200), lat, lon, args.radius + 1200, b)
    c = plan2d.to_xy(lat, lon)
    st = max(args.step, 8.0)
    k = int(args.radius // st)
    spots = [c + (i * st, j * st) for i in range(-k, k + 1) for j in range(-k, k + 1) if np.hypot(i, j) * st <= args.radius]
    if args.on_water:
        spots = [p for p in spots if pr.kind[tuple(((p - pr.origin) / pr.res).astype(int))] == 2]
    if args.window:  # a view from a high window or rooftop: only points on block outlines
        spots = list(raster.facade_spots(plan.near(lat, lon, args.radius), lat, lon, args.radius))
    print(f"  {len(spots)} spots", file=sys.stderr)
    hfovs = [args.hfov] if args.hfov else [50.0, 62.0, 75.0]
    ag, cam, hd, hf, eye, hz = raster.search(pr, labels, spots, lambda cam, hf: list(np.arange(0, 360, 12.0)), hfovs,
                                             eyes, horizons, keep=40)
    la, lo = plan2d.to_latlon(cam)
    return [{"lat": round(la, 6), "lon": round(lo, 6), "heading": round(hd, 1), "hfov": hf, "eye_m": eye,
             "horizon": round(hz, 3), "agreement": round(ag, 4), "map": f"https://www.google.com/maps?q={la:.6f},{lo:.6f}"}]


def _render_block(args, b, width):
    """A block number at a known column: every block with that number is first checked
    cheaply against what the photo shows (water, greenery nearby); the survivors are searched
    (on mapped water with --on-water, else a grid) with the heading pinned by the block's
    direction, and ranked by render-and-compare."""
    import numpy as np
    from PIL import Image

    from . import plan2d
    from .skyline import _exif_hfov

    raster, plan, labels, eyes, horizons, comp = _render_setup(args, b)
    num, x = args.block.rsplit("=", 1)
    x = float(x) / width if float(x) > 1 else float(x)
    hfov = args.hfov or _exif_hfov(Image.open(args.image))
    hfovs = [hfov] if hfov else [50.0, 60.0, 70.0]
    out = []
    for bid in plan2d.block_candidates(b, num):
        c0 = b.footprint(bid)[0].mean(0)
        la0, lo0 = float(c0[0]), float(c0[1])
        why = _fits_scene(plan, comp, la0, lo0, args.radius)
        if why:
            print(f"  {b.labels[bid]}: skipped ({why})", file=sys.stderr)
            continue
        cxy = plan2d.to_xy(la0, lo0)
        pr = raster.PlanRaster(plan.near(la0, lo0, args.radius + 1000), la0, lo0, args.radius + 1000, b)
        st = max(args.step, 8.0)
        k = int(args.radius // st)
        spots = [cxy + (i * st, j * st) for i in range(-k, k + 1) for j in range(-k, k + 1)
                 if 30 <= np.hypot(i, j) * st <= args.radius]
        if args.on_water:
            spots = [p for p in spots if pr.kind[tuple(((p - pr.origin) / pr.res).astype(int))] == 2]
        if not spots:
            print(f"  {b.labels[bid]}: no candidate spots", file=sys.stderr)
            continue
        ag, cam, hd, hf, eye, hz = raster.search(
            pr, labels, spots, lambda cam, hf: [plan2d._bearing(cxy - cam) - plan2d.angle_from_x(x, hf)], hfovs, eyes,
            horizons, keep=10)
        la, lo = plan2d.to_latlon(cam)
        print(f"  {b.labels[bid]}: agreement {ag:.3f} ({len(spots)} spots)", file=sys.stderr)
        out.append({"block": b.labels[bid], "lat": round(la, 6), "lon": round(lo, 6), "heading": round(hd, 1),
                    "hfov": hf, "eye_m": eye, "horizon": round(hz, 3), "agreement": round(ag, 4),
                    "map": f"https://www.google.com/maps?q={la:.6f},{lo:.6f}"})
    return sorted(out, key=lambda o: -o["agreement"])


def _parse_landmark(text: str, buildings, names, width: int):
    """'NAME[:part]=X' -> plan2d.Landmark. NAME: 'lat,lon', a building address, or a place name."""
    from . import plan2d

    if "=" not in text:
        raise SystemExit(f"landmark needs =column: {text!r}")
    name, x = text.rsplit("=", 1)
    part = "centre"
    if ":" in name and name.rsplit(":", 1)[1] in ("left", "right", "centre", "center"):
        name, part = name.rsplit(":", 1)
        part = "centre" if part == "center" else part
    x = float(x)
    x = x / width if x > 1 else x
    key = name.strip().lower()
    for i, label in enumerate(buildings.labels):  # a building address: use its outline
        if label and (label.lower() == key or label.lower() == "blk " + key):
            pts, _ = buildings.footprint(i)
            c = pts.mean(0)
            return plan2d.Landmark(x, float(c[0]), float(c[1]), part, building=i)
    lat, lon = _resolve_area(name, None, names)
    return plan2d.Landmark(x, lat, lon, part)


def _resolve_area(text: str, transit, names) -> tuple[float, float]:
    """'lat,lon', or a station / place name (stations first)."""
    parts = text.split(",")
    if len(parts) == 2:
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            pass
    t = text.lower()
    for st in (transit or {}).get("stations", []):
        if t in st["name"].lower():
            return st["lat"], st["lon"]
    for name, lat, lon in (names.entries if names else []):
        if t == name.lower() or t in name.lower():
            return lat, lon
    raise SystemExit(f"unknown area: {text!r} (give lat,lon)")


if __name__ == "__main__":
    sys.exit(main())
