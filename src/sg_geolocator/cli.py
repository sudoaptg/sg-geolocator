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
            print(json.dumps({"image": img, **(pred.as_dict() if pred else {"error": "no prediction"})}))
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


if __name__ == "__main__":
    sys.exit(main())
