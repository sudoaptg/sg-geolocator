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

    e = sub.add_parser("evaluate", help="Score retrieval on held-out photos of an index")
    e.add_argument("index")
    e.add_argument("--test-fraction", type=float, default=0.1)
    e.add_argument("--json-out")
    e.add_argument("--md-out")

    args = parser.parse_args(argv)

    if args.cmd == "predict":
        from .pipeline import Geolocator

        geo = Geolocator(index_path=args.index, use_exif=not args.no_exif)
        for img in args.images:
            pred = geo.predict(img)
            print(json.dumps({"image": img, **(pred.as_dict() if pred else {"error": "no prediction"})}))
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
        from .embedder import ClipEmbedder
        from .retrieval import build_index, build_index_from_csv

        if args.metadata:
            n = build_index_from_csv(args.metadata, ClipEmbedder(), args.out)
        else:
            n = build_index(args.folder, ClipEmbedder(), args.out)
        print(f"Indexed {n} photos -> {args.out}")
        return 0

    from .evaluate import evaluate_index, to_markdown

    result = evaluate_index(args.index, args.test_fraction, log=print)
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
