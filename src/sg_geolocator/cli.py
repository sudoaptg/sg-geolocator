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

    t = sub.add_parser("evaluate-text", help="Score the text-clue stage on held-out Mapillary photos")
    t.add_argument("metadata", help="metadata.csv from fetch-mapillary")
    t.add_argument("--test-fraction", type=float, default=0.1)
    t.add_argument("--limit", type=int)
    t.add_argument("--hires", action="store_true", help="Fetch 2048 px photos (needs MAPILLARY_TOKEN)")
    t.add_argument("--json-out")
    t.add_argument("--md-out")

    args = parser.parse_args(argv)

    if args.cmd == "predict":
        from .pipeline import Geolocator

        geo = Geolocator(index_path=args.index, use_exif=not args.no_exif, use_text=not args.no_text,
                         geocell_path=args.geocell)
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
        from .embedder import DEFAULT_MODEL, ClipEmbedder
        from .retrieval import build_index, build_index_from_csv

        embedder = ClipEmbedder(args.model or DEFAULT_MODEL)
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
        clf = GeoCellClassifier(cell_deg=args.cell).fit(data["embeddings"], data["coords"], log=print)
        clf.save(args.out, model=model)
        print(f"Saved area classifier ({len(clf.centroids)} cells) -> {args.out}")
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
