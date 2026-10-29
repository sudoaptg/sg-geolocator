"""Command-line interface: `sg-geolocate predict` and `sg-geolocate build-index`."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sg-geolocate")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="Estimate the location of one or more photos")
    p.add_argument("images", nargs="+")
    p.add_argument("--index", help="Retrieval index (.npz) from build-index")
    p.add_argument("--no-exif", action="store_true", help="Ignore EXIF GPS (useful for evaluation)")

    b = sub.add_parser("build-index", help="Build a retrieval index from geotagged photos")
    b.add_argument("folder")
    b.add_argument("--out", default="index.npz")

    args = parser.parse_args(argv)

    if args.cmd == "predict":
        from .pipeline import Geolocator

        geo = Geolocator(index_path=args.index, use_exif=not args.no_exif)
        for img in args.images:
            pred = geo.predict(img)
            print(json.dumps({"image": img, **(pred.as_dict() if pred else {"error": "no prediction"})}))
        return 0

    from .embedder import ClipEmbedder
    from .retrieval import build_index

    n = build_index(args.folder, ClipEmbedder(), args.out)
    print(f"Indexed {n} photos -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
