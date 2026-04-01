#!/usr/bin/env python3
"""
Show Lloyd-Max centroids and boundaries for a given bit width.

Examples:
  python3 show_lloyd_centroids.py
  python3 show_lloyd_centroids.py --bits 1
  python3 show_lloyd_centroids.py --bits 1 --dim 128
  python3 show_lloyd_centroids.py --bits 1 --format json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from turboquant_search.core import _lloyd_max_codebook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print Lloyd-Max centroids and boundaries for TurboQuant bit widths.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=1,
        help="Quantization bit width to inspect. Default: 1",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="If provided, also print the centroids/boundaries scaled by 1/sqrt(dim), as used by TurboQuant.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format. Default: text",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    centroids, boundaries = _lloyd_max_codebook(args.bits)
    payload = {
        "bits": args.bits,
        "centroids_raw": centroids.tolist(),
        "boundaries_raw": boundaries.tolist(),
    }

    if args.dim is not None:
        scale = float(np.sqrt(args.dim))
        payload["dim"] = args.dim
        payload["centroids_scaled"] = (centroids / scale).tolist()
        payload["boundaries_scaled"] = (boundaries / scale).tolist()

    if args.format == "json":
        print(json.dumps(payload, indent=2))
        return

    print(f"bits: {payload['bits']}")
    print("raw centroids:")
    for idx, value in enumerate(payload["centroids_raw"]):
        print(f"  [{idx}] {value:.10f}")

    if payload["boundaries_raw"]:
        print("raw boundaries:")
        for idx, value in enumerate(payload["boundaries_raw"]):
            print(f"  [{idx}] {value:.10f}")
    else:
        print("raw boundaries: []")

    if args.dim is not None:
        print(f"scaled for dim={args.dim}:")
        print("  centroids:")
        for idx, value in enumerate(payload["centroids_scaled"]):
            print(f"    [{idx}] {value:.10f}")

        if payload["boundaries_scaled"]:
            print("  boundaries:")
            for idx, value in enumerate(payload["boundaries_scaled"]):
                print(f"    [{idx}] {value:.10f}")
        else:
            print("  boundaries: []")


if __name__ == "__main__":
    main()
