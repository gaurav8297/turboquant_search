#!/usr/bin/env python3
"""
Write the first N vectors from a large .fvecs file into a smaller .fvecs file.

Example:
  python3 slice_fvecs.py \
    --input /path/to/big.fvecs \
    --output /path/to/small.fvecs \
    --num-vectors 10000
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy the first N vectors from a .fvecs file into a new .fvecs file.",
    )
    parser.add_argument("--input", required=True, help="Path to the input .fvecs file.")
    parser.add_argument("--output", required=True, help="Path to the output .fvecs file.")
    parser.add_argument(
        "--num-vectors",
        type=int,
        required=True,
        help="Number of vectors to copy from the input file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise ValueError(f"Input file does not exist: {input_path}")
    if args.num_vectors <= 0:
        raise ValueError("--num-vectors must be > 0")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    copied = 0
    dim = None

    with input_path.open("rb") as src, output_path.open("wb") as dst:
        while copied < args.num_vectors:
            dim_bytes = src.read(4)
            if not dim_bytes:
                break
            if len(dim_bytes) != 4:
                raise ValueError("Input file ended mid-record while reading vector dimension")

            current_dim = int.from_bytes(dim_bytes, byteorder="little", signed=True)
            if current_dim <= 0:
                raise ValueError(f"Invalid vector dimension {current_dim} at vector {copied}")

            if dim is None:
                dim = current_dim
            elif current_dim != dim:
                raise ValueError(
                    f"Inconsistent vector dimension: expected {dim}, got {current_dim} at vector {copied}"
                )

            vector_bytes = src.read(current_dim * 4)
            if len(vector_bytes) != current_dim * 4:
                raise ValueError("Input file ended mid-record while reading vector values")

            dst.write(dim_bytes)
            dst.write(vector_bytes)
            copied += 1

    if copied < args.num_vectors:
        print(
            f"Warning: requested {args.num_vectors} vectors, but input only contained {copied}. "
            f"Wrote {copied} vectors."
        )
    else:
        print(f"Wrote {copied} vectors to {output_path}")


if __name__ == "__main__":
    main()
