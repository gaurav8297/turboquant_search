#!/usr/bin/env python3
"""
Export rotated queries and rotated-space reconstructed data as .fvecs files.

Outputs:
  <output_dir>/rotated_queries.fvecs
  <output_dir>/1bit/rotated_data_reconstructed.fvecs
  <output_dir>/2bit/rotated_data_reconstructed.fvecs
  <output_dir>/3bit/rotated_data_reconstructed.fvecs
  <output_dir>/4bit/rotated_data_reconstructed.fvecs
  <output_dir>/8bit/rotated_data_reconstructed.fvecs
  <output_dir>/2bit_1signbit/rotated_data_reconstructed.fvecs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from turboquant_search.core import TurboQuantSearchIndex
from turboquant_search.datasets import _read_fvecs


MODES = [
    ("1bit", 1, False),
    ("2bit", 2, False),
    ("3bit", 3, False),
    ("4bit", 4, False),
    ("8bit", 8, False),
    ("2bit_1signbit", 2, True),
]


def _write_fvecs(path: Path, vectors: np.ndarray) -> None:
    """Write float32 vectors to .fvecs format."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D float32 array, got shape={vectors.shape}")

    dim_prefix = np.array([vectors.shape[1]], dtype="<i4").tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for vector in vectors:
            handle.write(dim_prefix)
            handle.write(np.asarray(vector, dtype="<f4").tobytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export rotated queries and reconstructed rotated data as .fvecs files.",
    )
    parser.add_argument("--queries", required=True, help="Path to input queries.fvecs")
    parser.add_argument("--data", required=True, help="Path to input data.fvecs")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where rotated/reconstructed .fvecs files will be written",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the orthogonal rotation matrix",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    queries = _read_fvecs(args.queries)
    data = _read_fvecs(args.data)
    if queries.ndim != 2 or data.ndim != 2:
        raise ValueError("Both queries and data must be 2D arrays")
    if queries.shape[1] != data.shape[1]:
        raise ValueError(
            f"Dim mismatch: queries dim={queries.shape[1]}, data dim={data.shape[1]}"
        )

    dim = data.shape[1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rotation_index = TurboQuantSearchIndex(
        dim=dim,
        bits=1,
        use_residual_sign=False,
        seed=args.seed,
    )
    rotated_queries = rotation_index.rotate_vectors(queries)
    rotated_queries_path = output_dir / "rotated_queries.fvecs"
    _write_fvecs(rotated_queries_path, rotated_queries)
    print(f"Wrote {rotated_queries_path}")

    for mode_name, bits, use_residual_sign in MODES:
        index = TurboQuantSearchIndex(
            dim=dim,
            bits=bits,
            use_residual_sign=use_residual_sign,
            seed=args.seed,
        )
        reconstructed = index.reconstruct_rotated_vectors(data)
        output_path = output_dir / mode_name / "rotated_data_reconstructed.fvecs"
        _write_fvecs(output_path, reconstructed)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
