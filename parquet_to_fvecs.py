#!/usr/bin/env python3
"""
Convert vectors stored in parquet files into a single .fvecs file.

Example:
  python3 parquet_to_fvecs.py \
    --directory /path/to/parquet_dir \
    --vector-column embedding \
    --num-files 10 \
    --output vectors.fvecs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read N parquet files from a directory and write a single .fvecs file.",
    )
    parser.add_argument(
        "--directory",
        required=True,
        help="Directory containing parquet files.",
    )
    parser.add_argument(
        "--vector-column",
        required=True,
        help="Name of the column containing vector embeddings.",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        required=True,
        help="Number of parquet files to read from the directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output .fvecs file.",
    )
    return parser.parse_args()


def _to_float32_vector(value: Any, expected_dim: int | None) -> np.ndarray:
    if hasattr(value, "as_py"):
        value = value.as_py()

    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1:
        raise ValueError(f"Expected a 1D vector, got shape={vector.shape}")

    if expected_dim is not None and vector.shape[0] != expected_dim:
        raise ValueError(
            f"Inconsistent vector dimension: expected {expected_dim}, got {vector.shape[0]}"
        )
    return vector


def _iter_parquet_vectors(parquet_path: Path, vector_column: str):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "This script requires pyarrow. Install it with `pip install pyarrow`."
        ) from exc

    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(columns=[vector_column]):
        if vector_column not in batch.schema.names:
            raise ValueError(f"Column '{vector_column}' not found in {parquet_path}")

        column = batch.column(batch.schema.get_field_index(vector_column))
        for value in column:
            yield value


def _write_fvec(handle, vector: np.ndarray) -> None:
    dim_prefix = np.array([vector.shape[0]], dtype="<i4")
    handle.write(dim_prefix.tobytes())
    handle.write(np.asarray(vector, dtype="<f4").tobytes())


def main() -> None:
    args = parse_args()

    directory = Path(args.directory)
    if not directory.is_dir():
        raise ValueError(f"Directory does not exist: {directory}")
    if args.num_files <= 0:
        raise ValueError("--num-files must be > 0")

    parquet_files = sorted(directory.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {directory}")
    if args.num_files > len(parquet_files):
        raise ValueError(
            f"Requested {args.num_files} files, but only found {len(parquet_files)} parquet files"
        )

    selected_files = parquet_files[: args.num_files]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_vectors = 0
    vector_dim: int | None = None

    with output_path.open("wb") as handle:
        for parquet_path in selected_files:
            print(f"Reading {parquet_path}")
            for raw_vector in _iter_parquet_vectors(parquet_path, args.vector_column):
                vector = _to_float32_vector(raw_vector, vector_dim)
                if vector_dim is None:
                    vector_dim = vector.shape[0]
                    print(f"Detected vector dimension: {vector_dim}")
                _write_fvec(handle, vector)
                total_vectors += 1

    print(f"Wrote {total_vectors} vectors to {output_path}")


if __name__ == "__main__":
    main()
