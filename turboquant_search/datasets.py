"""
Dataset loaders for vector search benchmarks.

Each loader returns (db_vectors, query_vectors, dataset_name) or None on failure.
All vectors are normalized to unit norm for inner product search.
"""

import numpy as np
from typing import Optional, Tuple


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize to unit norm."""
    vectors = vectors.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8)


def load_synthetic(
    n_vectors: int = 10000,
    n_queries: int = 200,
    dim: int = 128,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Generate synthetic clustered data. Always works, no download."""
    rng = np.random.RandomState(seed)

    n_clusters = max(10, n_vectors // 500)
    cluster_centers = rng.randn(n_clusters, dim).astype(np.float32)
    cluster_centers = _normalize(cluster_centers)

    vectors = []
    for i in range(n_vectors):
        c = i % n_clusters
        noise = rng.randn(dim).astype(np.float32) * 0.3
        v = cluster_centers[c] + noise
        vectors.append(v)
    vectors = _normalize(np.array(vectors, dtype=np.float32))

    queries = []
    for i in range(n_queries):
        if i < n_queries // 2:
            idx = rng.randint(n_vectors)
            noise = rng.randn(dim).astype(np.float32) * 0.1
            q = vectors[idx] + noise
        else:
            q = rng.randn(dim).astype(np.float32)
        queries.append(q)
    queries = _normalize(np.array(queries, dtype=np.float32))

    return vectors, queries, f"Synthetic clustered ({n_vectors:,}, dim={dim})"


def load_sift128(
    n_vectors: int = 10000,
    n_queries: int = 200,
    progress_fn=None,
) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Load SIFT-128 vectors from HuggingFace Hub.

    Source: open-vdb/sift-128-euclidean (standard ANN benchmark).
    Returns None if download fails.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Warning: 'datasets' package not installed. pip install datasets")
        return None

    try:
        if progress_fn:
            progress_fn("Loading SIFT-128 (downloads ~100MB on first run)...")

        train_ds = load_dataset(
            "open-vdb/sift-128-euclidean", "train",
            split=f"train[:{n_vectors}]",
        )
        test_ds = load_dataset(
            "open-vdb/sift-128-euclidean", "test",
            split=f"test[:{n_queries}]",
        )

        if progress_fn:
            progress_fn("Processing vectors...")

        db_vectors = np.array(train_ds["emb"], dtype=np.float32)
        query_vectors = np.array(test_ds["emb"], dtype=np.float32)

        db_vectors = _normalize(db_vectors)
        query_vectors = _normalize(query_vectors)

        actual_n = db_vectors.shape[0]
        actual_nq = query_vectors.shape[0]

        return (
            db_vectors, query_vectors,
            f"SIFT-128 ({actual_n:,} db, {actual_nq} queries)"
        )

    except Exception as e:
        print(f"Warning: Could not load SIFT-128: {e}")
        return None


def load_glove100(
    n_vectors: int = 10000,
    n_queries: int = 200,
    progress_fn=None,
) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Load GloVe-100 word embeddings from HuggingFace Hub.

    Source: open-vdb/glove-100-angular.
    Queries are sampled from the test split.
    Returns None if download fails.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Warning: 'datasets' package not installed. pip install datasets")
        return None

    try:
        if progress_fn:
            progress_fn("Loading GloVe-100 (downloads ~150MB on first run)...")

        train_ds = load_dataset(
            "open-vdb/glove-100-angular", "train",
            split=f"train[:{n_vectors}]",
        )
        test_ds = load_dataset(
            "open-vdb/glove-100-angular", "test",
            split=f"test[:{n_queries}]",
        )

        if progress_fn:
            progress_fn("Processing vectors...")

        db_vectors = np.array(train_ds["emb"], dtype=np.float32)
        query_vectors = np.array(test_ds["emb"], dtype=np.float32)

        db_vectors = _normalize(db_vectors)
        query_vectors = _normalize(query_vectors)

        actual_n = db_vectors.shape[0]
        actual_nq = query_vectors.shape[0]

        return (
            db_vectors, query_vectors,
            f"GloVe-100 ({actual_n:,} db, {actual_nq} queries)"
        )

    except Exception as e:
        print(f"Warning: Could not load GloVe-100: {e}")
        return None


# Registry of available datasets
DATASET_LOADERS = {
    "synthetic": load_synthetic,
    "sift-128": load_sift128,
    "glove-100": load_glove100,
}

DATASET_LABELS = {
    "synthetic": "Synthetic clustered (10K, dim=128)",
    "sift-128": "SIFT-128 (requires: pip install datasets)",
    "glove-100": "GloVe-100 (requires: pip install datasets)",
}
