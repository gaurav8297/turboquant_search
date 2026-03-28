"""
TurboQuant Search — Core Algorithm
===================================

Vector compression for similarity search, inspired by TurboQuant
(Zandieh et al., ICLR 2026, arXiv:2504.19874).

Technique:
  Stage 1: Random orthogonal rotation + Lloyd-Max optimal scalar
           quantization per coordinate.
  Stage 2: Sign-bit refinement — 1 extra bit per coordinate
           (above/below bin centroid), doubling effective resolution.

The original paper's Stage 2 uses QJL for unbiased inner product
estimation (suited for KV cache). We use sign-bit refinement instead
because search ranking needs low variance, not unbiased estimates.
This gives +7-11pp recall improvement on search benchmarks.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import time


# ─────────────────────────────────────────────────────────────
# Module-level caches for expensive computations
# ─────────────────────────────────────────────────────────────

_LLOYD_MAX_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
_ROTATION_CACHE: Dict[Tuple[int, int], np.ndarray] = {}

# Maximum elements in score matrix before batched search kicks in (~200MB)
_SCORE_MATRIX_LIMIT = 50_000_000


def _lloyd_max_codebook(bits: int, n_iter: int = 300, grid_size: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Lloyd-Max optimal codebook for the Beta(d/2, d/2) distribution.
    Results are cached by bit width since the codebook only depends on bits.

    After random orthogonal rotation, each coordinate of a unit vector follows
    approximately Beta(d/2, d/2) centered at 0. For large d, this is well
    approximated by a Gaussian N(0, 1/d). We use the Gaussian approximation
    for the Lloyd-Max optimization.

    Returns:
        centroids: (2^bits,) optimal reconstruction levels
        boundaries: (2^bits - 1,) decision boundaries
    """
    if bits in _LLOYD_MAX_CACHE:
        return _LLOYD_MAX_CACHE[bits]

    n_levels = 2 ** bits

    # Initialize with uniform quantile spacing on N(0,1)
    from scipy.stats import norm
    quantiles = np.linspace(0, 1, n_levels + 1)[1:-1]
    boundaries = norm.ppf(quantiles)

    # PDF for optimization - use standard normal
    x_grid = np.linspace(-4.0, 4.0, grid_size)
    pdf_vals = norm.pdf(x_grid)

    for _ in range(n_iter):
        # Compute centroids as conditional expectations
        centroids = np.zeros(n_levels)
        all_bounds = np.concatenate([[-np.inf], boundaries, [np.inf]])

        for i in range(n_levels):
            mask = (x_grid >= all_bounds[i]) & (x_grid < all_bounds[i + 1])
            if mask.sum() > 0:
                weighted = (x_grid[mask] * pdf_vals[mask]).sum()
                total = pdf_vals[mask].sum()
                centroids[i] = weighted / total if total > 0 else 0.0
            else:
                centroids[i] = (all_bounds[i] + all_bounds[i + 1]) / 2.0

        # Update boundaries as midpoints
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

    result = (centroids, boundaries)
    _LLOYD_MAX_CACHE[bits] = result
    return result


def _get_rotation_matrix(dim: int, seed: int) -> np.ndarray:
    """Get or compute a cached random orthogonal rotation matrix via QR decomposition."""
    key = (dim, seed)
    if key in _ROTATION_CACHE:
        return _ROTATION_CACHE[key]

    rng = np.random.RandomState(seed)
    G = rng.randn(dim, dim).astype(np.float32)
    Q, _ = np.linalg.qr(G)
    _ROTATION_CACHE[key] = Q
    return Q


_SUB_CENTROID_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


def _get_sub_centroids(bits: int, dim: int) -> np.ndarray:
    """
    Compute sub-centroids for sign-bit refinement.

    For each Lloyd-Max bin, splits it at the centroid and computes
    the conditional expectation for the lower and upper halves.
    This gives 2x finer reconstruction using 1 extra bit (the sign
    of the residual within each bin).

    Returns:
        sub_centroids: (2^bits, 2) array — [bin_idx, 0=lower / 1=upper]
    """
    key = (bits, dim)
    if key in _SUB_CENTROID_CACHE:
        return _SUB_CENTROID_CACHE[key]

    from scipy.stats import norm

    centroids_raw, boundaries_raw = _lloyd_max_codebook(bits)
    scale = np.sqrt(dim)
    centroids = centroids_raw / scale
    boundaries = boundaries_raw / scale

    n_levels = 2 ** bits
    all_bounds = np.concatenate([[-4.0 / scale], boundaries, [4.0 / scale]])

    grid = np.linspace(-4.0 / scale, 4.0 / scale, 50000)
    pdf = norm.pdf(grid * scale) * scale  # PDF of N(0, 1/dim)

    sub_centroids = np.zeros((n_levels, 2), dtype=np.float32)
    for i in range(n_levels):
        lo, hi = all_bounds[i], all_bounds[i + 1]
        mid = centroids[i]

        mask_lo = (grid >= lo) & (grid < mid)
        if mask_lo.sum() > 0:
            sub_centroids[i, 0] = np.average(grid[mask_lo], weights=pdf[mask_lo])
        else:
            sub_centroids[i, 0] = (lo + mid) / 2

        mask_hi = (grid >= mid) & (grid <= hi)
        if mask_hi.sum() > 0:
            sub_centroids[i, 1] = np.average(grid[mask_hi], weights=pdf[mask_hi])
        else:
            sub_centroids[i, 1] = (mid + hi) / 2

    _SUB_CENTROID_CACHE[key] = sub_centroids
    return sub_centroids


class TurboQuantSearchIndex:
    """
    TurboQuant-compressed vector search index.

    Compresses high-dimensional vectors using random orthogonal rotation
    followed by per-coordinate Lloyd-Max quantization, with optional
    sign-bit refinement that doubles effective resolution using 1 extra
    bit per coordinate.

    Parameters
    ----------
    dim : int
        Dimensionality of input vectors.
    bits : int
        Bits per coordinate for quantization (2, 3, or 4).
    use_residual_sign : bool
        Whether to apply sign-bit refinement (1 extra bit per coordinate).
        Doubles effective quantization levels. Default True.
    seed : int
        Random seed for reproducibility of the rotation matrix.
    """

    def __init__(self, dim: int, bits: int = 3, use_residual_sign: bool = True, seed: int = 42,
                 # Backward compat alias
                 use_qjl: bool = None):
        self.dim = dim
        self.bits = bits
        # use_qjl is kept as an alias for backward compatibility
        if use_qjl is not None:
            use_residual_sign = use_qjl
        self.use_qjl = use_residual_sign  # backward compat property
        self.use_residual_sign = use_residual_sign
        self.seed = seed
        self.n_levels = 2 ** bits

        # Cached random orthogonal rotation matrix via QR decomposition
        self.rotation_matrix = _get_rotation_matrix(dim, seed)

        # Cached Lloyd-Max codebook, scaled to match the distribution of
        # normalized rotated coordinates: N(0, 1/dim) instead of N(0, 1).
        centroids_raw, boundaries_raw = _lloyd_max_codebook(bits)
        dim_scale = np.sqrt(dim)
        self.centroids = (centroids_raw / dim_scale).astype(np.float32)
        self.boundaries = (boundaries_raw / dim_scale).astype(np.float32)

        # Sign-bit refinement: for each bin, pre-compute the conditional
        # centroid for the lower and upper halves. Storing 1 extra bit
        # (sign of residual) per coordinate doubles the effective resolution.
        if use_residual_sign:
            self.sub_centroids = _get_sub_centroids(bits, dim)

        # Storage
        self._indices = None       # (n, dim) uint8 quantization indices
        self._norms = None         # (n,) vector norms
        self._sign_bits = None     # (n, dim) 1-bit residual sign per coordinate
        self._n_vectors = 0

        # Metadata
        self.build_time = 0.0
        self.memory_bytes = 0
        self.memory_bytes_uncompressed = 0

    def _rotate(self, vectors: np.ndarray) -> np.ndarray:
        """Apply random orthogonal rotation."""
        return vectors @ self.rotation_matrix.T

    def _quantize_coords(self, rotated: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Scalar quantize each coordinate using Lloyd-Max codebook.

        Returns:
            indices: (n, dim) quantization bin indices
            reconstructed: (n, dim) dequantized values
            norms: (n,) vector norms
        """
        # Normalize by vector norms for unit-sphere quantization
        norms = np.linalg.norm(rotated, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = rotated / norms

        # Quantize using boundaries
        indices = np.digitize(normalized, self.boundaries).astype(np.uint8)

        # Reconstruct
        reconstructed = self.centroids[indices] * norms

        return indices, reconstructed, norms.reshape(-1)

    def _encode_sign_bits(self, normalized: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """
        Sign-bit refinement: store whether each coordinate's value is above
        or below its bin centroid. This 1 extra bit per coordinate doubles
        the effective quantization resolution.
        """
        residual = normalized - self.centroids[indices]
        return (residual >= 0).astype(np.uint8)

    def add(self, vectors: np.ndarray):
        """
        Add vectors to the index.

        Parameters
        ----------
        vectors : np.ndarray of shape (n, dim)
            Vectors to index. Will be compressed using TurboQuant.
        """
        assert vectors.shape[1] == self.dim, f"Expected dim={self.dim}, got {vectors.shape[1]}"
        vectors = vectors.astype(np.float32)
        # Replace NaN/inf with zeros to prevent matmul warnings
        vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)

        t0 = time.time()

        # Stage 1: Rotate and quantize
        rotated = self._rotate(vectors)
        indices, reconstructed, norms = self._quantize_coords(rotated)

        # Stage 2: Sign-bit refinement (1 extra bit per coordinate)
        sign_bits = None
        if self.use_residual_sign:
            normalized = rotated / np.maximum(norms[:, np.newaxis], 1e-8)
            sign_bits = self._encode_sign_bits(normalized, indices)

        # Store
        if self._indices is None:
            self._indices = indices
            self._norms = norms
            self._sign_bits = sign_bits
        else:
            self._indices = np.concatenate([self._indices, indices])
            self._norms = np.concatenate([self._norms, norms])
            if sign_bits is not None:
                self._sign_bits = np.concatenate([self._sign_bits, sign_bits])

        self._n_vectors += vectors.shape[0]
        self.build_time = time.time() - t0

        # Calculate memory usage
        bits_per_vector = self.bits * self.dim  # quantized coordinates
        bits_per_vector += 32  # norm (float32)
        if self.use_residual_sign:
            bits_per_vector += self.dim  # 1-bit sign per coordinate

        self.memory_bytes = (self._n_vectors * bits_per_vector) // 8
        self.memory_bytes_uncompressed = self._n_vectors * self.dim * 4  # float32

    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors using compressed representations.

        Uses asymmetric distance computation: query is kept in full precision,
        database vectors are in compressed form. Batches over queries when
        the score matrix would exceed the memory budget.

        Parameters
        ----------
        queries : np.ndarray of shape (nq, dim)
            Query vectors.
        k : int
            Number of neighbors to return.

        Returns
        -------
        distances : np.ndarray of shape (nq, k)
            Inner product scores (higher = more similar).
        indices : np.ndarray of shape (nq, k)
            Indices of nearest neighbors.
        """
        queries = queries.astype(np.float32)
        queries = np.nan_to_num(queries, nan=0.0, posinf=0.0, neginf=0.0)
        nq = queries.shape[0]
        k = min(k, self._n_vectors)

        # Rotate queries (but don't quantize — asymmetric search)
        q_rotated = self._rotate(queries)

        # Pre-compute database reconstruction using sign-bit refinement if available
        if self.use_residual_sign and self._sign_bits is not None:
            # Use sub-centroids: sub_centroids[bin_idx, sign_bit] for each coordinate
            db_reconstructed = self.sub_centroids[self._indices, self._sign_bits] * self._norms[:, np.newaxis]
        else:
            db_reconstructed = self.centroids[self._indices] * self._norms[:, np.newaxis]

        # Determine batch size to limit memory usage
        batch_size = max(1, _SCORE_MATRIX_LIMIT // max(self._n_vectors, 1))

        all_top_k_scores = np.empty((nq, k), dtype=np.float32)
        all_top_k_idx = np.empty((nq, k), dtype=np.int64)

        for start in range(0, nq, batch_size):
            end = min(start + batch_size, nq)
            batch_q = q_rotated[start:end]
            batch_nq = end - start

            # Compute inner products: q_rot . db_reconstructed
            # (Since rotation is orthogonal, <Rx, Ry> = <x, y>)
            scores = batch_q @ db_reconstructed.T  # (batch_nq, n)

            # Top-k selection
            if k >= self._n_vectors:
                top_k_idx = np.argsort(-scores, axis=1)[:, :k]
            else:
                top_k_idx = np.argpartition(-scores, k, axis=1)[:, :k]
                for i in range(batch_nq):
                    order = np.argsort(-scores[i, top_k_idx[i]])
                    top_k_idx[i] = top_k_idx[i][order]

            top_k_scores = np.take_along_axis(scores, top_k_idx, axis=1)

            all_top_k_scores[start:end] = top_k_scores
            all_top_k_idx[start:end] = top_k_idx

        return all_top_k_scores, all_top_k_idx

    def compress_with_details(self, vector: np.ndarray) -> dict:
        """
        Return intermediate compression results for visualization.

        Parameters
        ----------
        vector : np.ndarray of shape (dim,) or (1, dim)

        Returns
        -------
        dict with keys: original, rotated, norm, quantized_indices,
            reconstructed, residual, reconstruction_error,
            and if use_residual_sign: sign_bits, refined_reconstructed, refined_error
        """
        vector = vector.astype(np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        rotated = self._rotate(vector)
        indices, reconstructed, norms = self._quantize_coords(rotated)

        norms_val = float(norms) if np.ndim(norms) == 0 else float(norms[0])
        residual = (rotated - reconstructed).squeeze()

        result = {
            "original": vector.squeeze(),
            "rotated": rotated.squeeze(),
            "norm": norms_val,
            "quantized_indices": indices.squeeze(),
            "reconstructed": reconstructed.squeeze(),
            "residual": residual,
            "reconstruction_error": float(np.linalg.norm(residual)),
        }

        if self.use_residual_sign:
            normalized_v = rotated / np.maximum(norms_val, 1e-8)
            sign_bits = self._encode_sign_bits(normalized_v, indices)
            refined = self.sub_centroids[indices, sign_bits] * norms_val
            result["sign_bits"] = sign_bits.squeeze()
            result["refined_reconstructed"] = refined.squeeze()
            result["refined_error"] = float(np.linalg.norm((rotated - refined).squeeze()))

        return result

    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs float32."""
        if self.memory_bytes == 0:
            return 0.0
        return self.memory_bytes_uncompressed / self.memory_bytes

    def stats(self) -> dict:
        """Return index statistics."""
        return {
            "n_vectors": self._n_vectors,
            "dim": self.dim,
            "bits": self.bits,
            "residual_mode": "sign-refine" if self.use_residual_sign else "none",
            "memory_mb": self.memory_bytes / (1024 * 1024),
            "memory_uncompressed_mb": self.memory_bytes_uncompressed / (1024 * 1024),
            "compression_ratio": f"{self.compression_ratio:.1f}x",
            "build_time_s": f"{self.build_time:.3f}",
        }


class FlatSearchIndex:
    """Brute-force exact search baseline."""

    def __init__(self, dim: int):
        self.dim = dim
        self._vectors = None
        self._n_vectors = 0
        self.build_time = 0.0

    def add(self, vectors: np.ndarray):
        vectors = vectors.astype(np.float32)
        t0 = time.time()
        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.concatenate([self._vectors, vectors])
        self._n_vectors += vectors.shape[0]
        self.build_time = time.time() - t0

    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        queries = queries.astype(np.float32)
        k = min(k, self._n_vectors)
        scores = queries @ self._vectors.T
        if k >= self._n_vectors:
            top_k_idx = np.argsort(-scores, axis=1)[:, :k]
        else:
            top_k_idx = np.argpartition(-scores, k, axis=1)[:, :k]
            for i in range(queries.shape[0]):
                order = np.argsort(-scores[i, top_k_idx[i]])
                top_k_idx[i] = top_k_idx[i][order]
        top_k_scores = np.take_along_axis(scores, top_k_idx, axis=1)
        return top_k_scores, top_k_idx

    @property
    def memory_bytes(self):
        return self._n_vectors * self.dim * 4

    def stats(self) -> dict:
        return {
            "n_vectors": self._n_vectors,
            "dim": self.dim,
            "bits": 32,
            "memory_mb": self.memory_bytes / (1024 * 1024),
            "compression_ratio": "1.0x (baseline)",
            "build_time_s": f"{self.build_time:.3f}",
        }


class ProductQuantizationIndex:
    """
    Product Quantization baseline for comparison.

    Splits vectors into subspaces and quantizes each independently
    using k-means clustering.
    """

    def __init__(self, dim: int, n_subspaces: int = 8, n_clusters: int = 256, seed: int = 42):
        assert dim % n_subspaces == 0
        self.dim = dim
        self.n_subspaces = n_subspaces
        self.sub_dim = dim // n_subspaces
        self.n_clusters = n_clusters
        self.seed = seed

        self._codes = None  # (n, n_subspaces) uint8
        self._codebooks = None  # (n_subspaces, n_clusters, sub_dim)
        self._n_vectors = 0
        self.build_time = 0.0

    def _train_codebooks(self, vectors: np.ndarray):
        """Train PQ codebooks using k-means on subspaces."""
        from sklearn.cluster import MiniBatchKMeans

        n = vectors.shape[0]
        actual_clusters = min(self.n_clusters, n)
        self._actual_clusters = actual_clusters
        self._codebooks = np.zeros((self.n_subspaces, actual_clusters, self.sub_dim), dtype=np.float32)

        for m in range(self.n_subspaces):
            sub_vectors = vectors[:, m * self.sub_dim:(m + 1) * self.sub_dim]
            kmeans = MiniBatchKMeans(
                n_clusters=actual_clusters,
                random_state=self.seed,
                batch_size=min(1000, n),
                n_init=1,
                max_iter=20
            )
            kmeans.fit(sub_vectors)
            self._codebooks[m] = kmeans.cluster_centers_

    def add(self, vectors: np.ndarray):
        vectors = vectors.astype(np.float32)
        t0 = time.time()

        # Train codebooks
        self._train_codebooks(vectors)

        # Encode
        codes = np.zeros((vectors.shape[0], self.n_subspaces), dtype=np.uint8)
        for m in range(self.n_subspaces):
            sub_vectors = vectors[:, m * self.sub_dim:(m + 1) * self.sub_dim]
            # Find nearest centroid
            dists = np.sum((sub_vectors[:, np.newaxis, :] - self._codebooks[m][np.newaxis, :, :]) ** 2, axis=2)
            codes[:, m] = np.argmin(dists, axis=1).astype(np.uint8)

        self._codes = codes
        self._n_vectors = vectors.shape[0]
        self.build_time = time.time() - t0

    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        queries = queries.astype(np.float32)
        nq = queries.shape[0]
        k = min(k, self._n_vectors)

        # Precompute distance tables
        # For inner product: score = sum_m q_m . codebook[m][code[m]]
        n_cb = self._codebooks.shape[1]
        dist_tables = np.zeros((nq, self.n_subspaces, n_cb), dtype=np.float32)
        for m in range(self.n_subspaces):
            q_sub = queries[:, m * self.sub_dim:(m + 1) * self.sub_dim]
            dist_tables[:, m, :] = q_sub @ self._codebooks[m].T

        # Compute scores using lookup
        scores = np.zeros((nq, self._n_vectors), dtype=np.float32)
        for m in range(self.n_subspaces):
            scores += dist_tables[:, m, :][:, self._codes[:, m]]

        # Top-k
        if k >= self._n_vectors:
            top_k_idx = np.argsort(-scores, axis=1)[:, :k]
        else:
            top_k_idx = np.argpartition(-scores, k, axis=1)[:, :k]
            for i in range(nq):
                order = np.argsort(-scores[i, top_k_idx[i]])
                top_k_idx[i] = top_k_idx[i][order]

        top_k_scores = np.take_along_axis(scores, top_k_idx, axis=1)
        return top_k_scores, top_k_idx

    @property
    def memory_bytes(self):
        # codes: n * n_subspaces * 8 bits
        # codebooks: n_subspaces * n_clusters * sub_dim * 32 bits
        code_bytes = self._n_vectors * self.n_subspaces
        codebook_bytes = self.n_subspaces * self.n_clusters * self.sub_dim * 4
        return code_bytes + codebook_bytes

    def stats(self) -> dict:
        uncompressed = self._n_vectors * self.dim * 4
        return {
            "n_vectors": self._n_vectors,
            "dim": self.dim,
            "bits": f"8 (PQ, {self.n_subspaces} subspaces)",
            "memory_mb": self.memory_bytes / (1024 * 1024),
            "compression_ratio": f"{uncompressed / self.memory_bytes:.1f}x",
            "build_time_s": f"{self.build_time:.3f}",
        }
