"""
FAISS baseline indexes for fair benchmarking.

Wraps real FAISS implementations (not hand-written NumPy) so comparisons
against TurboQuant are credible.
"""

import numpy as np
import time
from typing import Tuple

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit norm (required for inner product search)."""
    vectors = np.ascontiguousarray(vectors.astype(np.float32))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8)


class FAISSFlatIndex:
    """FAISS exact inner product search — used as ground truth."""

    def __init__(self, dim: int):
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu is required: pip install faiss-cpu")
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self._n_vectors = 0
        self.build_time = 0.0

    def add(self, vectors: np.ndarray):
        vectors = _normalize(vectors)
        t0 = time.time()
        self.index.add(vectors)
        self._n_vectors = self.index.ntotal
        self.build_time = time.time() - t0

    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        queries = _normalize(queries)
        k = min(k, self._n_vectors)
        scores, indices = self.index.search(queries, k)
        return scores, indices

    @property
    def memory_bytes(self):
        return self._n_vectors * self.dim * 4

    def stats(self) -> dict:
        return {
            "n_vectors": self._n_vectors,
            "dim": self.dim,
            "bits": 32,
            "memory_mb": self.memory_bytes / (1024 * 1024),
            "compression_ratio": "1.0x",
            "build_time_s": f"{self.build_time:.3f}",
        }


class FAISSPQIndex:
    """FAISS Product Quantization — real PQ baseline (not hand-written)."""

    def __init__(self, dim: int, m: int = 8, nbits: int = 8):
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu is required: pip install faiss-cpu")
        self.dim = dim
        self._m_target = m
        self.nbits = nbits
        self._n_vectors = 0
        self.build_time = 0.0
        self._m = m
        self.index = None

    def add(self, vectors: np.ndarray):
        vectors = _normalize(vectors)
        n = vectors.shape[0]

        # Adjust m to divide dim evenly
        self._m = self._m_target
        while self.dim % self._m != 0 and self._m > 1:
            self._m -= 1

        self.index = faiss.IndexPQ(
            self.dim, self._m, self.nbits, faiss.METRIC_INNER_PRODUCT
        )

        t0 = time.time()
        self.index.train(vectors)
        self.index.add(vectors)
        self._n_vectors = self.index.ntotal
        self.build_time = time.time() - t0

    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        queries = _normalize(queries)
        k = min(k, self._n_vectors)
        scores, indices = self.index.search(queries, k)
        return scores, indices

    @property
    def memory_bytes(self):
        sub_dim = self.dim // max(self._m, 1)
        code_bytes = self._n_vectors * self._m  # 1 byte per subspace per vector
        codebook_bytes = self._m * (2 ** self.nbits) * sub_dim * 4
        return code_bytes + codebook_bytes

    def stats(self) -> dict:
        uncompressed = self._n_vectors * self.dim * 4
        mem = max(self.memory_bytes, 1)
        return {
            "n_vectors": self._n_vectors,
            "dim": self.dim,
            "bits": f"8 (FAISS PQ, m={self._m})",
            "memory_mb": mem / (1024 * 1024),
            "compression_ratio": f"{uncompressed / mem:.1f}x",
            "build_time_s": f"{self.build_time:.3f}",
        }


class FAISSIVFPQIndex:
    """FAISS IVF-PQ — inverted file + product quantization."""

    def __init__(self, dim: int, nlist: int = 100, m: int = 8,
                 nbits: int = 8, nprobe: int = 10):
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu is required: pip install faiss-cpu")
        self.dim = dim
        self._nlist_target = nlist
        self._m_target = m
        self.nbits = nbits
        self.nprobe = nprobe
        self._n_vectors = 0
        self.build_time = 0.0
        self._m = m
        self._nlist = nlist
        self.index = None

    def add(self, vectors: np.ndarray):
        vectors = _normalize(vectors)
        n = vectors.shape[0]

        # Adjust nlist for small datasets
        self._nlist = max(1, min(self._nlist_target, n // 39))

        # Adjust m to divide dim evenly
        self._m = self._m_target
        while self.dim % self._m != 0 and self._m > 1:
            self._m -= 1

        quantizer = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIVFPQ(
            quantizer, self.dim, self._nlist, self._m, self.nbits,
            faiss.METRIC_INNER_PRODUCT
        )
        self.index.nprobe = min(self.nprobe, self._nlist)

        t0 = time.time()
        self.index.train(vectors)
        self.index.add(vectors)
        self._n_vectors = self.index.ntotal
        self.build_time = time.time() - t0

    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        queries = _normalize(queries)
        k = min(k, self._n_vectors)
        scores, indices = self.index.search(queries, k)
        return scores, indices

    @property
    def memory_bytes(self):
        sub_dim = self.dim // max(self._m, 1)
        code_bytes = self._n_vectors * self._m
        codebook_bytes = self._m * (2 ** self.nbits) * sub_dim * 4
        ivf_bytes = self._nlist * self.dim * 4
        return code_bytes + codebook_bytes + ivf_bytes

    def stats(self) -> dict:
        uncompressed = self._n_vectors * self.dim * 4
        mem = max(self.memory_bytes, 1)
        return {
            "n_vectors": self._n_vectors,
            "dim": self.dim,
            "bits": f"8 (FAISS IVF-PQ, m={self._m})",
            "memory_mb": mem / (1024 * 1024),
            "compression_ratio": f"{uncompressed / mem:.1f}x",
            "build_time_s": f"{self.build_time:.3f}",
        }
