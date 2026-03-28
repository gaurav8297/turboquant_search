"""
TurboQuant Search
=================
Vector compression for similarity search, inspired by
TurboQuant (Zandieh et al., ICLR 2026, arXiv:2504.19874).

Uses random orthogonal rotation + Lloyd-Max quantization + sign-bit refinement.
"""

from .core import TurboQuantSearchIndex, FlatSearchIndex, ProductQuantizationIndex
from .faiss_baselines import FAISS_AVAILABLE
from .benchmarks import run_benchmark, compute_recall

if FAISS_AVAILABLE:
    from .faiss_baselines import FAISSFlatIndex, FAISSPQIndex, FAISSIVFPQIndex

__version__ = "0.1.0"
__all__ = [
    "TurboQuantSearchIndex",
    "FlatSearchIndex",
    "ProductQuantizationIndex",
    "FAISS_AVAILABLE",
    "run_benchmark",
    "compute_recall",
]
