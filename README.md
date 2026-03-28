[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![arXiv](https://img.shields.io/badge/arXiv-2504.19874-b31b1b.svg)](https://arxiv.org/abs/2504.19874)

# TurboQuant Search

**Vector compression for similarity search**, inspired by Google Research's [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., ICLR 2026).

Compresses vector embeddings by **6-10x** with **84-92% Recall@10**, **zero training**, and instant indexing. Uses random orthogonal rotation + Lloyd-Max optimal quantization + sign-bit refinement.

## Install

```bash
pip install turboquant-search
```

## Quick Start

```python
from turboquant_search import TurboQuantSearchIndex
import numpy as np

index = TurboQuantSearchIndex(dim=128, bits=3)

vectors = np.random.randn(10000, 128).astype(np.float32)
index.add(vectors)  # no training step

query = np.random.randn(1, 128).astype(np.float32)
scores, indices = index.search(query, k=10)

print(index.stats())
```

## Benchmark Results

### Synthetic (10K vectors, dim=128, 200 queries)

| Method | Memory | Compression | Recall@1 | Recall@10 |
|--------|--------|-------------|----------|-----------|
| Flat (exact) | 4.9 MB | 1.0x | 100% | 100% |
| PQ (FAISS) | 0.2 MB | 24.3x | 53% | 19% |
| IVF-PQ (FAISS) | 0.3 MB | 19.5x | 41% | 13% |
| **TQ 4-bit** | **0.8 MB** | **6.1x** | **95%** | **92%** |
| **TQ 3-bit** | **0.6 MB** | **7.5x** | **87%** | **85%** |
| **TQ 2-bit** | **0.5 MB** | **9.8x** | **80%** | **72%** |

### SIFT-128 (10K vectors, 200 queries)

| Method | Memory | Compression | Recall@1 | Recall@10 |
|--------|--------|-------------|----------|-----------|
| Flat (exact) | 4.9 MB | 1.0x | 100% | 100% |
| PQ (FAISS) | 0.2 MB | 24.3x | 22% | 39% |
| IVF-PQ (FAISS) | 0.3 MB | 19.5x | 26% | 42% |
| **TQ 4-bit** | **0.8 MB** | **6.1x** | **73%** | **84%** |
| **TQ 3-bit** | **0.6 MB** | **7.5x** | **60%** | **73%** |
| **TQ 2-bit** | **0.5 MB** | **9.8x** | **44%** | **55%** |

### GloVe-100 (10K vectors, 200 queries)

| Method | Memory | Compression | Recall@1 | Recall@10 |
|--------|--------|-------------|----------|-----------|
| Flat (exact) | 3.8 MB | 1.0x | 100% | 100% |
| PQ (FAISS) | 0.1 MB | 26.2x | 11% | 21% |
| IVF-PQ (FAISS) | 0.2 MB | 20.8x | 13% | 24% |
| **TQ 4-bit** | **0.6 MB** | **6.0x** | **89%** | **92%** |
| **TQ 3-bit** | **0.5 MB** | **7.4x** | **83%** | **83%** |
| **TQ 2-bit** | **0.4 MB** | **9.6x** | **66%** | **72%** |

All baselines use faiss-cpu. Results are deterministic (seed=42).

## How It Works

**Stage 1: Rotation + Lloyd-Max Quantization** — Multiply by a random orthogonal matrix (QR of Gaussian). Each coordinate becomes ~N(0, 1/d). Apply the optimal scalar quantizer for this distribution (b bits per coordinate). Store quantization indices + vector norm.

**Stage 2: Sign-Bit Refinement** — Split each quantization bin at its centroid. Store 1 extra bit (above/below) per coordinate. This doubles effective resolution from 2^b to 2^(b+1) levels using the conditional expectation of each half-bin.

**Asymmetric Search** — Queries are rotated but not quantized. Inner products are preserved since the rotation is orthogonal.

**Difference from the paper** — Stage 1 is inspired by the paper's approach. Stage 2 diverges: the paper uses QJL (Gaussian random projection + sign) for unbiased inner product estimation (important for KV cache). We use sign-bit refinement instead, which gives +7-11pp recall for search because ranking accuracy requires low variance, not unbiased estimates.

## Running Locally

Quickest way:

```bash
git clone https://github.com/tarun-ks/turboquant_search.git
cd turboquant_search
./run.sh
```

`run.sh` creates a virtual environment and installs all dependencies automatically.

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

## Project Structure

```
turboquant_search/
  core.py             # TurboQuant algorithm (rotation, Lloyd-Max, sign-bit refinement)
  faiss_baselines.py  # FAISS wrappers (Flat, PQ, IVF-PQ)
  benchmarks.py       # Benchmark runner
  datasets.py         # Dataset loaders (synthetic, SIFT-128, GloVe-100)
tests/
  test_core.py        # 36 unit tests
app.py                # Gradio demo
run.sh                # One-command setup + launch
```

## Limitations

- **NumPy reference implementation** — not optimized for production. No GPU, no sub-linear search (HNSW/IVF).
- **Brute-force search** — scans all vectors. Benchmarks compression quality, not search latency at scale.
- **Stage 2 differs from paper** — sign-bit refinement instead of QJL. See "How It Works" above.

## Citation

Inspired by:

```bibtex
@inproceedings{zandieh2026turboquant,
  title={Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## License

Apache 2.0

---

*Independent implementation inspired by the TurboQuant paper. Not affiliated with Google Research.*
