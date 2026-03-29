[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/turboquant-search)](https://pypi.org/project/turboquant-search/)

# TurboQuant Search

**Vector compression for similarity search**, inspired by Google Research's [TurboQuant](https://arxiv.org/abs/2504.19874).

Compresses vector embeddings by **6-10x** with **up to 92% Recall@10** (4-bit), **zero training**, and instant indexing. Uses random orthogonal rotation + Lloyd-Max optimal quantization + sign-bit refinement.

[LinkedIn](https://www.linkedin.com/in/tarun5986)

## Try It

```bash
pip install turboquant-search
tqs demo
```

Or without pip:

```bash
git clone https://github.com/tarun-ks/turboquant_search.git
cd turboquant_search
./run.sh
```

Both launch an interactive comparison dashboard (TurboQuant vs FAISS) in your browser.

![Dashboard — Memory, Build Time, Compression, and Recall comparison](docs/dashboard.png)

## When to Use TurboQuant Search

- **Your data changes frequently** — no retraining needed; add vectors one at a time or in bulk
- **You need instant indexing** — streaming data, real-time updates, no offline codebook step
- **You want predictable compression** — same ratio regardless of dataset, no tuning knobs
- **You're prototyping** — skip FAISS codebook configuration; just set `bits=3` and go
- **You need a lightweight dependency** — pure NumPy, no C++ build required

## Install

```bash
pip install turboquant-search            # core + CLI
pip install turboquant-search[all]       # + Gradio dashboard + FAISS baselines
```

## Quick Start

```python
from turboquant_search import TurboQuantSearchIndex
import numpy as np

# Your embeddings (e.g., from sentence-transformers, OpenAI, etc.)
# Here we simulate 10K document embeddings of dimension 128
document_embeddings = np.random.randn(10000, 128).astype(np.float32)

# Create a compressed index — no training needed
index = TurboQuantSearchIndex(dim=128, bits=3)
index.add(document_embeddings)

# Search with a query embedding
query_embedding = np.random.randn(1, 128).astype(np.float32)
scores, top_k_indices = index.search(query_embedding, k=10)

print(f"Top 10 results: {top_k_indices[0]}")
print(f"Compression: {index.stats()['compression_ratio']}")
# -> '7.5x' (3-bit + sign-bit refinement)
```

Works with any embedding model — just pass in your vectors. No codebook training, no dataset-specific tuning.

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

## Supported Embedding Dimensions

Works with any embedding model. Common configurations:

| Model | Dim | Provider |
|-------|-----|----------|
| all-MiniLM-L6-v2 | 384 | sentence-transformers |
| bge-base-en-v1.5 | 768 | BAAI |
| bge-large-en-v1.5 | 1024 | BAAI |
| text-embedding-3-small | 1536 | OpenAI |
| text-embedding-3-large | 3072 | OpenAI |
| embed-v4 | 1024 | Cohere |
| voyage-3 | 1024 | Voyage AI |
| gemini-embedding-001 | 3072 | Google |
| nomic-embed-text-v1.5 | 768 | Nomic |

Just pass your vectors in — TurboQuant handles any dimension with the same compression ratio.

## Vector Database Integration

TurboQuant Search is a **standalone compressed index** — it is not a drop-in plugin for Pinecone, Weaviate, Qdrant, or Milvus. In its current form, it's best used for:

- **Standalone search** on small-to-medium datasets (up to ~1M vectors) where you want compression without a database
- **Prototyping** compression tradeoffs before committing to a production vector DB
- **Understanding** the TurboQuant algorithm — the code is readable NumPy, not optimized C++

To use with a vector DB, you would compress with TurboQuant, then store the compressed representation in the DB's raw storage layer — but this requires custom integration per DB. Most production vector DBs already have built-in PQ/SQ compression options that are more tightly integrated with their indexing.

## Limitations & Honest Comparison

- **Reference implementation, not a FAISS replacement** — pure NumPy, no GPU, no sub-linear search (HNSW/IVF). Use this to understand and prototype TurboQuant-style compression, not for production serving at scale.
- **Brute-force search** — scans all vectors. Benchmarks measure compression quality and recall, not search latency.
- **PQ/IVF-PQ benchmarks at 10K vectors understate their performance** — product quantization improves with more training data. At 1M+ vectors, the FAISS PQ recall gap narrows.
- **Stage 2 differs from paper** — sign-bit refinement instead of QJL. See "Difference from the Paper" below.

## How It Works

**Stage 1: Rotation + Lloyd-Max Quantization** — Multiply by a random orthogonal matrix (QR of Gaussian). Each coordinate becomes ~N(0, 1/d). Apply the optimal scalar quantizer for this distribution (b bits per coordinate). Store quantization indices + vector norm.

**Stage 2: Sign-Bit Refinement** — Split each quantization bin at its centroid. Store 1 extra bit (above/below) per coordinate. This doubles effective resolution from 2^b to 2^(b+1) levels using the conditional expectation of each half-bin.

**Asymmetric Search** — Queries are rotated but not quantized. Inner products are preserved since the rotation is orthogonal: `<Pi*q, Pi*x> = <q, x>`.

## Difference from the Paper

Stage 1 (rotation + Lloyd-Max) follows the paper's approach. Stage 2 diverges:

| | Paper (QJL) | This Implementation (Sign-Bit) |
|---|---|---|
| **Goal** | Unbiased inner product estimation | Low-variance ranking |
| **Method** | Gaussian random projection + sign bit | Split quantization bins at centroid, store half-bin indicator |
| **Best for** | KV cache compression (need unbiased estimates) | Search/ranking (need correct ordering, not exact values) |
| **Recall impact** | Baseline | **Higher recall** for ranking tasks |

**Why the difference matters:** The paper's QJL stage gives unbiased estimates of inner products — critical when you need the actual numeric value (e.g., attention scores in KV cache). But for nearest-neighbor *search*, you only need the *ranking* to be correct. Sign-bit refinement trades unbiasedness for lower variance, which means fewer ranking inversions and better recall for search use cases.

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

## CLI

```bash
tqs demo                              # Launch interactive comparison dashboard
tqs benchmark --dataset synthetic     # Run benchmarks, print results table
tqs index --input vectors.npy --bits 3  # Index custom numpy embeddings
tqs search --index tq.index --query query.npy --top 10  # Search an existing index
```

## Project Structure

```
turboquant_search/
  core.py             # TurboQuant algorithm (rotation, Lloyd-Max, sign-bit refinement)
  faiss_baselines.py  # FAISS wrappers (Flat, PQ, IVF-PQ)
  benchmarks.py       # Benchmark runner
  datasets.py         # Dataset loaders (synthetic, SIFT-128, GloVe-100)
  dataset_hub.py      # Pre-embedded dataset download & cache
  cli.py              # CLI entry point (tqs command)
tests/
  test_core.py        # 36 unit tests
app.py                # Gradio comparison dashboard
run.sh                # One-command setup + launch
```

## Roadmap

This is currently a **reference implementation** for understanding and prototyping TurboQuant-style compression. Here's what would move it toward production use:

- [ ] **C++ core with pybind11** — the hot path (reconstruct + score + top-k) is ~200 lines of C++; this alone would make search latency competitive with FAISS
- [ ] **IVF-TQ hybrid index** — combine IVF partitioning with TQ compression for sub-linear search on 1M+ vectors; the compression is training-free, only k-means centroids needed
- [ ] **HNSW-TQ** — plug TQ asymmetric scoring into an HNSW graph for logarithmic search with compression
- [ ] **SIFT-1M / Deep-1M benchmarks at scale** — current benchmarks run at 10K vectors; large-scale results would give a fairer comparison against FAISS PQ (which improves with more training data)
- [ ] **Real embedded datasets** — replace synthetic placeholders with actual Wikipedia/arxiv embeddings from sentence-transformers, hosted on HuggingFace Hub
- [ ] **GPU acceleration** — the rotation + quantization pipeline is embarrassingly parallel; CuPy or custom CUDA kernels would unlock million-vector indexing in seconds
- [ ] **Streaming / incremental updates** — TQ already supports per-vector compression with no retraining; formalizing this as a streaming API would differentiate from PQ which requires batch training

Contributions welcome — especially on the C++ core and large-scale benchmarks.

## License

Apache 2.0

---

*Independent implementation inspired by the TurboQuant paper. Not affiliated with Google Research.*
