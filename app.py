"""
TurboQuant Search — Interactive Comparison Dashboard

Search Comparison: side-by-side TurboQuant vs FAISS PQ results
with live stats (memory, build time, recall, compression).

Launch via:
  - `tqs demo` (CLI with pre-loaded data)
  - `python app.py` (standalone with synthetic data)
"""

import gradio as gr
import numpy as np
import tempfile
import os

_TMPDIR = tempfile.gettempdir()

from turboquant_search.core import TurboQuantSearchIndex, FlatSearchIndex
from turboquant_search.faiss_baselines import FAISS_AVAILABLE
from turboquant_search.benchmarks import compute_recall

if not FAISS_AVAILABLE:
    print("\n" + "=" * 60)
    print("WARNING: faiss-cpu not installed.")
    print("FAISS baselines will use NumPy fallback.")
    print("Fix:  pip install faiss-cpu")
    print("=" * 60 + "\n")


# ──────────────────────────────────────────────────────────
# Global state for the comparison dashboard
# ──────────────────────────────────────────────────────────

class DashboardState:
    """Holds indexes and data for the current session."""

    def __init__(self):
        self.vectors = None
        self.texts = None
        self.info = None
        self.dim = None
        self.n_vectors = 0

        self.tq_index = None
        self.tq_bits = 3
        self.pq_index = None
        self.flat_index = None

        self.build_stats = {}

    def build_indexes(self, vectors, texts, bits=3):
        """Build all indexes from vectors."""
        self.vectors = vectors
        self.texts = texts if texts else [f"Vector #{i}" for i in range(len(vectors))]
        self.dim = vectors.shape[1]
        self.n_vectors = vectors.shape[0]
        self.tq_bits = bits
        self.build_stats = {}

        # Flat (ground truth)
        if FAISS_AVAILABLE:
            from turboquant_search.faiss_baselines import FAISSFlatIndex
            self.flat_index = FAISSFlatIndex(self.dim)
        else:
            self.flat_index = FlatSearchIndex(self.dim)
        self.flat_index.add(vectors)
        self.build_stats["Flat (exact)"] = {
            "memory_mb": self.flat_index.memory_bytes / (1024 * 1024),
            "build_time": self.flat_index.build_time,
            "compression": 1.0,
        }

        # TurboQuant — selected bits + all variants for stats chart
        for b in [2, 3, 4]:
            tq = TurboQuantSearchIndex(self.dim, bits=b, seed=42)
            tq.add(vectors)
            self.build_stats[f"TurboQuant {b}-bit"] = {
                "memory_mb": tq.memory_bytes / (1024 * 1024),
                "build_time": tq.build_time,
                "compression": tq.compression_ratio,
            }
            if b == bits:
                self.tq_index = tq

        # FAISS PQ
        self.pq_index = None
        if FAISS_AVAILABLE:
            from turboquant_search.faiss_baselines import FAISSPQIndex, FAISSIVFPQIndex
            m = 8
            while self.dim % m != 0 and m > 1:
                m -= 1
            self.pq_index = FAISSPQIndex(self.dim, m=m, nbits=8)
            self.pq_index.add(vectors)
            self.build_stats["FAISS PQ"] = {
                "memory_mb": self.pq_index.memory_bytes / (1024 * 1024),
                "build_time": self.pq_index.build_time,
                "compression": vectors.shape[0] * self.dim * 4 / max(self.pq_index.memory_bytes, 1),
            }

            ivfpq = FAISSIVFPQIndex(self.dim, nlist=max(1, min(100, self.n_vectors // 39)),
                                     m=m, nbits=8, nprobe=10)
            ivfpq.add(vectors)
            self.build_stats["FAISS IVF-PQ"] = {
                "memory_mb": ivfpq.memory_bytes / (1024 * 1024),
                "build_time": ivfpq.build_time,
                "compression": vectors.shape[0] * self.dim * 4 / max(ivfpq.memory_bytes, 1),
            }

    def search(self, query_vector, k=10):
        """Search both TQ and PQ, return results."""
        if self.tq_index is None:
            return None, None, None

        q = query_vector.reshape(1, -1).astype(np.float32)
        gt_scores, gt_indices = self.flat_index.search(q, k=k)
        tq_scores, tq_indices = self.tq_index.search(q, k=k)

        pq_scores, pq_indices = None, None
        if self.pq_index is not None:
            pq_scores, pq_indices = self.pq_index.search(q, k=k)

        return (
            (gt_scores[0], gt_indices[0]),
            (tq_scores[0], tq_indices[0]),
            (pq_scores[0], pq_indices[0]) if pq_indices is not None else None,
        )


_state = DashboardState()


# ──────────────────────────────────────────────────────────
# Chart helpers
# ──────────────────────────────────────────────────────────

def _color_for(name):
    if "Flat" in name: return "#94a3b8"
    if "IVF" in name: return "#f87171"
    if "FAISS PQ" in name or ("PQ" in name and "TurboQuant" not in name): return "#fb923c"
    if "4-bit" in name: return "#3b82f6"
    if "3-bit" in name: return "#8b5cf6"
    if "2-bit" in name: return "#ec4899"
    return "#6ee7b7"


def _style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#e2e8f0")
    ax.tick_params(colors="#64748b", labelsize=9)


def _make_stats_chart(build_stats, recall_data=None):
    """Create the live stats dashboard chart (2x2)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(build_stats.keys())
    # Sort: Flat first, then TQ variants by bits, then FAISS
    order = []
    for n in names:
        if "Flat" in n:
            order.append((0, 0, n))
        elif "TurboQuant" in n:
            bits = int(n.split()[1].split("-")[0])
            order.append((1, bits, n))
        elif "IVF" in n:
            order.append((3, 0, n))
        else:
            order.append((2, 0, n))
    order.sort()
    names = [n for _, _, n in order]

    colors = [_color_for(n) for n in names]
    x = np.arange(len(names))

    fig, axs = plt.subplots(2, 2, figsize=(16, 10), facecolor="white")
    fig.subplots_adjust(hspace=0.38, wspace=0.28, top=0.93, bottom=0.08,
                        left=0.08, right=0.97)

    # 1. Memory usage
    ax = axs[0, 0]
    mem = [build_stats[n]["memory_mb"] for n in names]
    bars = ax.bar(x, mem, color=colors, edgecolor="white", linewidth=1.5, width=0.6, zorder=3)
    for b, v in zip(bars, mem):
        lbl = f"{v:.1f}" if v >= 1 else f"{v:.2f}"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(mem) * 0.03,
                f"{lbl} MB", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=20, ha="right")
    ax.set_title("Memory Usage", fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("MB")
    ax.yaxis.grid(True, alpha=0.12, zorder=0)
    _style_ax(ax)

    # 2. Index build time
    ax = axs[0, 1]
    bt = [build_stats[n]["build_time"] for n in names]
    bars = ax.bar(x, bt, color=colors, edgecolor="white", linewidth=1.5, width=0.6, zorder=3)
    for b, v in zip(bars, bt):
        lbl = f"{v:.3f}s" if v < 1 else f"{v:.1f}s"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(max(bt), 0.001) * 0.04,
                lbl, ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=20, ha="right")
    ax.set_title("Index Build Time", fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("Seconds")
    ax.yaxis.grid(True, alpha=0.12, zorder=0)
    _style_ax(ax)

    # 3. Compression ratio
    ax = axs[1, 0]
    comp = [build_stats[n]["compression"] for n in names]
    bars = ax.bar(x, comp, color=colors, edgecolor="white", linewidth=1.5, width=0.6, zorder=3)
    for b, v in zip(bars, comp):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(comp) * 0.03,
                f"{v:.1f}x", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=20, ha="right")
    ax.set_title("Compression Ratio", fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("Ratio (higher = better)")
    ax.yaxis.grid(True, alpha=0.12, zorder=0)
    _style_ax(ax)

    # 4. Recall@10
    ax = axs[1, 1]
    if recall_data:
        r_names = list(recall_data.keys())
        r_names_sorted = sorted(r_names, key=lambda n: (
            0 if "Flat" in n else 1 if "TurboQuant" in n else 2
        ))
        r_colors = [_color_for(n) for n in r_names_sorted]
        r_vals = [recall_data[n] * 100 for n in r_names_sorted]
        r_x = np.arange(len(r_names_sorted))
        bars = ax.bar(r_x, r_vals, color=r_colors, edgecolor="white", linewidth=1.5, width=0.6, zorder=3)
        for b, v in zip(bars, r_vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(r_x)
        ax.set_xticklabels(r_names_sorted, fontsize=8, rotation=20, ha="right")
        ax.set_ylim(0, 115)
    else:
        ax.text(0.5, 0.5, "Run a search to see recall", ha="center", va="center",
                fontsize=12, color="#94a3b8", transform=ax.transAxes)
    ax.set_title("Recall@10", fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("Recall (%)")
    ax.yaxis.grid(True, alpha=0.12, zorder=0)
    _style_ax(ax)

    path = os.path.join(_TMPDIR, "tq_stats.png")
    fig.savefig(path, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────
# Dashboard logic
# ──────────────────────────────────────────────────────────

def _load_default_data():
    """Load synthetic data for standalone mode."""
    from turboquant_search.datasets import load_synthetic
    vectors, queries, label = load_synthetic(10000, 200, 128, 42)
    texts = [f"Synthetic document #{i+1} (dim=128, cluster-based)" for i in range(vectors.shape[0])]
    return vectors, queries, texts


def _try_load_demo_data():
    """Try to load CLI-provided demo data, fall back to synthetic."""
    try:
        from turboquant_search._app_launcher import get_demo_data
        v, q, t, info = get_demo_data()
        if v is not None:
            return v, q, t, info
    except (ImportError, Exception):
        pass
    return None


def init_dashboard(dataset_choice, bits):
    """Initialize or re-initialize the dashboard with given settings."""
    bits = int(bits)

    # Parse dataset choice — strip parenthetical model name if present
    # e.g. "wikipedia-768 (BGE-base)" -> "wikipedia-768"
    ds_key = dataset_choice.split(" (")[0].strip() if dataset_choice else "synthetic"

    # Try CLI-provided data first
    demo_data = _try_load_demo_data()
    if demo_data is not None and ds_key == "demo":
        vectors, queries, texts, info = demo_data
    elif ds_key != "synthetic":
        try:
            from turboquant_search.dataset_hub import load_dataset
            vectors, queries, texts, info = load_dataset(ds_key)
        except Exception:
            vectors, queries, texts = _load_default_data()
            info = None
    else:
        vectors, queries, texts = _load_default_data()
        info = None

    _state.build_indexes(vectors, texts, bits=bits)

    # Compute recall with sample queries
    from turboquant_search.datasets import load_synthetic
    if info:
        rng = np.random.RandomState(42)
        qi = rng.choice(vectors.shape[0], size=min(200, vectors.shape[0] // 5), replace=False)
        sample_q = vectors[qi].copy()
        sample_q += rng.randn(*sample_q.shape).astype(np.float32) * 0.05
        norms = np.linalg.norm(sample_q, axis=1, keepdims=True)
        sample_q = sample_q / np.maximum(norms, 1e-8)
    else:
        _, sample_q, _ = load_synthetic(10000, 200, 128, 42)
        sample_q = sample_q[:min(200, _state.n_vectors // 5)]

    # Ground truth
    _, gt_idx = _state.flat_index.search(sample_q, k=10)
    recall_data = {"Flat (exact)": 1.0}

    # TQ recall
    _, tq_idx = _state.tq_index.search(sample_q, k=10)
    tq_recall = compute_recall(gt_idx, tq_idx, 10)
    recall_data[f"TurboQuant {bits}-bit"] = tq_recall

    # PQ recall
    if _state.pq_index is not None:
        _, pq_idx = _state.pq_index.search(sample_q, k=10)
        pq_recall = compute_recall(gt_idx, pq_idx, 10)
        recall_data["FAISS PQ"] = pq_recall

    chart = _make_stats_chart(_state.build_stats, recall_data)
    n = _state.n_vectors
    dim = _state.dim
    summary = f"Indexed **{n:,}** vectors in **{dim}** dimensions"

    return chart, summary


def _format_results(indices, scores, gt_set, header):
    """Format a method's top-10 results as markdown."""
    lines = []
    for rank, (idx, score) in enumerate(zip(indices, scores)):
        text = _state.texts[idx] if idx < len(_state.texts) else f"Vector #{idx}"
        match = "+" if idx in gt_set else "-"
        lines.append(f"{match} **{rank+1}.** [{idx}] {text[:80]}  (score: {score:.4f})")
    matches = len(set(indices.tolist()) & gt_set)
    return f"{header} -- {matches}/10 match exact\n\n" + "\n\n".join(lines)


def do_search(query_idx_str, bits):
    """Perform a search using a random query vector (by index)."""
    if _state.vectors is None:
        return "Not initialized. Click Re-index first.", "", "", None

    bits = int(bits)

    try:
        qi = int(query_idx_str) % _state.n_vectors
    except (ValueError, TypeError):
        qi = np.random.randint(0, _state.n_vectors)

    rng = np.random.RandomState(qi)
    query = _state.vectors[qi].copy()
    query += rng.randn(len(query)).astype(np.float32) * 0.05
    query /= np.linalg.norm(query)

    gt_res, tq_res, pq_res = _state.search(query, k=10)

    if gt_res is None:
        return "Index not built.", "", "", None

    gt_scores, gt_indices = gt_res
    gt_set = set(gt_indices.tolist())

    # TQ results
    tq_text = _format_results(tq_res[1], tq_res[0], gt_set, f"### TurboQuant ({bits}-bit)")

    # FAISS PQ results
    if pq_res is not None:
        pq_text = _format_results(pq_res[1], pq_res[0], gt_set, "### FAISS PQ")
    else:
        pq_text = "### FAISS PQ (not available)\n\n*Install faiss-cpu for PQ comparison.*"

    # Ground truth
    gt_lines = [f"**{r+1}.** [{idx}] {_state.texts[idx][:80] if idx < len(_state.texts) else f'Vector #{idx}'}  (score: {s:.4f})"
                for r, (idx, s) in enumerate(zip(gt_indices, gt_scores))]
    gt_text = "### Exact (ground truth)\n\n" + "\n\n".join(gt_lines)

    return tq_text, pq_text, gt_text, f"Query: perturbed copy of vector #{qi}"


def random_search(bits):
    """Search with a random query index."""
    qi = np.random.randint(0, max(_state.n_vectors, 1))
    return do_search(str(qi), bits)


# ──────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────

_DESC = """# TurboQuant Search

Compresses vector embeddings by **6-10x** with **84-92% Recall@10** and **zero training**.

Inspired by [TurboQuant](https://arxiv.org/abs/2504.19874). \
Uses random orthogonal rotation + Lloyd-Max optimal quantization + sign-bit refinement. \
All baselines use real [faiss-cpu](https://github.com/facebookresearch/faiss). \
[GitHub](https://github.com/tarun-ks/turboquant_search) | \
[PyPI](https://pypi.org/project/turboquant-search/)"""

_CSS = """
.metric {
    background: linear-gradient(135deg, #eff6ff, #f0f9ff);
    border: 1px solid #bfdbfe; border-radius: 12px;
    padding: 16px; text-align: center;
}
.metric h2 { margin: 0; font-size: 1.6rem; color: #1e40af; }
.metric p { margin: 4px 0 0; font-size: 0.82rem; color: #475569; }
"""

with gr.Blocks(
    title="TurboQuant Search",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan"),
    css=_CSS,
) as demo:

    gr.Markdown(_DESC)

    # Key metrics
    with gr.Row():
        gr.HTML('<div class="metric"><h2>6-10x</h2><p>Compression</p></div>')
        gr.HTML('<div class="metric"><h2>84-92%</h2><p>Recall@10 (4-bit)</p></div>')
        gr.HTML('<div class="metric"><h2>~10x</h2><p>Faster than FAISS PQ</p></div>')
        gr.HTML('<div class="metric"><h2>0</h2><p>Training needed</p></div>')

    with gr.Tabs():

        # ── SEARCH COMPARISON ──
        with gr.TabItem("Search Comparison"):

            gr.Markdown("### Side-by-Side Search: TurboQuant vs FAISS PQ")
            gr.Markdown("*`+` = matches ground truth top-10. `-` = miss.*")

            # Controls
            with gr.Row():
                ds_search = gr.Dropdown(
                    [
                        "synthetic",
                        "wikipedia-384 (MiniLM)",
                        "wikipedia-768 (BGE-base)",
                        "wikipedia-1536 (OpenAI-dim)",
                        "arxiv-384 (MiniLM)",
                        "arxiv-1024 (BGE-large)",
                    ],
                    value="synthetic",
                    label="Dataset (name-dim)",
                    scale=2,
                )
                bits_slider = gr.Radio(["2", "3", "4"], value="3", label="TurboQuant bits")
                reindex_btn = gr.Button("Re-index", variant="primary", scale=1)
                search_btn = gr.Button("Random Search", variant="secondary", scale=1)

            # Status
            index_summary = gr.Markdown(value="*Loading...*")

            # Stats chart
            stats_chart = gr.Image(type="filepath", show_label=False, height=500)

            # Search results
            gr.Markdown("---")
            query_info = gr.Markdown(value="")
            with gr.Row():
                tq_results = gr.Markdown(value="*Loading...*")
                pq_results = gr.Markdown(value="")
            with gr.Accordion("Ground Truth (exact)", open=False):
                gt_results = gr.Markdown(value="")

            # Wire up
            reindex_btn.click(
                fn=init_dashboard,
                inputs=[ds_search, bits_slider],
                outputs=[stats_chart, index_summary],
            )
            search_btn.click(
                fn=random_search,
                inputs=[bits_slider],
                outputs=[tq_results, pq_results, gt_results, query_info],
            )

        # ── HOW IT WORKS ──
        with gr.TabItem("How It Works"):
            gr.Markdown("""## How It Works

**Stage 1** — Multiply vector by random orthogonal matrix. Each coordinate becomes ~N(0, 1/d).
Apply Lloyd-Max optimal quantizer (b bits per coord). Store index + norm.

**Stage 2** — Split each quantization bin at centroid. Store 1 extra bit (above/below).
Doubles effective resolution from 2^b to 2^(b+1) levels.

**Search** — Query is rotated but NOT quantized. Inner products preserved: <Pi*q, Pi*x> = <q, x>.

**vs Original Paper** — Stage 1 (rotation + Lloyd-Max) is inspired by the paper's approach.
Stage 2 diverges: the paper uses QJL for unbiased estimation (suited for KV cache);
we use sign-bit refinement (better for search ranking, +7-11pp recall).

**vs FAISS PQ** — No training, instant per-vector compression, near-optimal distortion (within 2.7x of lower bound).
PQ compresses more (24x vs 6-10x) but needs training on your data.

[Paper](https://arxiv.org/abs/2504.19874) |
[QJL](https://arxiv.org/abs/2406.03482) |
[PolarQuant](https://arxiv.org/abs/2502.02617)""")

    gr.Markdown("---\n*Independent implementation inspired by the TurboQuant paper. Not affiliated with Google Research.*")

    # Auto-initialize on page load
    def _on_load():
        chart, summary = init_dashboard("synthetic", "3")
        tq, pq, gt, qi = random_search("3")
        return chart, summary, tq, pq, gt, qi

    demo.load(
        fn=_on_load,
        outputs=[stats_chart, index_summary, tq_results, pq_results, gt_results, query_info],
    )

demo.queue()
if __name__ == "__main__":
    demo.launch(show_error=True)
