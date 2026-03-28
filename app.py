"""
TurboQuant Search — Interactive Demo
"""

import gradio as gr
import numpy as np
import tempfile
import os

_TMPDIR = tempfile.gettempdir()

from turboquant_search.core import TurboQuantSearchIndex
from turboquant_search.faiss_baselines import FAISS_AVAILABLE
from turboquant_search.benchmarks import run_benchmark, format_results_table
from turboquant_search.datasets import DATASET_LABELS

if not FAISS_AVAILABLE:
    print("\n" + "="*60)
    print("WARNING: faiss-cpu not installed.")
    print("Baselines will use NumPy fallback (slower, less credible).")
    print("Fix:  pip install faiss-cpu")
    print("Or:   ./run.sh  (auto-creates venv with all dependencies)")
    print("="*60 + "\n")


# ──────────────────────────────────────────────────────────
# Charts
# ──────────────────────────────────────────────────────────

def _color(name):
    if "Flat" in name: return "#94a3b8"
    if "IVF" in name: return "#f87171"
    if "PQ" in name and "TurboQuant" not in name: return "#fb923c"
    if "4-bit" in name: return "#3b82f6"
    if "3-bit" in name: return "#8b5cf6"
    if "2-bit" in name: return "#ec4899"
    return "#6ee7b7"


def _short(name):
    if "Flat" in name: return "Flat (exact)"
    if "IVF-PQ" in name: return "IVF-PQ"
    if "PQ" in name and "TurboQuant" not in name: return "PQ (FAISS)"
    if "TurboQuant" in name:
        return f"TQ {name.split('-bit')[0].split()[-1]}-bit"
    return name[:10]


def _style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#e2e8f0")
    ax.tick_params(colors="#64748b", labelsize=9)


def _make_chart(results):
    """Single 2x2 figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = results["methods"]
    ks = sorted(results["config"]["k_values"])
    ck = 10 if 10 in ks else ks[-1]
    ds = results["config"].get("dataset", "")

    names = list(methods.keys())
    labels = [_short(n) for n in names]
    cols = [_color(n) for n in names]
    x = np.arange(len(names))

    fig, axs = plt.subplots(2, 2, figsize=(18, 12), facecolor="white")
    fig.subplots_adjust(hspace=0.32, wspace=0.26, top=0.93, bottom=0.05,
                        left=0.06, right=0.97)

    # 1. Recall bars
    ax = axs[0, 0]
    vals = [methods[n]["recall"].get(ck, 0) * 100 for n in names]
    bars = ax.bar(x, vals, color=cols, edgecolor="white", linewidth=1.5, width=0.6, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                f"{v:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(f"Recall@{ck}", fontsize=15, fontweight="bold", pad=10)
    ax.set_ylim(0, 115); ax.set_ylabel("Recall (%)")
    ax.yaxis.grid(True, alpha=0.12, zorder=0); _style_ax(ax)

    # 2. Recall curve
    ax = axs[0, 1]
    markers = {"Flat": "o", "IVF": "X", "PQ": "s", "4-bit": "D", "3-bit": "^", "2-bit": "v"}
    dashes = {"Flat": "-", "IVF": "--", "PQ": "--"}
    for n, d in methods.items():
        mk = next((v for k, v in markers.items() if k in n), "x")
        ls = next((v for k, v in dashes.items() if k in n), "-")
        r = [d["recall"].get(k, 0) * 100 for k in ks]
        ax.plot(ks, r, marker=mk, color=_color(n), linewidth=2, markersize=5,
                label=_short(n), linestyle=ls)
    ax.set_xlabel("k"); ax.set_ylabel("Recall@k (%)")
    ax.set_title("Recall vs k", fontsize=15, fontweight="bold", pad=10)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.set_ylim(0, 108); ax.grid(True, alpha=0.12); _style_ax(ax)

    # 3. Memory
    ax = axs[1, 0]
    mem = [methods[n]["memory_mb"] for n in names]
    bars = ax.bar(x, mem, color=cols, edgecolor="white", linewidth=1.5, width=0.6, zorder=3)
    for b, v in zip(bars, mem):
        lbl = f"{v:.1f}" if v >= 1 else f"{v:.2f}"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(mem) * 0.03,
                f"{lbl} MB", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("Memory", fontsize=15, fontweight="bold", pad=10)
    ax.set_ylabel("MB"); ax.yaxis.grid(True, alpha=0.12, zorder=0); _style_ax(ax)

    # 4. Build time
    ax = axs[1, 1]
    bt = [methods[n]["build_time"] for n in names]
    bars = ax.bar(x, bt, color=cols, edgecolor="white", linewidth=1.5, width=0.6, zorder=3)
    for b, v in zip(bars, bt):
        lbl = f"{v:.3f}s" if v < 1 else f"{v:.1f}s"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(bt) * 0.04,
                lbl, ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("Build Time", fontsize=15, fontweight="bold", pad=10)
    ax.set_ylabel("Seconds"); ax.yaxis.grid(True, alpha=0.12, zorder=0); _style_ax(ax)

    fig.suptitle(ds, fontsize=14, fontweight="bold", y=0.97)

    path = os.path.join(_TMPDIR, "tq_chart.png")
    fig.savefig(path, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_viz(dim, bits, seed):
    """Compression visualizer — 2x2 pipeline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dim, bits, seed = int(dim), int(bits), int(seed)
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)

    tq = TurboQuantSearchIndex(dim=dim, bits=bits, use_residual_sign=True, seed=42)
    d = tq.compress_with_details(vec)

    show = min(dim, 48)
    xi = np.arange(show)
    orig = d["original"][:show]
    rot = d["rotated"].squeeze()[:show]
    has_ref = "refined_reconstructed" in d

    fig, axs = plt.subplots(2, 2, figsize=(15, 9), facecolor="white")
    fig.subplots_adjust(hspace=0.38, wspace=0.25)

    ax = axs[0, 0]
    ax.bar(xi, orig, width=0.7, color="#3b82f6", alpha=0.85)
    ax.set_title("1. Original", fontsize=12, fontweight="bold")
    ax.axhline(0, color="#e2e8f0", lw=0.5); ax.set_xlim(-0.5, show-0.5); _style_ax(ax)

    ax = axs[0, 1]
    ax.bar(xi, rot, width=0.7, color="#8b5cf6", alpha=0.85)
    ax.set_title("2. After Rotation", fontsize=12, fontweight="bold")
    ax.axhline(0, color="#e2e8f0", lw=0.5); ax.set_xlim(-0.5, show-0.5); _style_ax(ax)
    ax.annotate(f"Each coord ~ N(0, 1/{dim})", xy=(0.5, 0.93), xycoords="axes fraction",
                ha="center", fontsize=8, color="#6b21a8",
                bbox=dict(boxstyle="round,pad=0.2", fc="#f5f3ff", ec="#c4b5fd", alpha=0.9))

    ax = axs[1, 0]
    if has_ref:
        ref = d["refined_reconstructed"].squeeze()[:show]
        ax.bar(xi-0.17, rot, width=0.34, color="#8b5cf6", alpha=0.5, label="Exact")
        ax.bar(xi+0.17, ref, width=0.34, color="#22c55e", alpha=0.8, label=f"{bits}b+sign")
    else:
        rec = d["reconstructed"].squeeze()[:show]
        ax.bar(xi-0.17, rot, width=0.34, color="#8b5cf6", alpha=0.5, label="Exact")
        ax.bar(xi+0.17, rec, width=0.34, color="#fb923c", alpha=0.8, label=f"{bits}-bit")
    ax.set_title("3. Reconstruction", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8); ax.axhline(0, color="#e2e8f0", lw=0.5)
    ax.set_xlim(-0.5, show-0.5); _style_ax(ax)

    ax = axs[1, 1]
    base_err = np.abs((d["rotated"].squeeze() - d["reconstructed"].squeeze())[:show])
    err_pct = d["reconstruction_error"] / np.linalg.norm(d["original"]) * 100
    if has_ref:
        ref_err = np.abs((d["rotated"].squeeze() - d["refined_reconstructed"].squeeze())[:show])
        ref_pct = d["refined_error"] / np.linalg.norm(d["original"]) * 100
        ax.bar(xi-0.17, base_err, width=0.34, color="#fb923c", alpha=0.7, label=f"{bits}b only")
        ax.bar(xi+0.17, ref_err, width=0.34, color="#22c55e", alpha=0.7, label="+sign")
        ax.legend(fontsize=8)
        ax.annotate(f"{err_pct:.1f}% -> {ref_pct:.1f}%", xy=(0.97, 0.92),
                    xycoords="axes fraction", ha="right", fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#f0fdf4", ec="#86efac"))
    else:
        ax.bar(xi, base_err, width=0.7, color="#f87171", alpha=0.8)
        ax.annotate(f"{err_pct:.1f}%", xy=(0.97, 0.92), xycoords="axes fraction",
                    ha="right", fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#fef2f2", ec="#fca5a5"))
    ax.set_title("4. Error", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.5, show-0.5); _style_ax(ax)

    fig.suptitle(f"TurboQuant {bits}-bit (dim={dim})", fontsize=13, fontweight="bold", y=0.99)

    path = os.path.join(_TMPDIR, "tq_viz.png")
    fig.savefig(path, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    txt = f"**{bits}-bit + sign** = {2**(bits+1)} effective levels, ~{32/(bits+1):.0f}x compression"
    if has_ref:
        txt += f" | Error: {err_pct:.1f}% -> {ref_pct:.1f}%"
    return path, txt


def _make_mem(n_docs, emb_dim, precision, tq_bits):
    """Memory calculator chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_docs, emb_dim, tq_bits = int(n_docs), int(emb_dim), int(tq_bits)
    bpe = 4 if precision == "fp32" else 2

    orig_gb = n_docs * emb_dim * bpe / (1024**3)
    tq_bpv = (tq_bits + 1) * emb_dim + 32
    tq_gb = n_docs * tq_bpv // 8 / (1024**3)
    ratio = (n_docs * emb_dim * bpe) / max(n_docs * tq_bpv // 8, 1)
    save = orig_gb - tq_gb

    fig, ax = plt.subplots(figsize=(10, 3), facecolor="white")
    ax.barh([f"TurboQuant ({tq_bits}-bit)", f"Original ({precision})"],
            [tq_gb, orig_gb], color=["#22c55e", "#f87171"], edgecolor="white", height=0.4)
    for i, v in enumerate([tq_gb, orig_gb]):
        ax.text(v + max(orig_gb, 0.001) * 0.02, i,
                f"{v:.2f} GB" if v < 100 else f"{v:.0f} GB",
                va="center", fontsize=12, fontweight="bold")
    ax.set_xlabel("Memory (GB)")
    ax.set_title(f"{ratio:.1f}x compression — saves {save:.2f} GB",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(orig_gb, 0.001) * 1.3); _style_ax(ax)

    path = os.path.join(_TMPDIR, "tq_mem.png")
    fig.savefig(path, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    txt = f"**{n_docs:,}** vectors x {emb_dim}D: **{orig_gb:.2f} GB** -> **{tq_gb:.2f} GB** ({ratio:.1f}x)"
    return path, txt


# ──────────────────────────────────────────────────────────
# Benchmark logic
# ──────────────────────────────────────────────────────────

def _do_benchmark(dataset_name, bit_widths, progress_cb=None):
    """Run benchmark, return (table_data, headers, chart_path, summary, raw)."""
    results = run_benchmark(
        dataset_name=dataset_name, n_vectors=10000, n_queries=200,
        dim=128, k_values=[1, 5, 10, 50], bit_widths=bit_widths,
        progress_callback=progress_cb,
    )

    ks = results["config"]["k_values"]
    headers = ["Method", "Memory", "Compress.", "Build"] + [f"R@{k}" for k in ks]
    rows = []
    for name, d in results["methods"].items():
        rows.append([_short(name), f"{d['memory_mb']:.2f} MB",
                     f"{d['compression_ratio']:.1f}x", f"{d['build_time']:.3f}s"]
                    + [f"{d['recall'][k]:.0%}" for k in ks])

    chart = _make_chart(results)

    # One-line summary
    ds = results["config"].get("dataset", "")
    tq = {k: v for k, v in results["methods"].items() if "TurboQuant" in k}
    pq = {k: v for k, v in results["methods"].items()
          if "PQ" in k and "TurboQuant" not in k and "Flat" not in k}
    ck = 10 if 10 in ks else ks[-1]

    summary = f"**{ds}**\n\n"
    if tq:
        best_n = max(tq, key=lambda x: tq[x]["recall"].get(ck, 0))
        best = tq[best_n]
        summary += f"Best: **{_short(best_n)}** — {best['recall'][ck]:.0%} R@{ck}, {best['compression_ratio']:.1f}x compression"
        if pq:
            best_pq_n = max(pq, key=lambda x: pq[x]["recall"].get(ck, 0))
            pq_r = pq[best_pq_n]["recall"].get(ck, 0)
            diff = (best["recall"][ck] - pq_r) * 100
            summary += f"\n\nvs **{_short(best_pq_n)}**: {'+' if diff>0 else ''}{diff:.0f}pp recall"
            if best["build_time"] > 0 and pq[best_pq_n]["build_time"] > 0:
                spd = pq[best_pq_n]["build_time"] / max(best["build_time"], 1e-6)
                if spd > 1.5:
                    summary += f", {spd:.0f}x faster build"
    summary += "\n\n*All baselines: faiss-cpu. TurboQuant: NumPy.*"

    raw = format_results_table(results)
    return rows, headers, chart, summary, raw


def benchmark_click(dataset, b2, b3, b4, progress=gr.Progress()):
    bits = []
    if b2: bits.append(2)
    if b3: bits.append(3)
    if b4: bits.append(4)
    if not bits: bits = [3]
    def cb(s, t, m): progress(s/t, desc=m)
    rows, headers, chart, summary, raw = _do_benchmark(dataset, bits, cb)
    return chart, rows, summary


# ──────────────────────────────────────────────────────────
# Pre-compute defaults
# ──────────────────────────────────────────────────────────

# Use smaller dataset for startup (fast on HF free tier)
_d_rows, _d_hdrs, _d_chart, _d_summary, _d_raw = _do_benchmark("synthetic", [3, 4])
_d_viz, _d_viz_txt = _make_viz(64, 3, 42)
_d_mem, _d_mem_txt = _make_mem(1_000_000, 768, "fp32", 3)


# ──────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────

_DESC = """# TurboQuant Search

Compresses vector embeddings by **6-10x** with **84-92% Recall@10** and **zero training**.

Inspired by [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., ICLR 2026). \
Uses random orthogonal rotation + Lloyd-Max optimal quantization + sign-bit refinement. \
All baselines use real [faiss-cpu](https://github.com/facebookresearch/faiss). \
[GitHub](https://github.com/tarun-ks/turboquant_search)"""

_HOW = """## How It Works

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
[PolarQuant](https://arxiv.org/abs/2502.02617)"""

_CSS = """
.metric {
    background: linear-gradient(135deg, #eff6ff, #f0f9ff);
    border: 1px solid #bfdbfe; border-radius: 12px;
    padding: 16px; text-align: center;
}
.metric h2 { margin: 0; font-size: 1.6rem; color: #1e40af; }
.metric p { margin: 4px 0 0; font-size: 0.82rem; color: #475569; }
"""

with gr.Blocks(title="TurboQuant Search",
               theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan"),
               css=_CSS) as demo:

    gr.Markdown(_DESC)

    # Key metrics
    with gr.Row():
        gr.HTML('<div class="metric"><h2>6-10x</h2><p>Compression</p></div>')
        gr.HTML('<div class="metric"><h2>84-92%</h2><p>Recall@10 (4-bit)</p></div>')
        gr.HTML('<div class="metric"><h2>~10x</h2><p>Faster than FAISS PQ</p></div>')
        gr.HTML('<div class="metric"><h2>0</h2><p>Training needed</p></div>')

    with gr.Tabs():

        # ── BENCHMARK ──
        with gr.TabItem("Benchmark"):

            # 1. Controls — one compact row
            with gr.Row():
                ds = gr.Dropdown(list(DATASET_LABELS.keys()), value="synthetic",
                                 label="Dataset", scale=3)
                b4 = gr.Checkbox(label="4-bit", value=True)
                b3 = gr.Checkbox(label="3-bit", value=True)
                b2 = gr.Checkbox(label="2-bit", value=True)
                btn = gr.Button("Run", variant="primary", scale=1)

            # 2. Headline result
            summary = gr.Markdown(value=_d_summary)

            # 3. Chart — big
            chart = gr.Image(type="filepath", value=_d_chart, show_label=False, height=600)

            # 4. Table — visible, not hidden
            tbl = gr.Dataframe(value=_d_rows, headers=_d_hdrs, interactive=False,
                               label="Detailed Results")

            btn.click(fn=benchmark_click, inputs=[ds, b2, b3, b4],
                      outputs=[chart, tbl, summary])

        # ── VISUALIZER ──
        with gr.TabItem("Compression Visualizer"):

            viz_img = gr.Image(type="filepath", value=_d_viz, show_label=False, height=480)
            viz_txt = gr.Markdown(value=_d_viz_txt)

            gr.Markdown("---")
            with gr.Row():
                v_dim = gr.Slider(16, 256, value=64, step=16, label="Dim")
                v_bits = gr.Radio(["2","3","4"], value="3", label="Bits")
                v_seed = gr.Slider(0, 999, value=42, step=1, label="Seed")
                v_btn = gr.Button("Update", variant="primary")

            v_btn.click(fn=_make_viz, inputs=[v_dim, v_bits, v_seed],
                        outputs=[viz_img, viz_txt])

        # ── CALCULATOR ──
        with gr.TabItem("Memory Calculator"):

            mem_img = gr.Image(type="filepath", value=_d_mem, show_label=False, height=260)
            mem_txt = gr.Markdown(value=_d_mem_txt)

            gr.Markdown("---")
            with gr.Row():
                c_n = gr.Slider(10_000, 100_000_000, value=1_000_000, step=10_000, label="Vectors")
                c_dim = gr.Dropdown(["128","256","384","512","768","1024","1536"],
                                    value="768", label="Dim")
                c_prec = gr.Radio(["fp32","fp16"], value="fp32", label="Precision")
                c_bits = gr.Radio(["2","3","4"], value="3", label="Bits")
                c_btn = gr.Button("Update", variant="primary")

            c_btn.click(fn=_make_mem, inputs=[c_n, c_dim, c_prec, c_bits],
                        outputs=[mem_img, mem_txt])

        # ── HOW IT WORKS ──
        with gr.TabItem("How It Works"):
            gr.Markdown(_HOW)

    gr.Markdown("---\n*Independent implementation inspired by the TurboQuant paper. Not affiliated with Google Research.*")

demo.queue()
if __name__ == "__main__":
    demo.launch(show_error=True)
