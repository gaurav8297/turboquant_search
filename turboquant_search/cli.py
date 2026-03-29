"""
CLI entry point for TurboQuant Search.

Usage:
    tqs demo                              # Launch Gradio comparison dashboard
    tqs demo --dataset arxiv              # Use arxiv embeddings
    tqs benchmark --dataset sift-1m       # Run benchmarks, print table
    tqs index --input vectors.npy --bits 3  # Index custom embeddings
    tqs search --index tq.index --query "text" --top 10
"""

import sys
import click
import numpy as np


@click.group()
@click.version_option(package_name="turboquant-search")
def cli():
    """TurboQuant Search — vector compression for similarity search."""
    pass


@cli.command()
@click.option(
    "--dataset", "-d",
    default="wikipedia-384",
    type=click.Choice([
        "wikipedia-384", "wikipedia-768", "wikipedia-1536",
        "arxiv-384", "arxiv-1024",
        "wikipedia", "arxiv",  # aliases for 384-dim
    ]),
    help="Pre-embedded dataset (name-dim). E.g. wikipedia-768, arxiv-1024.",
)
@click.option("--port", "-p", default=7860, help="Gradio server port.")
@click.option("--share", is_flag=True, help="Create a public Gradio link.")
def demo(dataset, port, share):
    """Launch the interactive comparison dashboard."""
    click.echo(f"Loading {dataset} dataset...")

    from .dataset_hub import load_dataset
    vectors, queries, texts, info = load_dataset(dataset)

    click.echo(
        f"Loaded {info['count']:,} vectors (dim={info['dim']}, model={info['model']})"
    )
    click.echo("Starting Gradio dashboard...")

    # Import and launch the app with the loaded data
    from . import _app_launcher
    _app_launcher.launch_demo(
        vectors=vectors,
        queries=queries,
        texts=texts,
        info=info,
        port=port,
        share=share,
    )


@cli.command()
@click.option(
    "--dataset", "-d",
    default="synthetic",
    type=click.Choice(["synthetic", "sift-128", "glove-100", "sift-1m"]),
    help="Dataset for benchmarks.",
)
@click.option("--n-vectors", "-n", default=10000, help="Number of vectors (synthetic only).")
@click.option("--bits", "-b", multiple=True, type=int, default=(2, 3, 4), help="Bit widths to test.")
def benchmark(dataset, n_vectors, bits):
    """Run benchmarks and print results table."""
    from .benchmarks import run_benchmark, format_results_table

    bits = list(bits)
    click.echo(f"Running benchmark: {dataset} ({n_vectors:,} vectors, bits={bits})")
    click.echo()

    def progress_cb(step, total, msg):
        click.echo(f"  [{step}/{total}] {msg}")

    results = run_benchmark(
        dataset_name=dataset,
        n_vectors=n_vectors,
        n_queries=200,
        k_values=[1, 5, 10, 50],
        bit_widths=bits,
        progress_callback=progress_cb,
    )

    click.echo()
    click.echo(format_results_table(results))


@cli.command("index")
@click.option("--input", "-i", "input_path", required=True, help="Path to .npy embeddings file.")
@click.option("--bits", "-b", default=3, type=click.IntRange(2, 4), help="Quantization bits (2-4).")
@click.option("--output", "-o", default=None, help="Output index path (default: <input>.tqindex).")
def index_cmd(input_path, bits, output):
    """Index custom numpy embeddings."""
    import pickle

    click.echo(f"Loading vectors from {input_path}...")
    vectors = np.load(input_path).astype(np.float32)
    n, dim = vectors.shape
    click.echo(f"  {n:,} vectors, dim={dim}")

    from .core import TurboQuantSearchIndex

    click.echo(f"Building {bits}-bit index...")
    idx = TurboQuantSearchIndex(dim=dim, bits=bits)
    idx.add(vectors)

    stats = idx.stats()
    click.echo(f"  Compression: {stats['compression_ratio']}")
    click.echo(f"  Memory: {stats['memory_mb']:.2f} MB")
    click.echo(f"  Build time: {idx.build_time:.3f}s")

    if output is None:
        output = input_path.rsplit(".", 1)[0] + ".tqindex"

    with open(output, "wb") as f:
        pickle.dump({"index": idx, "dim": dim, "bits": bits, "n_vectors": n}, f)
    click.echo(f"Saved index to {output}")


@cli.command("search")
@click.option("--index", "-i", "index_path", required=True, help="Path to .tqindex file.")
@click.option("--query", "-q", required=True, help="Path to query .npy file (one or more vectors).")
@click.option("--top", "-k", default=10, help="Number of results to return.")
def search_cmd(index_path, query, top):
    """Search an existing index."""
    import pickle

    click.echo(f"Loading index from {index_path}...")
    with open(index_path, "rb") as f:
        data = pickle.load(f)

    idx = data["index"]
    click.echo(f"  {data['n_vectors']:,} vectors, dim={data['dim']}, {data['bits']}-bit")

    query_vectors = np.load(query).astype(np.float32)
    if query_vectors.ndim == 1:
        query_vectors = query_vectors.reshape(1, -1)

    click.echo(f"Searching for top-{top} results ({query_vectors.shape[0]} queries)...")
    scores, indices = idx.search(query_vectors, k=top)

    for i in range(query_vectors.shape[0]):
        click.echo(f"\nQuery {i}:")
        for rank, (score, idx_val) in enumerate(zip(scores[i], indices[i])):
            click.echo(f"  {rank+1:>3}. index={idx_val:<8} score={score:.4f}")


def main():
    cli()


if __name__ == "__main__":
    main()
