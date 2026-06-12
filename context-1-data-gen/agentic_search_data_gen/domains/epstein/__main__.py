"""Unified Epstein pipeline: python -m agentic_search_data_gen.domains.epstein [args]

Runs index → explore → verify.
"""

import argparse
import os
import sys
import time

from .index import run_index
from .utils import init_utils
from .explore import EpsteinExplorerAgent
from .verify import EpsteinVerifier
from ...core.utils import get_anthropic_client


def _fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s" if m else f"{s}s"


def _print_header(name: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the full Epstein email data-generation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Individual stages are still accessible via their own entry points:\n"
               "  python -m agentic_search_data_gen.domains.epstein.index   --help\n"
               "  python -m agentic_search_data_gen.domains.epstein.explore --help\n"
               "  python -m agentic_search_data_gen.domains.epstein.verify  --help",
    )

    parser.add_argument("--input", "-i", default=None,
                        help="Path to epstein_only.json (default: download from Google Drive)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory")
    parser.add_argument("--collection", "-c", required=True,
                        help="ChromaDB collection name")
    parser.add_argument("--num", "-n", type=int, default=10,
                        help="Number of explorations (default: 10)")
    parser.add_argument("--max-workers", "-w", type=int, default=8,
                        help="Parallel workers (default: 8)")
    parser.add_argument("--max-iterations", type=int, default=20,
                        help="Max iterations per exploration (default: 20)")

    model_group = parser.add_argument_group("model options")
    model_group.add_argument("--explore-model", default="claude-sonnet-4-5",
                             help="Model for exploration (default: claude-sonnet-4-5)")
    model_group.add_argument("--verify-model", default="claude-opus-4-5",
                             help="Model for verification (default: claude-opus-4-5)")
    model_group.add_argument("--embedding-model", default="text-embedding-3-small",
                             help="OpenAI embedding model (default: text-embedding-3-small)")

    args = parser.parse_args()

    # --- Validate inputs ---
    if args.input and not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    import chromadb
    chroma_client = chromadb.CloudClient(
        api_key=os.getenv("CHROMA_API_KEY"),
        database=os.getenv("CHROMA_DATABASE"),
    )
    existing = [c.name for c in chroma_client.list_collections()]
    if args.collection in existing:
        print(f"Error: ChromaDB collection '{args.collection}' already exists. Use a unique collection name.")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print(f"Input: {args.input or '(download from Google Drive)'}")
    print(f"Output: {args.output}")
    print(f"Collection: {args.collection}")
    print(f"Explorations: {args.num}")
    print(f"Max workers: {args.max_workers}")
    print(f"Max iterations: {args.max_iterations}")

    pipeline_start = time.time()
    stages = []

    # --- Stage 1: Index ---
    _print_header("Stage 1: Index (download → chunk → index)")
    t0 = time.time()
    index_output_dir = os.path.join(args.output, "index_output")
    index_result = run_index(
        args.collection,
        output_dir=index_output_dir,
        input_json=args.input,
        embedding_model=args.embedding_model,
    )
    elapsed = time.time() - t0
    stages.append(("Index", elapsed))

    # --- Configure utils for explore/verify ---
    chunks_path = index_result["chunks_path"]
    init_utils(collection_name=args.collection, corpus_path=chunks_path)

    # --- Stage 2: Explore ---
    _print_header("Stage 2: Explore")
    t0 = time.time()
    agent = EpsteinExplorerAgent(
        model=args.explore_model,
        max_iterations=args.max_iterations,
    )
    explore_result = agent.run_batch(
        args.num,
        output_dir=args.output,
        max_workers=args.max_workers,
    )
    elapsed = time.time() - t0
    stages.append(("Explore", elapsed))

    print(f"  Total: {explore_result['total']}")
    print(f"  Successful: {explore_result['successful']}")
    print(f"  Failed: {explore_result['failed']}")

    # --- Stage 3: Verify ---
    _print_header("Stage 3: Verify")
    t0 = time.time()
    client = get_anthropic_client()
    verifier = EpsteinVerifier(client=client, model=args.verify_model)
    verify_result = verifier.run_batch(args.output, max_workers=args.max_workers)
    elapsed = time.time() - t0
    stages.append(("Verify", elapsed))

    print(f"  Processed: {verify_result['processed']}")
    print(f"  Passed verification: {verify_result['passed_verification']}")
    print(f"  Failed verification: {verify_result['failed_verification']}")

    # --- Summary ---
    total_elapsed = time.time() - pipeline_start
    print(f"\n{'=' * 60}")
    print("  Pipeline complete")
    print(f"{'=' * 60}")
    for name, elapsed in stages:
        print(f"  {name:30s} {_fmt_elapsed(elapsed)}")
    print(f"  {'─' * 40}")
    print(f"  {'Total':30s} {_fmt_elapsed(total_elapsed)}")


if __name__ == "__main__":
    main()
