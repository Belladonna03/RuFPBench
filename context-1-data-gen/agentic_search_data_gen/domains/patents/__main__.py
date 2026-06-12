"""Unified patents pipeline: python -m agentic_search_data_gen.domains.patents [args]

Runs process → extract → generate → index.
"""

import argparse
import asyncio
import os
import sys
import time

from ...core.utils import get_anthropic_client
from .process import PatentDataProcessor, read_application_numbers
from .extract import Extractor
from .generate import EvalGenNew
from .index import run_index


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
        description="Run the full patents data-generation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Individual stages are still accessible via their own entry points:\n"
               "  python -m agentic_search_data_gen.domains.patents.process  --help\n"
               "  python -m agentic_search_data_gen.domains.patents.extract  --help\n"
               "  python -m agentic_search_data_gen.domains.patents.generate --help\n"
               "  python -m agentic_search_data_gen.domains.patents.index    --help",
    )

    parser.add_argument("--seeds", "-s", required=True,
                        help="Path to seeds file (one application number per line)")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--collection", "-c", required=True,
                        help="ChromaDB collection name for indexing")
    parser.add_argument("--concurrent", type=int, default=10,
                        help="Max concurrent requests for process stage (default: 10)")
    parser.add_argument("--max-workers", "-w", type=int, default=4,
                        help="Parallel workers for extract/generate stages (default: 4)")

    model_group = parser.add_argument_group("model options")
    model_group.add_argument("--extract-model", default="claude-opus-4-5",
                             help="Anthropic model for extraction (default: claude-opus-4-5)")
    model_group.add_argument("--generate-model", default="claude-opus-4-5",
                             help="Anthropic model for generation (default: claude-opus-4-5)")
    model_group.add_argument("--embedding-model", default="text-embedding-3-small",
                             help="OpenAI embedding model for indexing (default: text-embedding-3-small)")

    args = parser.parse_args()

    # Validate seeds file
    if not os.path.exists(args.seeds):
        print(f"Error: Seeds file not found: {args.seeds}")
        sys.exit(1)

    seeds = read_application_numbers(args.seeds)

    if not seeds:
        print(f"Error: No application numbers found in {args.seeds}")
        sys.exit(1)

    # Validate ChromaDB collection doesn't already exist
    import chromadb
    chroma_client = chromadb.CloudClient(
        api_key=os.getenv("CHROMA_API_KEY"),
        database=os.getenv("CHROMA_DATABASE"),
    )
    existing = [c.name for c in chroma_client.list_collections()]
    if args.collection in existing:
        print(f"Error: ChromaDB collection '{args.collection}' already exists. Use a unique collection name.")
        sys.exit(1)

    print(f"Seeds: {len(seeds)} from {args.seeds}")
    print(f"Output: {args.output}")
    print(f"Collection: {args.collection}")
    print(f"Concurrent (process): {args.concurrent}")
    print(f"Max workers (extract/generate): {args.max_workers}")

    pipeline_start = time.time()
    stages = []

    # --- Stage 1: Process ---
    _print_header("Stage 1: Process")
    t0 = time.time()

    async def _run_process():
        processor = PatentDataProcessor()
        try:
            await processor.process_multiple_patents(
                seeds, output_dir=args.output, max_concurrent=args.concurrent
            )
        finally:
            await processor.close()

    asyncio.run(_run_process())
    elapsed = time.time() - t0
    stages.append(("Process", elapsed))

    # --- Stage 2: Extract ---
    _print_header("Stage 2: Extract")
    t0 = time.time()
    anthropic_client = get_anthropic_client()
    extractor = Extractor(anthropic_client, model=args.extract_model)
    extractor.run_batch(args.output, max_workers=args.max_workers)
    elapsed = time.time() - t0
    stages.append(("Extract", elapsed))

    # --- Stage 3: Generate ---
    _print_header("Stage 3: Generate")
    t0 = time.time()
    eval_gen = EvalGenNew(anthropic_client, model=args.generate_model)
    eval_gen.run_batch(args.output, max_workers=args.max_workers)
    elapsed = time.time() - t0
    stages.append(("Generate", elapsed))

    # --- Stage 4: Index ---
    _print_header("Stage 4: Index")
    t0 = time.time()
    index_result = run_index(args.output, args.collection, embedding_model=args.embedding_model)
    elapsed = time.time() - t0
    stages.append(("Index", elapsed))

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
