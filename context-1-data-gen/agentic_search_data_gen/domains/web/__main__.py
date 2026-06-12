"""Unified web pipeline: python -m agentic_search_data_gen.domains.web [args]

Runs explore → verify → distract → verify-distractors, then optionally
repeats extend → verify → distract → verify-distractors for N rounds.
"""

import argparse
import os
import sys
import time

from ...core.utils import get_anthropic_client
from .explore import WebExplorerAgent
from .verify import WebVerifier
from .distract import WebDistractorAgent
from .extend import WebExtensionAgent
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


def _print_result(result: dict) -> None:
    for k, v in result.items():
        if k == "errors":
            continue
        print(f"  {k}: {v}")
    for err in result.get("errors", []):
        seed_or_file = err.get("seed") or err.get("file", "?")
        print(f"  ERROR {seed_or_file}: {err['error']}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the full web data-generation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Individual stages are still accessible via their own entry points:\n"
               "  python -m agentic_search_data_gen.domains.web.explore --help\n"
               "  python -m agentic_search_data_gen.domains.web.verify  --help\n"
               "  python -m agentic_search_data_gen.domains.web.distract --help\n"
               "  python -m agentic_search_data_gen.domains.web.extend  --help\n"
               "  python -m agentic_search_data_gen.domains.web.index   --help",
    )

    parser.add_argument("--seeds", "-s", required=True, help="Path to seeds file (one topic per line)")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--collection", "-c", required=True, help="ChromaDB collection name for indexing")
    parser.add_argument("--extension-rounds", type=int, default=0, help="Number of extend rounds (default: 0)")
    parser.add_argument("--max-workers", "-w", type=int, default=8, help="Parallel workers (default: 8)")

    explore_g = parser.add_argument_group("explore")
    explore_g.add_argument("--explore-model", default="claude-sonnet-4-5", help="Model for explore (default: claude-sonnet-4-5)")
    explore_g.add_argument("--explore-max-iterations", type=int, default=20, help="Max iterations for explore (default: 20)")

    verify_g = parser.add_argument_group("verify")
    verify_g.add_argument("--verify-model", default="claude-opus-4-5", help="Model for verify (default: claude-opus-4-5)")
    verify_g.add_argument("--verify-max-retries", type=int, default=3, help="Max retries for verify (default: 3)")

    distract_g = parser.add_argument_group("distract")
    distract_g.add_argument("--distract-model", default="claude-sonnet-4-5", help="Model for distract (default: claude-sonnet-4-5)")
    distract_g.add_argument("--distract-max-iterations", type=int, default=15, help="Max iterations for distract (default: 15)")

    extend_g = parser.add_argument_group("extend")
    extend_g.add_argument("--extend-model", default="claude-sonnet-4-5", help="Model for extend (default: claude-sonnet-4-5)")
    extend_g.add_argument("--extend-max-iterations", type=int, default=20, help="Max iterations for extend (default: 20)")

    args = parser.parse_args()

    # Validate seeds file
    if not os.path.exists(args.seeds):
        print(f"Error: Seeds file not found: {args.seeds}")
        sys.exit(1)

    with open(args.seeds) as f:
        seeds = [line.strip() for line in f if line.strip()]

    if not seeds:
        print(f"Error: No seeds found in {args.seeds}")
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
    print(f"Extension rounds: {args.extension_rounds}")
    print(f"Max workers: {args.max_workers}")

    pipeline_start = time.time()
    stages = []

    # --- Stage 1: Explore ---
    _print_header("Stage 1: Explore")
    t0 = time.time()
    explorer = WebExplorerAgent(model=args.explore_model, max_iterations=args.explore_max_iterations)
    result = explorer.run_batch(seeds, output_dir=args.output, max_workers=args.max_workers)
    elapsed = time.time() - t0
    _print_result(result)
    stages.append(("Explore", elapsed, result))

    # --- Stage 2: Verify supporting items ---
    _print_header("Stage 2: Verify supporting items")
    t0 = time.time()
    client = get_anthropic_client()
    verifier = WebVerifier(client=client, model=args.verify_model, max_retries=args.verify_max_retries)
    result = verifier.run_batch(args.output, max_workers=args.max_workers)
    elapsed = time.time() - t0
    _print_result(result)
    stages.append(("Verify", elapsed, result))

    # --- Stage 3: Mine distractors ---
    _print_header("Stage 3: Mine distractors")
    t0 = time.time()
    distractor = WebDistractorAgent(model=args.distract_model, max_iterations=args.distract_max_iterations)
    result = distractor.run_batch(args.output, level_filter=None, max_workers=args.max_workers)
    elapsed = time.time() - t0
    _print_result(result)
    stages.append(("Distract", elapsed, result))

    # --- Stage 4: Verify distractors ---
    _print_header("Stage 4: Verify distractors")
    t0 = time.time()
    result = verifier.run_distractor_verification_batch(args.output, max_workers=args.max_workers)
    elapsed = time.time() - t0
    _print_result(result)
    stages.append(("Verify distractors", elapsed, result))

    # --- Extension rounds ---
    for round_num in range(1, args.extension_rounds + 1):
        stage_offset = 4 + (round_num - 1) * 4

        _print_header(f"Stage {stage_offset + 1}: Extend (round {round_num})")
        t0 = time.time()
        extender = WebExtensionAgent(model=args.extend_model, max_iterations=args.extend_max_iterations)
        result = extender.run_batch(args.output, max_workers=args.max_workers)
        elapsed = time.time() - t0
        _print_result(result)
        stages.append((f"Extend r{round_num}", elapsed, result))

        _print_header(f"Stage {stage_offset + 2}: Verify (round {round_num})")
        t0 = time.time()
        result = verifier.run_batch(args.output, max_workers=args.max_workers)
        elapsed = time.time() - t0
        _print_result(result)
        stages.append((f"Verify r{round_num}", elapsed, result))

        _print_header(f"Stage {stage_offset + 3}: Distract (round {round_num})")
        t0 = time.time()
        distractor_r = WebDistractorAgent(model=args.distract_model, max_iterations=args.distract_max_iterations)
        result = distractor_r.run_batch(args.output, level_filter=round_num, max_workers=args.max_workers)
        elapsed = time.time() - t0
        _print_result(result)
        stages.append((f"Distract r{round_num}", elapsed, result))

        _print_header(f"Stage {stage_offset + 4}: Verify distractors (round {round_num})")
        t0 = time.time()
        result = verifier.run_distractor_verification_batch(args.output, max_workers=args.max_workers)
        elapsed = time.time() - t0
        _print_result(result)
        stages.append((f"Verify distractors r{round_num}", elapsed, result))

    # --- Index ---
    stage_num = 5 + args.extension_rounds * 4
    _print_header(f"Stage {stage_num}: Index")
    t0 = time.time()
    index_result = run_index(args.output, args.collection)
    elapsed = time.time() - t0
    _print_result(index_result)
    stages.append(("Index", elapsed, index_result))

    # --- Summary ---
    total_elapsed = time.time() - pipeline_start
    print(f"\n{'=' * 60}")
    print("  Pipeline complete")
    print(f"{'=' * 60}")
    for name, elapsed, _ in stages:
        print(f"  {name:30s} {_fmt_elapsed(elapsed)}")
    print(f"  {'─' * 40}")
    print(f"  {'Total':30s} {_fmt_elapsed(total_elapsed)}")


if __name__ == "__main__":
    main()
