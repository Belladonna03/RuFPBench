"""Unified SEC pipeline: python -m agentic_search_data_gen.domains.sec [args]

Runs index -> assign truth types -> explore -> verify -> collect ->
verify-collect, then repeats extend -> verify-extension for N rounds.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from .index import CorpusProcessor
from .explore import SecExplorerAgent
from .verify import run_batch as verify_run_batch, run_collect_batch as verify_run_collect_batch
from .collect import ChunkCollectorAgent
from .extend import SECBridgingAgent
from .prompts import TRUTH_TYPES


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


def _print_index_result(successful: list, failed: list) -> None:
    print(f"  successful: {len(successful)}")
    print(f"  failed: {len(failed)}")
    for item in failed:
        print(f"  ERROR {item['ticker']}: {item['error']}")


def _assign_truth_types(output_dir: str) -> dict:
    """Assign random truth_type to JSON files that are missing one."""
    input_path = Path(output_dir)
    json_files = list(input_path.glob("*.json"))

    assigned = 0
    skipped = 0

    for filepath in json_files:
        with open(filepath, "r") as f:
            data = json.load(f)

        if "truth_type" in data:
            skipped += 1
            continue

        data["truth_type"] = random.choice(TRUTH_TYPES)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
        assigned += 1

    return {"total": len(json_files), "assigned": assigned, "skipped": skipped}


def main():
    module_dir = Path(__file__).parent
    default_seeds = module_dir / "seeds.txt"

    parser = argparse.ArgumentParser(
        description="Run the full SEC data-generation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Individual stages are still accessible via their own entry points:\n"
               "  python -m agentic_search_data_gen.domains.sec.index   --help\n"
               "  python -m agentic_search_data_gen.domains.sec.explore --help\n"
               "  python -m agentic_search_data_gen.domains.sec.verify  --help\n"
               "  python -m agentic_search_data_gen.domains.sec.collect --help\n"
               "  python -m agentic_search_data_gen.domains.sec.extend  --help",
    )

    parser.add_argument("--seeds", "-s", default=str(default_seeds),
                        help=f"Path to seeds file (default: {default_seeds})")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--collection", "-c", required=True, help="ChromaDB collection name")
    parser.add_argument("--max-workers", "-w", type=int, default=8, help="Parallel workers (default: 8)")
    parser.add_argument("--extension-rounds", type=int, default=1,
                        help="Number of extend rounds (default: 1)")
    parser.add_argument("--no-index", action="store_true", help="Skip ChromaDB indexing stage")
    parser.add_argument("--identity", required=True,
                        help="SEC EDGAR identity string (e.g. 'Name email@example.com')")

    explore_g = parser.add_argument_group("explore")
    explore_g.add_argument("--explore-model", default="claude-sonnet-4-5",
                           help="Model for explore (default: claude-sonnet-4-5)")
    explore_g.add_argument("--explore-max-iterations", type=int, default=20,
                           help="Max iterations for explore (default: 20)")

    verify_g = parser.add_argument_group("verify")
    verify_g.add_argument("--verify-model", default="claude-opus-4-5",
                          help="Model for verify (default: claude-opus-4-5)")
    verify_g.add_argument("--verify-max-retries", type=int, default=3,
                          help="Max retries for verify (default: 3)")

    collect_g = parser.add_argument_group("collect")
    collect_g.add_argument("--collect-model", default="claude-sonnet-4-5",
                           help="Model for collect (default: claude-sonnet-4-5)")
    collect_g.add_argument("--collect-max-iterations", type=int, default=15,
                           help="Max iterations for collect (default: 15)")

    extend_g = parser.add_argument_group("extend")
    extend_g.add_argument("--extend-agent-model", default="claude-sonnet-4-5",
                          help="Model for extend agent (default: claude-sonnet-4-5)")
    extend_g.add_argument("--extend-verification-model", default="claude-opus-4-5",
                          help="Model for extend verification (default: claude-opus-4-5)")
    extend_g.add_argument("--extend-max-iterations-phase1", type=int, default=10,
                          help="Max iterations for extend phase 1 (default: 10)")
    extend_g.add_argument("--extend-max-iterations-phase2", type=int, default=15,
                          help="Max iterations for extend phase 2 (default: 15)")

    args = parser.parse_args()

    # --- Validate seeds file ---
    if not os.path.exists(args.seeds):
        print(f"Error: Seeds file not found: {args.seeds}")
        sys.exit(1)

    with open(args.seeds) as f:
        tickers = [line.strip() for line in f if line.strip()]

    if not tickers:
        print(f"Error: No tickers found in {args.seeds}")
        sys.exit(1)

    # --- Validate ChromaDB collection doesn't already exist ---
    if not args.no_index:
        import chromadb
        chroma_client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            database=os.getenv("CHROMA_DATABASE"),
        )
        existing = [c.name for c in chroma_client.list_collections()]
        if args.collection in existing:
            print(f"Error: ChromaDB collection '{args.collection}' already exists. Use a unique collection name.")
            sys.exit(1)

    # --- Filter out already-indexed tickers ---
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_tickers = set()
    tickers_to_index = []
    for ticker in tickers:
        json_path = output_dir / f"{ticker}.json"
        if json_path.exists():
            existing_tickers.add(ticker)
        else:
            tickers_to_index.append(ticker)

    print(f"Seeds: {len(tickers)} from {args.seeds}")
    print(f"Output: {args.output}")
    print(f"Collection: {args.collection}")
    print(f"Max workers: {args.max_workers}")
    print(f"Extension rounds: {args.extension_rounds}")
    print(f"Already indexed: {len(existing_tickers)}")
    print(f"To index: {len(tickers_to_index)}")

    pipeline_start = time.time()
    stages = []

    # --- Stage 1: Index ---
    if not args.no_index:
        _print_header("Stage 1: Index")
        t0 = time.time()

        if tickers_to_index:
            processor = CorpusProcessor(
                collection_name=args.collection,
                identity_str=args.identity
            )
            summary = processor.process_batch(
                tickers=tickers_to_index,
                output_dir=args.output,
                index=True,
                max_workers=args.max_workers
            )
            _print_index_result(summary["successful"], summary["failed"])
        else:
            print("  All tickers already indexed, skipping.")

        elapsed = time.time() - t0
        stages.append(("Index", elapsed))
    else:
        print("\nSkipping index stage (--no-index)")

    # --- Stage 2: Assign truth types ---
    _print_header("Stage 2: Assign truth types")
    t0 = time.time()
    result = _assign_truth_types(args.output)
    elapsed = time.time() - t0
    _print_result(result)
    stages.append(("Assign truth types", elapsed))

    # --- Stage 3: Explore ---
    _print_header("Stage 3: Explore")
    t0 = time.time()
    explorer = SecExplorerAgent(
        collection_name=args.collection,
        max_iterations=args.explore_max_iterations,
        model=args.explore_model
    )
    result = explorer.run_batch(args.output, max_workers=args.max_workers)
    elapsed = time.time() - t0
    _print_result(result)
    stages.append(("Explore", elapsed))

    # --- Stage 4: Verify (main) ---
    _print_header("Stage 4: Verify")
    t0 = time.time()
    result = verify_run_batch(
        input_dir=args.output,
        model=args.verify_model,
        max_workers=args.max_workers,
        max_retries=args.verify_max_retries
    )
    elapsed = time.time() - t0
    _print_result(result)
    stages.append(("Verify", elapsed))

    # --- Stage 5: Collect ---
    _print_header("Stage 5: Collect")
    t0 = time.time()
    collector = ChunkCollectorAgent(
        collection_name=args.collection,
        max_iterations=args.collect_max_iterations,
        model=args.collect_model
    )
    result = collector.run_batch(args.output, max_workers=args.max_workers)
    elapsed = time.time() - t0
    _print_result(result)
    stages.append(("Collect", elapsed))

    # --- Stage 6: Verify (collect) ---
    _print_header("Stage 6: Verify (collect)")
    t0 = time.time()
    result = verify_run_collect_batch(
        input_dir=args.output,
        model=args.verify_model,
        max_workers=args.max_workers,
        max_retries=args.verify_max_retries
    )
    elapsed = time.time() - t0
    _print_result(result)
    stages.append(("Verify (collect)", elapsed))

    # --- Extension rounds ---
    for round_num in range(1, args.extension_rounds + 1):
        stage_offset = 6 + (round_num - 1) * 2

        _print_header(f"Stage {stage_offset + 1}: Extend (round {round_num})")
        t0 = time.time()
        bridging_agent = SECBridgingAgent(
            collection_name=args.collection,
            max_iterations_phase1=args.extend_max_iterations_phase1,
            max_iterations_phase2=args.extend_max_iterations_phase2,
            agent_model=args.extend_agent_model,
            verification_model=args.extend_verification_model
        )
        result = bridging_agent.run_batch(args.output, max_workers=args.max_workers)
        elapsed = time.time() - t0
        _print_result(result)
        stages.append((f"Extend r{round_num}", elapsed))

        _print_header(f"Stage {stage_offset + 2}: Verify extension (round {round_num})")
        t0 = time.time()
        result = verify_run_batch(
            input_dir=args.output,
            model=args.verify_model,
            max_workers=args.max_workers,
            max_retries=args.verify_max_retries
        )
        elapsed = time.time() - t0
        _print_result(result)
        stages.append((f"Verify ext r{round_num}", elapsed))

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
