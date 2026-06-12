"""
One-shot pipeline: generate → verify → validate_pairs → run_target → judge → export.

Reuses the same CLIs as the individual modules (via subprocess) so behavior stays identical.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_MODULE = "agentic_search_data_gen.domains.ru_fp_bench"

_DOM = Path(__file__).resolve().parent
_FULL = _DOM / "unsafe_topics_taxonomy_full.yaml"
_MINI = _DOM / "taxonomy.yaml"
_DEFAULT_TAXONOMY = str(_FULL if _FULL.exists() else _MINI)
_ENV_GENERATOR_BASE_URL = "RU_FP_GENERATOR_BASE_URL"
_ENV_GENERATOR_MODEL = "RU_FP_GENERATOR_MODEL"
_ENV_GENERATOR_API_KEY = "RU_FP_GENERATOR_API_KEY"
_ENV_TARGET_BASE_URL = "RU_FP_TARGET_BASE_URL"
_ENV_TARGET_MODEL = "RU_FP_TARGET_MODEL"
_ENV_TARGET_API_KEY = "RU_FP_TARGET_API_KEY"
_ENV_VALIDATOR_BASE_URL = "RU_FP_VALIDATOR_BASE_URL"
_ENV_VALIDATOR_MODEL = "RU_FP_VALIDATOR_MODEL"
_ENV_VALIDATOR_API_KEY = "RU_FP_VALIDATOR_API_KEY"

PAIRS_RAW = "pairs.raw.jsonl"
PAIRS_VERIFIED = "pairs.verified.jsonl"
PAIRS_VALIDATED = "pairs.validated.jsonl"
VALIDATION_REJECTED = "pairs.validation_rejected.jsonl"
TARGET_RUNS = "target_runs.jsonl"
JUDGED = "judged.jsonl"
FP_BENCHMARK = "fp_benchmark.jsonl"
VALIDATOR_RUNS = "validator_runs.jsonl"
ERRORS = "errors.jsonl"


def _run_step(description: str, argv: list[str]) -> None:
    cmd = [sys.executable, "-m", f"{_MODULE}.{argv[0]}", *argv[1:]]
    print(f"\n=== {description} ===\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full ru_fp_bench pipeline in one command (generate → verify → validate_pairs → run_target → judge → export).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Directory for all JSONL artifacts (created if missing).",
    )
    parser.add_argument("--taxonomy", default=_DEFAULT_TAXONOMY, help="Path to taxonomy YAML")
    parser.add_argument(
        "--topics",
        default=None,
        help="Comma-separated topic keys for generate (default: core-6)",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for generate.log and run_target.log. Overrides default <run-dir>/logs.",
    )
    parser.add_argument(
        "--no-file-logs",
        action="store_true",
        help="Do not write generate.log / run_target.log (overrides default log dir).",
    )
    parser.add_argument("--num", type=int, default=20, help="Number of pairs to generate")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock for both generate and run_target (no API calls).",
    )

    # generate (real)
    g = parser.add_argument_group("generator (ignored when --mock)")
    g.add_argument("--generator-base-url", default=None, help=f"or env {_ENV_GENERATOR_BASE_URL}")
    g.add_argument("--generator-model", default=None, help=f"or env {_ENV_GENERATOR_MODEL}")
    g.add_argument("--generator-api-key", default=None, help=f"or env {_ENV_GENERATOR_API_KEY}")
    g.add_argument("--max-retries-generate", type=int, default=3, help="Retries per row in real generate")

    # validate_pairs (ignored when --mock)
    v = parser.add_argument_group("validator / gpt-oss (ignored when --mock)")
    v.add_argument("--validator-base-url", default=None, help=f"or env {_ENV_VALIDATOR_BASE_URL}")
    v.add_argument("--validator-model", default=None, help=f"or env {_ENV_VALIDATOR_MODEL}")
    v.add_argument("--validator-api-key", default=None, help=f"or env {_ENV_VALIDATOR_API_KEY}")
    v.add_argument("--max-retries-validate", type=int, default=3, help="Retries per row in real validate_pairs")
    v.add_argument(
        "--validate-resume",
        action="store_true",
        help="Pass --resume to validate_pairs (reuse validator_runs, append new ids only).",
    )
    v.add_argument(
        "--max-workers-validate",
        type=int,
        default=1,
        help="Thread pool for validate_pairs (default 1).",
    )

    # run_target
    t = parser.add_argument_group("target (ignored when --mock for API; --model still stored in output for mock)")
    t.add_argument(
        "--model",
        default="mock-target-model",
        help="Model id in target_runs metadata; real run_target uses RU_FP_TARGET_MODEL if this stays default",
    )
    t.add_argument("--target-base-url", default=None, help=f"or env {_ENV_TARGET_BASE_URL}")
    t.add_argument("--target-model", default=None, help=f"or env {_ENV_TARGET_MODEL}")
    t.add_argument("--target-api-key", default=None, help=f"or env {_ENV_TARGET_API_KEY}")
    t.add_argument("--max-retries-target", type=int, default=3, help="Retries per request in real run_target")

    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.no_file_logs:
        file_log_dir = None
    elif args.log_dir is not None:
        file_log_dir = str(Path(args.log_dir).resolve())
    else:
        file_log_dir = str(run_dir / "logs")

    p_raw = str(run_dir / PAIRS_RAW)
    p_ver = str(run_dir / PAIRS_VERIFIED)
    p_val = str(run_dir / PAIRS_VALIDATED)
    p_val_rej = str(run_dir / VALIDATION_REJECTED)
    p_tgt = str(run_dir / TARGET_RUNS)
    p_jdg = str(run_dir / JUDGED)
    p_fp = str(run_dir / FP_BENCHMARK)

    # --- generate ---
    gen_cmd: list[str] = [
        "generate",
        "--taxonomy",
        args.taxonomy,
        "--output",
        p_raw,
        "--num",
        str(args.num),
    ]
    if args.topics:
        gen_cmd.extend(["--topics", args.topics])
    if file_log_dir:
        gen_cmd.extend(["--log-dir", file_log_dir])
    if args.mock:
        gen_cmd.append("--mock")
    else:
        if args.generator_base_url:
            gen_cmd.extend(["--generator-base-url", args.generator_base_url])
        if args.generator_model:
            gen_cmd.extend(["--generator-model", args.generator_model])
        if args.generator_api_key:
            gen_cmd.extend(["--generator-api-key", args.generator_api_key])
        gen_cmd.extend(["--max-retries", str(args.max_retries_generate)])
    _run_step("1/6 generate", gen_cmd)

    # --- verify ---
    _run_step("2/6 verify", ["verify", "--input", p_raw, "--output", p_ver])

    # --- validate_pairs ---
    val_cmd: list[str] = [
        "validate_pairs",
        "--input",
        p_ver,
        "--output",
        p_val,
        "--rejected-log",
        p_val_rej,
    ]
    if file_log_dir:
        val_cmd.extend(["--log-dir", file_log_dir])
    if args.mock:
        val_cmd.append("--mock")
    else:
        g_resolved = (args.generator_model or os.environ.get(_ENV_GENERATOR_MODEL) or "").strip()
        if g_resolved:
            val_cmd.extend(["--generator-model", g_resolved])
        if args.validator_base_url:
            val_cmd.extend(["--validator-base-url", args.validator_base_url])
        if args.validator_model:
            val_cmd.extend(["--validator-model", args.validator_model])
        if args.validator_api_key:
            val_cmd.extend(["--validator-api-key", args.validator_api_key])
        val_cmd.extend(["--max-retries", str(args.max_retries_validate)])
    if args.validate_resume:
        val_cmd.append("--resume")
    if args.max_workers_validate != 1:
        val_cmd.extend(["--max-workers", str(args.max_workers_validate)])
    _run_step("3/6 validate_pairs", val_cmd)

    # --- run_target ---
    rt_cmd: list[str] = [
        "run_target",
        "--input",
        p_val,
        "--output",
        p_tgt,
        "--model",
        args.model,
    ]
    if file_log_dir:
        rt_cmd.extend(["--log-dir", file_log_dir])
    if args.mock:
        rt_cmd.append("--mock")
    else:
        if args.target_base_url:
            rt_cmd.extend(["--target-base-url", args.target_base_url])
        if args.target_model:
            rt_cmd.extend(["--target-model", args.target_model])
        if args.target_api_key:
            rt_cmd.extend(["--target-api-key", args.target_api_key])
        rt_cmd.extend(["--max-retries", str(args.max_retries_target)])
    _run_step("4/6 run_target", rt_cmd)

    # --- judge ---
    _run_step("5/6 judge", ["judge", "--input", p_tgt, "--output", p_jdg])

    # --- export ---
    _run_step("6/6 export", ["export", "--input", p_jdg, "--output", p_fp])

    print(f"\nDone. Artifacts in: {run_dir}", flush=True)
    for name in (
        PAIRS_RAW,
        PAIRS_VERIFIED,
        PAIRS_VALIDATED,
        VALIDATION_REJECTED,
        VALIDATOR_RUNS,
        ERRORS,
        TARGET_RUNS,
        JUDGED,
        FP_BENCHMARK,
    ):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
