"""Typer CLI for Stage 3."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer

from .config import effective_artifacts_root, ensure_stage3_run_dir
from .io import save_json, save_jsonl
from .nodes.candidate_merger import merge_candidate_pool_with_summary
from .nodes.balancer_slice_builder import run_balancer_slice_builder
from .nodes.hard_subset import run_hard_subset
from .nodes.qc_dedup_cluster import run_qc_dedup_cluster
from .nodes.split_builder_exporter import run_splits, write_stage3_split_outputs
from .pipeline.orchestrator import run_stage3_pipeline
from .policy.loader import Stage3ConfigError, load_stage3_policy
from .audit.packs import export_stage3_audit_packs as write_audit_packs
from .release.bundle import prepare_stage3_release_from_disk

app = typer.Typer(help="RuFP Bench Stage 3 — QC, dedup, balance, splits, export")


@app.command("run-stage3")
def run_stage3(
    stage3_run_id: str = typer.Option(
        ...,
        "--stage3-run-id",
        "--run-id",
        help="Stage 3 output folder: artifacts/stage3/<run_id>",
    ),
    stage2_run_id: Optional[str] = typer.Option(
        None,
        "--stage2-run-id",
        help="Override policy: use artifacts/stage2/<id> as Stage 2 input",
    ),
    stage25_run_id: Optional[str] = typer.Option(
        None,
        "--stage25-run-id",
        help="Override policy: use artifacts/stage25/<id> as Stage 2.5 input (optional)",
    ),
    config: Optional[str] = typer.Option(None, "--config", help="configs/stage3.yaml or RUFP_STAGE3_CONFIG"),
    resume: bool = typer.Option(False, "--resume", help="Skip completed steps when outputs exist"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Cap rows per source (see dry_run.max_rows_per_source)"),
    debug: bool = typer.Option(False, "--debug", help="Debug logging"),
):
    """Run full Stage 3 pipeline (merger → QC/dedup → balance → hard subset → splits/export → report)."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, stream=sys.stderr)
    try:
        asyncio.run(
            run_stage3_pipeline(
                stage3_run_id,
                config_path=config,
                resume=resume,
                dry_run=dry_run,
                stage2_run_id=stage2_run_id,
                stage25_run_id=stage25_run_id,
            )
        )
    except Stage3ConfigError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1)
    except RuntimeError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1)


@app.command("build-stage3-candidate-pool")
def build_stage3_candidate_pool(
    stage2_run_id: str = typer.Option(..., "--stage2-run-id", help="Stage 2 artifact run id"),
    stage3_run_id: str = typer.Option(..., "--stage3-run-id", help="Stage 3 output run id"),
    stage25_run_id: Optional[str] = typer.Option(
        None,
        "--stage25-run-id",
        help="Stage 2.5 artifact run id (optional if pool is Stage 2 only)",
    ),
    stage1_run_id: Optional[str] = typer.Option(
        None,
        "--stage1-run-id",
        help="Stage 1 run id for family_to_prompt_map under artifacts/stage1/<id>",
    ),
    config: Optional[str] = typer.Option(None, "--config", help="configs/stage3.yaml or RUFP_STAGE3_CONFIG"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Cap rows per source (dry_run.max_rows_per_source)"),
    debug: bool = typer.Option(False, "--debug", help="Debug logging"),
):
    """Build ``stage3_candidate_pool.jsonl`` (Node 1: candidate merger) from Stage 2 + Stage 2.5 runs."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, stream=sys.stderr)
    try:
        policy = load_stage3_policy(config)
    except Stage3ConfigError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1)

    root = Path(effective_artifacts_root())
    policy = policy.model_copy(deep=True)
    policy.inputs.stage2_dir = str(root / "artifacts" / "stage2" / stage2_run_id)
    policy.inputs.stage25_dir = (
        str(root / "artifacts" / "stage25" / stage25_run_id) if stage25_run_id else None
    )
    if stage1_run_id:
        policy.inputs.stage1_dir = str(root / "artifacts" / "stage1" / stage1_run_id)

    rows, summary = merge_candidate_pool_with_summary(policy, dry_run=dry_run)
    out_dir = ensure_stage3_run_dir(stage3_run_id)
    pool_path = out_dir / "stage3_candidate_pool.jsonl"
    sum_path = out_dir / "stage3_candidate_merge_summary.json"
    save_jsonl(str(pool_path), rows)
    save_json(str(sum_path), summary)
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))
    typer.echo(f"Wrote {pool_path} ({len(rows)} rows)", err=False)


@app.command("run-stage3-qc")
def run_stage3_qc(
    run_id: str = typer.Option(..., "--run-id", help="Stage 3 run id under artifacts/stage3/<run_id>"),
    config: Optional[str] = typer.Option(None, "--config", help="configs/stage3.yaml or RUFP_STAGE3_CONFIG"),
    debug: bool = typer.Option(False, "--debug", help="Debug logging"),
):
    """Run Node 2: QC + dedup + clustering → ``stage3_qc_labels.jsonl`` + ``stage3_dedup_clusters.jsonl``."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, stream=sys.stderr)
    try:
        policy = load_stage3_policy(config)
    except Stage3ConfigError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1)

    base = ensure_stage3_run_dir(run_id)
    pool_path = base / "stage3_candidate_pool.jsonl"
    if not pool_path.is_file():
        typer.echo(f"Missing candidate pool: {pool_path}", err=True)
        raise typer.Exit(code=1)

    from .io import load_jsonl, save_jsonl
    from .schemas import Stage3CandidateRecord

    pool = load_jsonl(str(pool_path), Stage3CandidateRecord)
    qc, clusters, _kept = run_qc_dedup_cluster(pool, policy)
    save_jsonl(str(base / "stage3_qc_labels.jsonl"), qc)
    save_jsonl(str(base / "stage3_dedup_clusters.jsonl"), clusters)
    typer.echo(
        json.dumps(
            {"qc_labels": len(qc), "clusters": len(clusters), "pool_survivors": len(_kept)},
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("run-stage3-balancer")
def run_stage3_balancer(
    run_id: str = typer.Option(..., "--run-id", help="Stage 3 run id under artifacts/stage3/<run_id>"),
    config: Optional[str] = typer.Option(None, "--config", help="configs/stage3.yaml or RUFP_STAGE3_CONFIG"),
    debug: bool = typer.Option(False, "--debug", help="Debug logging"),
):
    """Run Node 3: balancer + slice index → ``stage3_balanced_pool.jsonl`` + metadata + ``stage3_slice_index.json``."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, stream=sys.stderr)
    try:
        policy = load_stage3_policy(config)
    except Stage3ConfigError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1)

    base = ensure_stage3_run_dir(run_id)
    pool_path = base / "stage3_candidate_pool.jsonl"
    qc_path = base / "stage3_qc_labels.jsonl"
    if not pool_path.is_file():
        typer.echo(f"Missing candidate pool: {pool_path}", err=True)
        raise typer.Exit(code=1)
    if not qc_path.is_file():
        typer.echo(f"Missing QC labels (run Node 2 first): {qc_path}", err=True)
        raise typer.Exit(code=1)

    from .io import load_jsonl
    from .schemas import QCLabelRecord, Stage3CandidateRecord

    pool = load_jsonl(str(pool_path), Stage3CandidateRecord)
    qc = load_jsonl(str(qc_path), QCLabelRecord)
    qp = {q.item_id for q in qc if q.qc_pass}
    incoming = [r for r in pool if r.item_id in qp]
    rows, meta, slice_index = run_balancer_slice_builder(incoming, policy)
    save_jsonl(str(base / "stage3_balanced_pool.jsonl"), rows)
    save_jsonl(str(base / "stage3_balance_metadata.jsonl"), meta)
    save_json(str(base / "stage3_slice_index.json"), slice_index)
    typer.echo(
        json.dumps(
            {
                "balanced_rows": len(rows),
                "strata_rows": len(meta),
                "slice_names": sorted((slice_index.get("slices") or {}).keys()),
                "counts": slice_index.get("counts"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("run-stage3-hard-subset")
def run_stage3_hard_subset(
    run_id: str = typer.Option(..., "--run-id", help="Stage 3 run id under artifacts/stage3/<run_id>"),
    config: Optional[str] = typer.Option(None, "--config", help="configs/stage3.yaml or RUFP_STAGE3_CONFIG"),
    debug: bool = typer.Option(False, "--debug", help="Debug logging"),
):
    """Run Node 4: hard subset → ``stage3_hard_subset.jsonl``."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, stream=sys.stderr)
    try:
        policy = load_stage3_policy(config)
    except Stage3ConfigError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1)

    base = ensure_stage3_run_dir(run_id)
    bal_path = base / "stage3_balanced_pool.jsonl"
    if not bal_path.is_file():
        typer.echo(f"Missing balanced pool (run Node 3 first): {bal_path}", err=True)
        raise typer.Exit(code=1)

    from .io import load_jsonl, save_jsonl
    from .schemas import Stage3CandidateRecord

    rows = load_jsonl(str(bal_path), Stage3CandidateRecord)
    hard = run_hard_subset(rows, policy)
    out_path = base / "stage3_hard_subset.jsonl"
    save_jsonl(str(out_path), hard)
    typer.echo(json.dumps({"hard_subset_rows": len(hard), "path": str(out_path)}, ensure_ascii=False, indent=2))


@app.command("run-stage3-splits")
def run_stage3_splits(
    run_id: str = typer.Option(..., "--run-id", help="Stage 3 run id under artifacts/stage3/<run_id>"),
    config: Optional[str] = typer.Option(None, "--config", help="configs/stage3.yaml or RUFP_STAGE3_CONFIG"),
    debug: bool = typer.Option(False, "--debug", help="Debug logging"),
):
    """Run Node 5: splits + exports (dev/test/holdout JSONL, ``split_assignment.jsonl``, ``export_manifest.json``, ``stage3_exports/``)."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, stream=sys.stderr)
    try:
        policy = load_stage3_policy(config)
    except Stage3ConfigError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1)

    base = ensure_stage3_run_dir(run_id)
    bal_path = base / "stage3_balanced_pool.jsonl"
    dedup_path = base / "stage3_dedup_clusters.jsonl"
    hard_path = base / "stage3_hard_subset.jsonl"
    if not bal_path.is_file():
        typer.echo(f"Missing balanced pool: {bal_path}", err=True)
        raise typer.Exit(code=1)

    from .io import load_jsonl
    from .schemas import DedupClusterRecord, HardSubsetRecord, Stage3CandidateRecord

    balanced = load_jsonl(str(bal_path), Stage3CandidateRecord)
    dedup = load_jsonl(str(dedup_path), DedupClusterRecord) if dedup_path.is_file() else []
    hard_rows = load_jsonl(str(hard_path), HardSubsetRecord) if hard_path.is_file() else []

    splits = run_splits(balanced, policy, dedup)
    exports_dir = str(base / "stage3_exports")
    manifest = write_stage3_split_outputs(str(base), exports_dir, balanced, splits, hard_rows, policy)
    typer.echo(json.dumps({"splits": len(splits), "export_manifest": manifest.get("files", [])[:5]}, ensure_ascii=False, indent=2))


@app.command("prepare-stage3-release")
def prepare_stage3_release(
    run_id: str = typer.Option(..., "--run-id", help="Stage 3 run under artifacts/stage3/<run_id>"),
    config: Optional[str] = typer.Option(None, "--config", help="configs/stage3.yaml (for dataset card policy text)"),
    debug: bool = typer.Option(False, "--debug", help="Debug logging"),
):
    """Regenerate ``final_lineage.jsonl``, ``dataset_card.md``, ``release_checklist.md`` from existing run artifacts."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, stream=sys.stderr)
    try:
        paths = prepare_stage3_release_from_disk(run_id, config_path=config)
    except Stage3ConfigError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1)
    except (FileNotFoundError, OSError, ValueError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1)
    typer.echo(json.dumps(paths, ensure_ascii=False, indent=2))


@app.command("export-stage3-audit-packs")
def export_stage3_audit_packs_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Stage 3 run under artifacts/stage3/<run_id>"),
    config: Optional[str] = typer.Option(None, "--config", help="configs/stage3.yaml (audit_packs.* defaults)"),
    random_n: Optional[int] = typer.Option(None, "--random-n", help="Override audit_packs.random_count"),
    hard_n: Optional[int] = typer.Option(None, "--hard-n", help="Override audit_packs.hard_count"),
    repaired_n: Optional[int] = typer.Option(None, "--repaired-n", help="Override audit_packs.repaired_count"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Override audit_packs.random_seed"),
    random_split: Optional[str] = typer.Option(
        None,
        "--random-split",
        help="all | dev | test | review_holdout — limit random pack to this split",
    ),
    hard_order: Optional[str] = typer.Option(
        None,
        "--hard-order",
        help="score_desc | random — how to pick hard-pack rows",
    ),
    debug: bool = typer.Option(False, "--debug", help="Debug logging"),
):
    """Write small JSONL audit packs (random / hard / repaired) under stage3_exports/ for human review."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, stream=sys.stderr)
    try:
        policy = load_stage3_policy(config)
    except Stage3ConfigError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1)
    if random_split is not None and random_split not in ("all", "dev", "test", "review_holdout"):
        typer.echo("--random-split must be all|dev|test|review_holdout", err=True)
        raise typer.Exit(code=1)
    if hard_order is not None and hard_order not in ("score_desc", "random"):
        typer.echo("--hard-order must be score_desc|random", err=True)
        raise typer.Exit(code=1)
    try:
        out = write_audit_packs(
            run_id,
            policy,
            random_count=random_n,
            hard_count=hard_n,
            repaired_count=repaired_n,
            random_seed=seed,
            sample_random_from_split=random_split,
            hard_order=hard_order,
        )
    except FileNotFoundError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1)
    typer.echo(json.dumps(out, ensure_ascii=False, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
