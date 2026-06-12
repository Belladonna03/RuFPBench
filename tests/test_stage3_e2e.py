"""End-to-end smoke: Stage 3 orchestrator writes report, manifest, and metrics bundle."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "configs" / "stage3.yaml"


@pytest.mark.skipif(not CFG.is_file(), reason="configs/stage3.yaml missing")
def test_stage3_e2e_orchestrator_outputs(monkeypatch):
    monkeypatch.chdir(REPO)
    rid = f"s3_e2e_{uuid.uuid4().hex[:8]}"
    from rufp_stage3.pipeline.orchestrator import run_stage3_pipeline

    summary = asyncio.run(
        run_stage3_pipeline(
            rid,
            config_path=str(CFG),
            dry_run=True,
            resume=False,
        )
    )
    assert summary.pool_size >= 0
    base = REPO / "artifacts" / "stage3" / rid
    assert (base / "stage3_summary.json").is_file()
    assert (base / "stage3_final_report.md").is_file()
    assert (base / "stage3_run_manifest.json").is_file()

    ex = base / "stage3_exports"
    assert (ex / "final_lineage.jsonl").is_file()
    assert (ex / "dataset_card.md").is_file()
    assert (ex / "release_checklist.md").is_file()

    mdir = base / "metrics"
    for name in (
        "category_distribution.json",
        "subtype_distribution.json",
        "source_stage_distribution.json",
        "probe_profile_summary.json",
        "hard_subset_stats.json",
        "split_stats.json",
    ):
        p = mdir / name
        assert p.is_file(), f"missing {p}"
        json.loads(p.read_text(encoding="utf-8"))

    man = json.loads((base / "stage3_run_manifest.json").read_text(encoding="utf-8"))
    assert man.get("started_at")
    assert man.get("finished_at")
    nm = man.get("node_metrics") or {}
    for node in (
        "build_pool",
        "qc_dedup_cluster",
        "balance",
        "hard_subset",
        "split_builder_exporter",
    ):
        assert node in nm, f"missing metrics for {node}"
        assert nm[node].get("status") == "ok"
        assert "duration_ms" in nm[node]
        assert "outputs" in nm[node]


@pytest.mark.skipif(not CFG.is_file(), reason="configs/stage3.yaml missing")
def test_run_stage3_cli_with_stage_run_ids(monkeypatch):
    monkeypatch.chdir(REPO)
    rid = f"s3_cli_{uuid.uuid4().hex[:8]}"
    from typer.testing import CliRunner

    from rufp_stage3.cli import app

    runner = CliRunner()
    r = runner.invoke(
        app,
        [
            "run-stage3",
            "--stage3-run-id",
            rid,
            "--stage2-run-id",
            "test_run",
            "--stage25-run-id",
            "lineage_demo",
            "--config",
            str(CFG),
            "--dry-run",
        ],
    )
    assert r.exit_code == 0, r.output
    base = REPO / "artifacts" / "stage3" / rid
    assert (base / "stage3_final_report.md").is_file()
    assert (base / "metrics" / "split_stats.json").is_file()
