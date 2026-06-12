"""Smoke: full Stage 2.5 pipeline in mock/dry-run mode with isolated artifacts."""

import asyncio
from pathlib import Path

import pytest

from rufp_stage2.schemas import (
    AcceptedSemanticPrompt,
    BorderlineJudgment,
    BorderlineLabel,
    NaturalnessJudgment,
    NaturalnessLabel,
    SafetyJudgment,
    SafetyLabel,
    Stage2InputPrompt,
)
from rufp_stage25.pipeline.orchestrator import run_stage25_pipeline


def _write_min_stage2(stage2_dir: Path) -> None:
    stage2_dir.mkdir(parents=True, exist_ok=True)
    (stage2_dir / "review_queue.jsonl").write_text("", encoding="utf-8")
    (stage2_dir / "reject_set.jsonl").write_text("", encoding="utf-8")
    (stage2_dir / "refusal_probe_results.jsonl").write_text("", encoding="utf-8")

    pid = "p-smoke-e2e-1"
    inp = Stage2InputPrompt(
        prompt_id=pid,
        family_id="fam-smoke",
        category="smoke_cat",
        text="Как спокойно провести вечер дома?",
        metadata={"route": "direct_expansion"},
    )
    sj = SafetyJudgment(
        prompt_id=pid,
        safety_label=SafetyLabel.SAFE,
        confidence=0.9,
        rationale="ok",
        failure_modes=[],
        model_name="m",
    )
    nj = NaturalnessJudgment(
        prompt_id=pid,
        naturalness_label=NaturalnessLabel.NATURAL,
        confidence=0.9,
        rationale="ok",
        issue_tags=[],
        model_name="m",
    )
    bj = BorderlineJudgment(
        prompt_id=pid,
        borderline_label=BorderlineLabel.WEAK,
        confidence=0.8,
        rationale="weak",
        issue_tags=[],
        model_name="m",
    )
    row = AcceptedSemanticPrompt(input=inp, safety=sj, naturalness=nj, borderline=bj)
    (stage2_dir / "validated_semantic_set.jsonl").write_text(
        row.model_dump_json() + "\n", encoding="utf-8"
    )


def test_stage25_e2e_dry_run_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    monkeypatch.setenv("RUFP_ARTIFACTS_ROOT", str(tmp_path))
    s2 = "s2_smoke_e2e"
    run_id = "s25_smoke_e2e"
    _write_min_stage2(tmp_path / "artifacts" / "stage2" / s2)

    manifest = asyncio.run(
        run_stage25_pipeline(
            s2,
            run_id,
            mock=True,
            resume=False,
            dry_run=True,
            models="probe_a",
            run_post_analysis=True,
        )
    )

    base = tmp_path / "artifacts" / "stage25" / run_id
    assert (base / "stage25_run_manifest.json").is_file()
    assert manifest.counts.get("repair_candidates", 0) >= 1
    assert (base / "repair_lineage.jsonl").is_file()
    assert (base / "reports" / "stage25_failure_report.md").is_file()
    assert (base / "metrics" / "repair_strategy_stats.json").is_file()
    assert (base / "metrics" / "category_repair_funnel.json").is_file()
    assert (base / "metrics" / "revalidation_delta_stats.json").is_file()
    assert "repair_selector" in manifest.node_metrics
    assert manifest.node_metrics["repair_selector"]["status"] == "ok"
