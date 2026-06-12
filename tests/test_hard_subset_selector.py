"""Tests for hard_subset_selector (Node 4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rufp_stage3.nodes.hard_subset_selector import compute_hard_score, run_hard_subset, supporting_probe_stats
from rufp_stage3.policy.schema import FilenameMap, HardSubsetConfig, InputPaths, Stage3Policy
from rufp_stage3.schemas import Stage3CandidateRecord


def _probe_row(
    item_id: str,
    *,
    probes: list[dict],
    statuses: list[str] | None = None,
) -> Stage3CandidateRecord:
    return Stage3CandidateRecord(
        item_id=item_id,
        source="test",
        prompt_id=item_id,
        original_prompt_id=item_id,
        prompt_text="x" * 150,
        family_id="f1",
        category="c1",
        subtype="s1",
        source_stages=["stage2"],
        source_statuses=statuses or ["validated", "probe_positive"],
        probe_profile={"probe_results": probes},
    )


def test_eligible_with_two_model_refusal():
    cfg = HardSubsetConfig(
        min_distinct_models_with_refusal=2,
        key_probe_refusal_alone_ok=False,
        min_score=0.0,
    )
    policy = Stage3Policy(
        inputs=InputPaths(stage2_dir="artifacts/stage2/x", stage25_dir=None),
        filenames=FilenameMap(),
        hard_subset=cfg,
    )
    probes = [
        {"model_name": "m1", "response_label": "refusal"},
        {"model_name": "m2", "response_label": "refusal"},
    ]
    row = _probe_row("a", probes=probes)
    out = run_hard_subset([row], policy)
    assert len(out) == 1
    assert out[0].hard_score > 0
    assert "refusal" in out[0].why_hard.lower() or "refusal_signal" in out[0].why_hard


def test_hard_score_components():
    cfg = HardSubsetConfig(min_score=0.0, dominated_penalty=0.38)
    policy = Stage3Policy(
        inputs=InputPaths(stage2_dir="artifacts/stage2/x", stage25_dir=None),
        filenames=FilenameMap(),
        hard_subset=cfg,
    )
    row = _probe_row(
        "b",
        probes=[
            {"model_name": "a", "response_label": "compliance"},
            {"model_name": "b", "response_label": "compliance"},
        ],
    )
    stats = supporting_probe_stats(row, None)
    score, why, st2, _tags = compute_hard_score(row, dominated=False, cfg=policy.hard_subset)
    assert 0 <= score <= 1
    assert st2["models_with_refusal_count"] == 0


REPO = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (REPO / "configs" / "stage3.yaml").is_file(), reason="configs/stage3.yaml missing")
def test_cli_run_stage3_hard_subset(tmp_path, monkeypatch):
    monkeypatch.setenv("RUFP_ARTIFACTS_ROOT", str(tmp_path))
    run_dir = tmp_path / "artifacts" / "stage3" / "hard_cli"
    run_dir.mkdir(parents=True)
    row = _probe_row(
        "h1",
        probes=[
            {"model_name": "m1", "response_label": "refusal"},
            {"model_name": "m2", "response_label": "refusal"},
        ],
    )
    (run_dir / "stage3_balanced_pool.jsonl").write_text(row.model_dump_json() + "\n", encoding="utf-8")

    from typer.testing import CliRunner

    from rufp_stage3.cli import app

    runner = CliRunner()
    res = runner.invoke(
        app,
        ["run-stage3-hard-subset", "--run-id", "hard_cli", "--config", str(REPO / "configs" / "stage3.yaml")],
    )
    assert res.exit_code == 0, res.output
    p = run_dir / "stage3_hard_subset.jsonl"
    assert p.is_file()
    line = json.loads(p.read_text(encoding="utf-8").strip().split("\n")[0])
    assert "hard_score" in line
