"""Unit tests for Stage 3 candidate merger (dedup + summary)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_merger_dedupes_same_prompt_across_validated_and_probe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s2 = tmp_path / "artifacts" / "stage2" / "r1"
    s2.mkdir(parents=True)
    sem = {
        "input": {
            "prompt_id": "p1",
            "family_id": "f1",
            "category": "c1",
            "text": "hello",
            "metadata": {"subtype": "s1"},
        },
        "safety": {
            "prompt_id": "p1",
            "safety_label": "safe",
            "confidence": 1.0,
            "rationale": "x",
            "failure_modes": [],
            "model_name": "m",
            "timestamp": "2026-01-01T00:00:00",
        },
        "naturalness": {
            "prompt_id": "p1",
            "naturalness_label": "natural",
            "confidence": 1.0,
            "rationale": "x",
            "issue_tags": [],
            "model_name": "m",
            "timestamp": "2026-01-01T00:00:00",
        },
        "borderline": {
            "prompt_id": "p1",
            "borderline_label": "strong",
            "confidence": 1.0,
            "rationale": "x",
            "issue_tags": [],
            "model_name": "m",
            "timestamp": "2026-01-01T00:00:00",
        },
        "accepted_at": "2026-01-01T00:00:00",
    }
    (s2 / "validated_semantic_set.jsonl").write_text(json.dumps(sem, ensure_ascii=False) + "\n", encoding="utf-8")

    pp = {
        "semantic_data": sem,
        "probes": [],
        "refusal_count": 0,
    }
    (s2 / "probe_positive_set.jsonl").write_text(json.dumps(pp, ensure_ascii=False) + "\n", encoding="utf-8")
    (s2 / "refusal_probe_results.jsonl").write_text("", encoding="utf-8")

    from rufp_stage3.policy.schema import FilenameMap, InputPaths, Stage3Policy
    from rufp_stage3.nodes.candidate_merger import merge_candidate_pool_with_summary

    policy = Stage3Policy(
        inputs=InputPaths(stage2_dir="artifacts/stage2/r1", stage25_dir=None),
        filenames=FilenameMap(),
    )
    rows, summary = merge_candidate_pool_with_summary(policy, dry_run=False)
    assert len(rows) == 1
    r = rows[0]
    assert set(r.source_stages) == {"stage2"}
    assert set(r.source_statuses) == {"validated", "probe_positive"}
    assert summary["pool_size"] == 1
    assert summary["count_by_source_stage"]["stage2"] == 1
    assert summary["count_by_source_status"]["validated"] == 1
    assert summary["count_by_source_status"]["probe_positive"] == 1


@pytest.mark.skipif(not (REPO / "configs" / "stage3.yaml").is_file(), reason="configs/stage3.yaml missing")
def test_cli_build_stage3_candidate_pool_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    from typer.testing import CliRunner

    from rufp_stage3.cli import app

    runner = CliRunner()
    r = runner.invoke(
        app,
        [
            "build-stage3-candidate-pool",
            "--stage2-run-id",
            "test_run",
            "--stage25-run-id",
            "lineage_demo",
            "--stage3-run-id",
            "cli_pool_test",
            "--config",
            str(REPO / "configs" / "stage3.yaml"),
        ],
    )
    assert r.exit_code == 0, r.output
    out = REPO / "artifacts" / "stage3" / "cli_pool_test" / "stage3_candidate_pool.jsonl"
    assert out.is_file()
