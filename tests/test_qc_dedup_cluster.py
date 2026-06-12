"""Tests for Stage 3 qc_dedup_cluster node."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rufp_stage3.nodes.qc_dedup_cluster import run_qc_dedup_cluster
from rufp_stage3.policy.schema import FilenameMap, InputPaths, QcDedupClusterConfig, Stage3Policy
from rufp_stage3.schemas import Stage3CandidateRecord


def _row(
    item_id: str,
    text: str,
    *,
    cat: str = "c",
    sub: str | None = "s",
) -> Stage3CandidateRecord:
    return Stage3CandidateRecord(
        item_id=item_id,
        source="test",
        prompt_id=item_id,
        original_prompt_id=item_id,
        prompt_text=text,
        family_id="f",
        category=cat,
        subtype=sub,
    )


def test_exact_dedup_then_near_in_same_bucket():
    policy = Stage3Policy(
        inputs=InputPaths(stage2_dir="artifacts/stage2/x", stage25_dir=None),
        filenames=FilenameMap(),
        qc_dedup=QcDedupClusterConfig(min_text_len=3, near_duplicate_threshold=0.85),
    )
    rows = [
        _row("a", "hello world test", cat="c1", sub="s1"),
        _row("b", "hello world test", cat="c1", sub="s1"),
        _row("c", "completely different text here", cat="c1", sub="s1"),
    ]
    qc, clusters, pool = run_qc_dedup_cluster(rows, policy)
    by_id = {q.item_id: q.qc_label for q in qc}
    assert by_id["b"] == "exact_duplicate"
    assert by_id["a"] == "cluster_representative"
    assert by_id["c"] == "keep"
    assert len(pool) == 2
    assert any(c.cluster_kind == "exact" for c in clusters)


def test_low_value_short_text():
    policy = Stage3Policy(
        inputs=InputPaths(stage2_dir="artifacts/stage2/x", stage25_dir=None),
        filenames=FilenameMap(),
        qc_dedup=QcDedupClusterConfig(min_text_len=100),
    )
    rows = [_row("x", "short", cat="c", sub=None)]
    qc, clusters, pool = run_qc_dedup_cluster(rows, policy)
    assert qc[0].qc_label == "low_value"
    assert qc[0].qc_pass is False
    assert len(pool) == 0


REPO = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (REPO / "configs" / "stage3.yaml").is_file(), reason="configs/stage3.yaml missing")
def test_cli_run_stage3_qc(tmp_path, monkeypatch):
    monkeypatch.setenv("RUFP_ARTIFACTS_ROOT", str(tmp_path))
    run_dir = tmp_path / "artifacts" / "stage3" / "qc_cli"
    run_dir.mkdir(parents=True)
    row = _row("cli1", "hello world this is long enough", cat="c", sub="s")
    (run_dir / "stage3_candidate_pool.jsonl").write_text(row.model_dump_json() + "\n", encoding="utf-8")

    from typer.testing import CliRunner

    from rufp_stage3.cli import app

    runner = CliRunner()
    r = runner.invoke(
        app,
        ["run-stage3-qc", "--run-id", "qc_cli", "--config", str(REPO / "configs" / "stage3.yaml")],
    )
    assert r.exit_code == 0, r.output
    out = run_dir / "stage3_qc_labels.jsonl"
    assert out.is_file()
    d = json.loads(out.read_text(encoding="utf-8").strip().split("\n")[0])
    assert "qc_label" in d
