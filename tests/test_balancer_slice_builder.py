"""Tests for balancer_slice_builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rufp_stage3.nodes.balancer_slice_builder import run_balancer_slice_builder
from rufp_stage3.policy.schema import BalanceConfig, FilenameMap, InputPaths, Stage3Policy
from rufp_stage3.schemas import Stage3CandidateRecord


def _row(
    item_id: str,
    *,
    cat: str = "c",
    fam: str = "f1",
    text: str = "hello world " * 10,
    statuses: list[str] | None = None,
    stages: list[str] | None = None,
) -> Stage3CandidateRecord:
    return Stage3CandidateRecord(
        item_id=item_id,
        source="test",
        prompt_id=item_id,
        original_prompt_id=item_id,
        prompt_text=text,
        family_id=fam,
        category=cat,
        subtype="s",
        source_stages=stages or ["stage2"],
        source_statuses=statuses or ["validated"],
        probe_profile={},
    )


def test_family_cap_drops_excess():
    cfg = BalanceConfig(max_rows_per_family_per_category=1)
    policy = Stage3Policy(
        inputs=InputPaths(stage2_dir="artifacts/stage2/x", stage25_dir=None),
        filenames=FilenameMap(),
        balance=cfg,
    )
    rows = [
        _row("a", fam="famX", text="x" * 20),
        _row("b", fam="famX", text="y" * 20),
    ]
    out, meta, idx = run_balancer_slice_builder(rows, policy)
    assert len(out) == 1
    assert out[0].item_id == "a"
    assert "category:c" in idx["slices"]


def test_probe_and_repaired_slices():
    cfg = BalanceConfig()
    policy = Stage3Policy(
        inputs=InputPaths(stage2_dir="artifacts/stage2/x", stage25_dir=None),
        filenames=FilenameMap(),
        balance=cfg,
    )
    rows = [
        _row("p1", statuses=["probe_positive"]),
        _row("r1", statuses=["repaired_accept"], stages=["stage25"]),
    ]
    _out, _meta, idx = run_balancer_slice_builder(rows, policy)
    assert set(idx["slices"]["probe_positive"]) == {"p1"}
    assert set(idx["slices"]["repaired"]) == {"r1"}


REPO = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (REPO / "configs" / "stage3.yaml").is_file(), reason="configs/stage3.yaml missing")
def test_cli_run_stage3_balancer(tmp_path, monkeypatch):
    monkeypatch.setenv("RUFP_ARTIFACTS_ROOT", str(tmp_path))
    run_dir = tmp_path / "artifacts" / "stage3" / "bal_cli"
    run_dir.mkdir(parents=True)
    r = _row("cli1", text="hello world " * 5)
    (run_dir / "stage3_candidate_pool.jsonl").write_text(r.model_dump_json() + "\n", encoding="utf-8")
    from rufp_stage3.schemas import QCLabelRecord

    qc = QCLabelRecord(item_id="cli1", qc_label="keep", qc_pass=True)
    (run_dir / "stage3_qc_labels.jsonl").write_text(qc.model_dump_json() + "\n", encoding="utf-8")

    from typer.testing import CliRunner

    from rufp_stage3.cli import app

    runner = CliRunner()
    res = runner.invoke(
        app,
        ["run-stage3-balancer", "--run-id", "bal_cli", "--config", str(REPO / "configs" / "stage3.yaml")],
    )
    assert res.exit_code == 0, res.output
    si = json.loads((run_dir / "stage3_slice_index.json").read_text(encoding="utf-8"))
    assert "slices" in si
    assert (run_dir / "stage3_balanced_pool.jsonl").is_file()
