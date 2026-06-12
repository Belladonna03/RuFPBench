"""Tests for split_builder_exporter (Node 5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rufp_stage3.nodes.split_builder_exporter import assign_splits, write_stage3_split_outputs
from rufp_stage3.policy.schema import DedupConfig, FilenameMap, InputPaths, SplitExportConfig, SplitConfig, Stage3Policy
from rufp_stage3.schemas import DedupClusterRecord, HardSubsetRecord, QCLabelRecord, Stage3CandidateRecord


def _row(iid: str, fam: str = "f1", cat: str = "c1") -> Stage3CandidateRecord:
    return Stage3CandidateRecord(
        item_id=iid,
        source="t",
        prompt_id=iid,
        original_prompt_id=iid,
        prompt_text="hello " * 20,
        family_id=fam,
        category=cat,
        subtype="s",
        source_stages=["stage2"],
        source_statuses=["validated"],
    )


def test_family_stays_single_split():
    r1, r2 = _row("a", fam="fx"), _row("b", fam="fx")
    policy = Stage3Policy(
        inputs=InputPaths(stage2_dir="x", stage25_dir=None),
        filenames=FilenameMap(),
        dedup=DedupConfig(),
        split=SplitConfig(dev_fraction=0.34, test_fraction=0.33, random_seed=1),
        split_export=SplitExportConfig(
            enforce_single_split_per_family=True,
            enforce_single_split_per_dedup_cluster=False,
            shuffle_groups_with_seed=False,
        ),
    )
    sp = assign_splits([r1, r2], [], policy)
    splits = {x.split for x in sp}
    assert len(splits) == 1


def test_cluster_members_share_split():
    r1, r2 = _row("x1"), _row("x2")
    cl = DedupClusterRecord(
        cluster_id="cl1",
        representative_item_id="x1",
        member_item_ids=["x1", "x2"],
        cluster_kind="near",
        category="c1",
    )
    policy = Stage3Policy(
        inputs=InputPaths(stage2_dir="x", stage25_dir=None),
        filenames=FilenameMap(),
        split=SplitConfig(dev_fraction=0.34, test_fraction=0.33, random_seed=2),
        split_export=SplitExportConfig(
            enforce_single_split_per_family=False,
            enforce_single_split_per_dedup_cluster=True,
            shuffle_groups_with_seed=False,
        ),
    )
    sp = assign_splits([r1, r2], [cl], policy)
    assert {x.split for x in sp}.__len__() == 1


REPO = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (REPO / "configs" / "stage3.yaml").is_file(), reason="configs/stage3.yaml missing")
def test_cli_run_stage3_splits(tmp_path, monkeypatch):
    monkeypatch.setenv("RUFP_ARTIFACTS_ROOT", str(tmp_path))
    run_dir = tmp_path / "artifacts" / "stage3" / "spl_cli"
    run_dir.mkdir(parents=True)
    r = _row("s1")
    (run_dir / "stage3_balanced_pool.jsonl").write_text(r.model_dump_json() + "\n", encoding="utf-8")
    qc = QCLabelRecord(item_id="s1", qc_label="keep", qc_pass=True)
    (run_dir / "stage3_qc_labels.jsonl").write_text(qc.model_dump_json() + "\n", encoding="utf-8")
    (run_dir / "stage3_dedup_clusters.jsonl").write_text("", encoding="utf-8")
    (run_dir / "stage3_hard_subset.jsonl").write_text("", encoding="utf-8")

    from typer.testing import CliRunner

    from rufp_stage3.cli import app

    runner = CliRunner()
    res = runner.invoke(app, ["run-stage3-splits", "--run-id", "spl_cli", "--config", str(REPO / "configs" / "stage3.yaml")])
    assert res.exit_code == 0, res.output
    assert (run_dir / "split_assignment.jsonl").is_file()
    assert (run_dir / "export_manifest.json").is_file()
    assert (run_dir / "stage3_exports").is_dir()
