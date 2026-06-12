"""Release layer: lineage, dataset card, prepare CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "configs" / "stage3.yaml"


def _write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@pytest.mark.skipif(not CFG.is_file(), reason="configs/stage3.yaml missing")
def test_prepare_stage3_release_from_disk_writes_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("RUFP_ARTIFACTS_ROOT", str(tmp_path))
    rid = "rel_smoke"
    base = tmp_path / "artifacts" / "stage3" / rid
    base.mkdir(parents=True)

    cand = {
        "item_id": "s3-testitem000001",
        "source": "merged",
        "prompt_id": "p1",
        "original_prompt_id": "p1",
        "stage1_prompt_id": None,
        "prompt_text": "hello world " * 8,
        "family_id": "fam1",
        "category": "test_cat",
        "subtype": "st",
        "source_stages": ["stage2"],
        "source_statuses": ["validated"],
        "probe_profile": {},
        "lineage_refs": [],
        "generation_route": "",
        "lineage": {},
        "metadata": {
            "stage2_validated_labels": {
                "input": {
                    "prompt_id": "p1",
                    "category": "test_cat",
                    "family_id": "fam1",
                    "metadata": {"subtype": "st"},
                },
                "safety": {"safety_label": "safe", "confidence": 0.9},
                "naturalness": {"naturalness_label": "natural", "confidence": 0.8},
                "borderline": {"borderline_label": "strong", "confidence": 0.7},
            }
        },
    }
    _write_jsonl(base / "stage3_balanced_pool.jsonl", [cand])

    qc = {
        "item_id": "s3-testitem000001",
        "qc_label": "keep",
        "qc_pass": True,
        "qc_flags": [],
        "notes": "",
        "cluster_id": None,
    }
    _write_jsonl(base / "stage3_qc_labels.jsonl", [qc])

    sp = {
        "item_id": "s3-testitem000001",
        "split": "dev",
        "fold_seed": 42,
        "family_id": "fam1",
        "cluster_id": None,
        "constraint_group_id": "s3-testitem000001",
        "metadata": {},
    }
    _write_jsonl(base / "split_assignment.jsonl", [sp])

    from rufp_stage3.release.bundle import prepare_stage3_release_from_disk

    out = prepare_stage3_release_from_disk(rid, config_path=str(CFG))
    assert "final_lineage" in out
    exp = base / "stage3_exports"
    assert (exp / "final_lineage.jsonl").is_file()
    assert (exp / "dataset_card.md").is_file()
    assert (exp / "release_checklist.md").is_file()

    line = (exp / "final_lineage.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
    row = json.loads(line)
    assert row["schema_version"] == 1
    assert row["item_id"] == "s3-testitem000001"
    assert row["stage1_prompt"]["family_id"] == "fam1"
    assert "validated_semantic" in row["stage2"]
    assert row["stage3_qc"]["qc_label"] == "keep"
    assert row["final_split_assignment"]["split"] == "dev"
    assert row["in_hard_subset"] is False


def test_final_lineage_record_schema():
    from rufp_stage3.schemas import FinalLineageRecord

    r = FinalLineageRecord(
        stage3_run_id="r",
        item_id="i",
        stage1_prompt={"prompt_id": "p"},
        stage2={"source_stages": ["stage2"]},
        stage3_qc={"qc_pass": True},
        final_split_assignment={"split": "test"},
    )
    assert r.model_dump(mode="json")["schema_version"] == 1
