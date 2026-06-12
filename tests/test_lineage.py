"""Tests for Stage 2.5 lineage hashing and audit helpers."""

from rufp_stage25.utils.hashing import text_sha256
from rufp_stage25.utils.diff_summary import simple_change_summary
from rufp_stage25.schemas import (
    RepairStrategy,
    RepairedPrompt,
    FinalRepairDecision,
)
from rufp_stage25.nodes.lineage_builder import build_lineage_record, infer_decision_from_rewrite


def test_text_sha256_stable():
    assert text_sha256("hello") == text_sha256("hello")
    assert text_sha256("hello") != text_sha256("hello ")


def test_infer_decision():
    assert infer_decision_from_rewrite(False) == FinalRepairDecision.NO_IMPROVEMENT
    assert infer_decision_from_rewrite(True) == FinalRepairDecision.PENDING_REVALIDATION


def test_build_lineage_record():
    rp = RepairedPrompt(
        repair_id="r1",
        repaired_prompt_id="rep1",
        original_prompt_id="orig1",
        parent_stage2_run_id="s2",
        parent_stage25_run_id="s25",
        stage1_prompt_id="orig1",
        family_id="f",
        category="c",
        subtype=None,
        generation_route="direct",
        repair_reason="too_bland",
        repair_strategy=RepairStrategy.NATURALIZE_RUSSIAN,
        rewrite_changed=True,
        original_hash="a",
        repaired_hash="b",
        original_prompt_text="x",
        repaired_text="y",
        change_summary="len_delta=+1",
    )
    rec = build_lineage_record(
        rp,
        stage2_labels_snapshot={"s": 1},
        repair_plan_snapshot={"p": 2},
        revalidation_snapshot={},
        final_repair_decision=FinalRepairDecision.PENDING_REVALIDATION,
    )
    assert rec.repair_id == "r1"
    assert rec.stage1_prompt_id == "orig1"
    assert rec.repair_changed is True


def test_simple_change_summary_no_change():
    assert simple_change_summary("a", "a") == "no_change"
