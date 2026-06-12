"""Unit tests for repair decision aggregator."""

from datetime import datetime

from rufp_stage2.schemas import (
    SafetyJudgment,
    NaturalnessJudgment,
    BorderlineJudgment,
    SafetyLabel,
    NaturalnessLabel,
    BorderlineLabel,
    ProbeResult,
    RefusalSignal,
)

from rufp_stage25.schemas import (
    RepairedPrompt,
    RepairRevalidationResult,
    RepairStrategy,
)
from rufp_stage25.nodes.repair_decision_aggregator import classify_one, extract_before_labels_from_snapshot


def _rp() -> RepairedPrompt:
    return RepairedPrompt(
        repair_id="r1",
        repaired_prompt_id="rep-1",
        original_prompt_id="orig-1",
        parent_stage2_run_id="s2",
        parent_stage25_run_id="s25",
        stage1_prompt_id="orig-1",
        family_id="f",
        category="idioms",
        subtype=None,
        generation_route="direct_expansion",
        repair_reason="test",
        repair_strategy=RepairStrategy.NATURALIZE_RUSSIAN,
        rewrite_changed=True,
        original_hash="a",
        repaired_hash="b",
        original_prompt_text="x",
        repaired_text="y",
        change_summary="c",
    )


def _reval(
    *,
    safety: SafetyLabel,
    naturalness: NaturalnessLabel,
    borderline: BorderlineLabel,
) -> RepairRevalidationResult:
    pid = "rep-1"
    return RepairRevalidationResult(
        repaired_prompt_id=pid,
        original_prompt_id="orig-1",
        safety_label=safety,
        naturalness_label=naturalness,
        borderline_label=borderline,
        safety=SafetyJudgment(
            prompt_id=pid,
            safety_label=safety,
            confidence=0.9,
            rationale="r",
            failure_modes=[],
            model_name="m",
        ),
        naturalness=NaturalnessJudgment(
            prompt_id=pid,
            naturalness_label=naturalness,
            confidence=0.9,
            rationale="r",
            issue_tags=[],
            model_name="m",
        ),
        borderline=BorderlineJudgment(
            prompt_id=pid,
            borderline_label=borderline,
            confidence=0.9,
            rationale="r",
            issue_tags=[],
            model_name="m",
        ),
        probe_results=[
            ProbeResult(
                prompt_id=pid,
                model_name="p1",
                raw_response_text="ok",
                response_label=RefusalSignal.COMPLIANCE,
                latency_ms=1.0,
                provider_status="success",
            )
        ],
        timing_ms_total=10.0,
        timing_ms_breakdown={},
        revalidated_at=datetime.now(),
    )


def test_classify_fail_unsafe():
    rp = _rp()
    r = _reval(safety=SafetyLabel.UNSAFE, naturalness=NaturalnessLabel.NATURAL, borderline=BorderlineLabel.STRONG)
    b, reason = classify_one(rp, r, {})
    assert b == "fail"
    assert "unsafe" in reason


def test_classify_promote_triple():
    rp = _rp()
    r = _reval(safety=SafetyLabel.SAFE, naturalness=NaturalnessLabel.NATURAL, borderline=BorderlineLabel.STRONG)
    b, reason = classify_one(rp, r, {})
    assert b == "promote"
    assert "safe_natural_strong" in reason


def test_extract_flat_snapshot():
    snap = {"safety": "safe", "naturalness": "broken", "borderline": "not_borderline"}
    out = extract_before_labels_from_snapshot(snap)
    assert out.get("safety") == "safe"


def test_classify_review_weak_borderline():
    rp = _rp()
    r = _reval(safety=SafetyLabel.SAFE, naturalness=NaturalnessLabel.NATURAL, borderline=BorderlineLabel.WEAK)
    b, reason = classify_one(rp, r, {})
    assert b == "review"
    assert "weak" in reason
