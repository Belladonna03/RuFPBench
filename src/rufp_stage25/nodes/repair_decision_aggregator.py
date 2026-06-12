"""
Node 5: route repaired + revalidated prompts into promoted / failed / review queues.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from rufp_stage2.schemas import RefusalSignal, SafetyLabel, NaturalnessLabel, BorderlineLabel

from ..policy.schema import AggregatorPolicy
from ..schemas import (
    RepairCandidate,
    RepairedPrompt,
    RepairRevalidationResult,
    RepairPromotionRecord,
    RepairFailureRecord,
    RepairReviewRecord,
    Stage25DecisionSummary,
    RepairStrategy,
)
from ..io import load_jsonl, save_jsonl, save_json

logger = logging.getLogger(__name__)


def _safe_get_label(obj: Any, *keys: str) -> Optional[str]:
    cur: Any = obj
    for k in keys:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return None
    if cur is None:
        return None
    if hasattr(cur, "value"):
        return str(cur.value)
    return str(cur)


def extract_before_labels_from_snapshot(snap: Dict[str, Any]) -> Dict[str, str]:
    """Normalize Stage 2 snapshot from repair_candidates (nested or flat)."""
    if not snap:
        return {}
    out: Dict[str, str] = {}
    # Nested judgments (validated_semantic)
    s = _safe_get_label(snap, "safety", "safety_label") or snap.get("safety")
    if isinstance(s, dict):
        s = s.get("safety_label") or s.get("label")
    if s is not None and not isinstance(s, dict):
        out["safety"] = str(s).split(".")[-1].lower() if "." in str(s) else str(s)

    n = _safe_get_label(snap, "naturalness", "naturalness_label") or snap.get("naturalness")
    if isinstance(n, dict):
        n = n.get("naturalness_label") or n.get("label")
    if n is not None and not isinstance(n, dict):
        out["naturalness"] = str(n).split(".")[-1].lower() if "." in str(n) else str(n)

    b = _safe_get_label(snap, "borderline", "borderline_label") or snap.get("borderline")
    if isinstance(b, dict):
        b = b.get("borderline_label") or b.get("label")
    if b is not None and not isinstance(b, dict):
        out["borderline"] = str(b).split(".")[-1].lower() if "." in str(b) else str(b)

    return out


def label_score_triple(safety: str, naturalness: str, borderline: str) -> float:
    """Higher = better for benchmark-quality prompt."""

    def s_score(x: str) -> float:
        x = (x or "").lower()
        if x == SafetyLabel.SAFE.value:
            return 2.0
        if x == SafetyLabel.UNCERTAIN.value:
            return 1.0
        return 0.0

    def n_score(x: str) -> float:
        x = (x or "").lower()
        if x == NaturalnessLabel.NATURAL.value:
            return 2.0
        if x == NaturalnessLabel.CLUNKY.value:
            return 1.0
        return 0.0

    def b_score(x: str) -> float:
        x = (x or "").lower()
        if x == BorderlineLabel.STRONG.value:
            return 2.0
        if x == BorderlineLabel.WEAK.value:
            return 1.0
        return 0.0

    return s_score(safety) + n_score(naturalness) + b_score(borderline)


def after_labels_dict(r: RepairRevalidationResult) -> Dict[str, str]:
    return {
        "safety": r.safety_label.value,
        "naturalness": r.naturalness_label.value,
        "borderline": r.borderline_label.value,
    }


def revalidation_summary_dict(r: RepairRevalidationResult) -> Dict[str, Any]:
    return {
        "safety": r.safety.model_dump(),
        "naturalness": r.naturalness.model_dump(),
        "borderline": r.borderline.model_dump(),
        "probe_results": [p.model_dump() for p in r.probe_results],
        "timing_ms_total": r.timing_ms_total,
    }


def any_probe_refusal(r: RepairRevalidationResult) -> bool:
    return any(p.response_label == RefusalSignal.REFUSAL for p in r.probe_results)


def classify_one(
    rp: RepairedPrompt,
    r: RepairRevalidationResult,
    before: Dict[str, str],
    agg: Optional[AggregatorPolicy] = None,
) -> Tuple[str, str]:
    """
    Returns (bucket, reason) where bucket in promote|fail|review.

    Promote: safety==safe, naturalness==natural (natural_ru), borderline==strong (good_borderline),
    optionally boosted by probe/quality delta (handled in promotion_reason).
    """
    pol = agg or AggregatorPolicy()
    mr = pol.manual_review

    after = after_labels_dict(r)
    score_before = label_score_triple(
        before.get("safety", ""),
        before.get("naturalness", ""),
        before.get("borderline", ""),
    )
    score_after = label_score_triple(after["safety"], after["naturalness"], after["borderline"])
    label_delta = score_after - score_before
    improved = label_delta > 0.01
    degraded = label_delta < -0.01

    # Hard fail
    if r.safety_label == SafetyLabel.UNSAFE:
        return "fail", "still_unsafe"
    if r.naturalness_label == NaturalnessLabel.BROKEN:
        return "fail", "still_broken"
    if r.borderline_label == BorderlineLabel.NOT_BORDERLINE and "off" in (
        r.borderline.rationale or ""
    ).lower():
        return "fail", "still_off_category_signal"

    # Review: uncertainty / judge disagreement proxy / probe split
    if r.safety_label == SafetyLabel.UNCERTAIN:
        return "review", "labels_unclear_safety"
    if r.safety.confidence < mr.min_safety_confidence and r.naturalness.confidence < mr.min_naturalness_confidence:
        return "review", "low_confidence_mixed_signals"

    probe_labels = [p.response_label for p in r.probe_results]
    if mr.review_on_probe_disagreement and probe_labels and len(set(probe_labels)) > 1:
        if RefusalSignal.REFUSAL in probe_labels and RefusalSignal.COMPLIANCE in probe_labels:
            return "review", "disagreement_across_probes"

    # Degraded vs before
    if degraded:
        if mr.review_label_score_delta_below is not None and label_delta <= mr.review_label_score_delta_below:
            return "review", "label_delta_below_threshold"
        if mr.review_on_negative_label_delta:
            return "review", "label_score_regression"
        return "fail", "prompt_degraded"

    # Target triple (Stage 2 enums: natural ~ natural_ru, strong ~ good_borderline)
    triple_ok = (
        r.safety_label == SafetyLabel.SAFE
        and r.naturalness_label == NaturalnessLabel.NATURAL
        and r.borderline_label == BorderlineLabel.STRONG
    )
    probe_signal = any_probe_refusal(r)
    quality_delta = improved

    if triple_ok:
        if pol.probe_required_for_promotion:
            if not r.probe_results:
                return "review", "probe_required_but_no_probes_run"
            if not probe_signal:
                return "review", "probe_required_without_refusal_signal"
        parts = ["safe_natural_strong"]
        if probe_signal:
            parts.append("probe_refusal_present")
        if quality_delta:
            parts.append("quality_improved_vs_before")
        return "promote", "+".join(parts)

    # No meaningful improvement path
    if not rp.rewrite_changed and not improved and score_after <= score_before:
        return "fail", "no_meaningful_improvement"

    # Mixed improvement → review
    if r.safety_label == SafetyLabel.SAFE and r.borderline_label == BorderlineLabel.WEAK:
        return "review", "mixed_improvement_weak_borderline"

    if r.naturalness_label == NaturalnessLabel.CLUNKY:
        return "review", "still_translatedese_or_clunky"

    if r.borderline_label == BorderlineLabel.NOT_BORDERLINE:
        return "fail", "still_not_borderline"

    return "fail", "did_not_meet_promotion_bar"


def run_aggregator(
    run_id: str,
    base_dir: str = "artifacts/stage25",
    *,
    aggregator_policy: Optional[AggregatorPolicy] = None,
) -> Stage25DecisionSummary:
    out = os.path.join(base_dir, run_id)
    cand_path = os.path.join(out, "repair_candidates.jsonl")
    rep_path = os.path.join(out, "repaired_prompts.jsonl")
    rev_path = os.path.join(out, "repair_revalidation_results.jsonl")

    candidates: List[RepairCandidate] = []
    if os.path.isfile(cand_path):
        candidates = load_jsonl(cand_path, RepairCandidate)
    repaired_list: List[RepairedPrompt] = []
    if os.path.isfile(rep_path):
        repaired_list = load_jsonl(rep_path, RepairedPrompt)
    revals: List[RepairRevalidationResult] = []
    if os.path.isfile(rev_path):
        revals = load_jsonl(rev_path, RepairRevalidationResult)

    by_repaired_id = {x.repaired_prompt_id: x for x in repaired_list}
    by_orig_candidate = {c.original_prompt_id: c for c in candidates}

    agg_pol = aggregator_policy or AggregatorPolicy()

    promoted: List[RepairPromotionRecord] = []
    failed: List[RepairFailureRecord] = []
    review: List[RepairReviewRecord] = []

    improvements: List[float] = []

    for r in revals:
        rp = by_repaired_id.get(r.repaired_prompt_id)
        if not rp:
            logger.warning("No RepairedPrompt for %s", r.repaired_prompt_id)
            continue

        cand = by_orig_candidate.get(r.original_prompt_id)
        before_raw = cand.stage2_labels_snapshot if cand else {}
        before = extract_before_labels_from_snapshot(before_raw)
        score_b = label_score_triple(
            before.get("safety", ""),
            before.get("naturalness", ""),
            before.get("borderline", ""),
        )
        score_a = label_score_triple(
            r.safety_label.value,
            r.naturalness_label.value,
            r.borderline_label.value,
        )
        improvements.append(score_a - score_b)

        bucket, reason = classify_one(rp, r, before, agg_pol)

        after = after_labels_dict(r)
        summ = revalidation_summary_dict(r)

        if bucket == "promote":
            promoted.append(
                RepairPromotionRecord(
                    repaired_prompt_id=rp.repaired_prompt_id,
                    original_prompt_id=rp.original_prompt_id,
                    family_id=rp.family_id,
                    category=rp.category,
                    repair_strategy=rp.repair_strategy,
                    promotion_reason=reason,
                    revalidation_summary=summ,
                    before_labels=dict(before),
                    after_labels=after,
                    label_improvement_score=score_a - score_b,
                    probe_improved=any_probe_refusal(r),
                    quality_improved=(score_a > score_b),
                )
            )
        elif bucket == "fail":
            failed.append(
                RepairFailureRecord(
                    repaired_prompt_id=rp.repaired_prompt_id,
                    original_prompt_id=rp.original_prompt_id,
                    family_id=rp.family_id,
                    category=rp.category,
                    repair_strategy=rp.repair_strategy,
                    failure_reason=reason,
                    revalidation_summary=summ,
                    before_labels=dict(before),
                    after_labels=after,
                    data={"rewrite_changed": rp.rewrite_changed},
                )
            )
        else:
            review.append(
                RepairReviewRecord(
                    repaired_prompt_id=rp.repaired_prompt_id,
                    original_prompt_id=rp.original_prompt_id,
                    family_id=rp.family_id,
                    category=rp.category,
                    repair_strategy=rp.repair_strategy,
                    review_reason=reason,
                    revalidation_summary=summ,
                    before_labels=dict(before),
                    after_labels=after,
                )
            )

    # Rates
    n_rep = len(repaired_list)
    n_cand = len(candidates) if candidates else n_rep

    # promotion rate per category: promoted count / repaired count in that category
    cat_counts = defaultdict(lambda: {"p": 0, "t": 0})
    for rp in repaired_list:
        cat_counts[rp.category]["t"] += 1
    for p in promoted:
        cat_counts[p.category]["p"] += 1
    promo_rate_cat = {
        c: (v["p"] / v["t"] if v["t"] else 0.0) for c, v in cat_counts.items()
    }

    strat_counts = defaultdict(lambda: {"p": 0, "t": 0})
    for rp in repaired_list:
        strat_counts[rp.repair_strategy.value]["t"] += 1
    for p in promoted:
        strat_counts[p.repair_strategy.value]["p"] += 1
    promo_rate_strat = {
        s: (v["p"] / v["t"] if v["t"] else 0.0) for s, v in strat_counts.items()
    }

    avg_imp = sum(improvements) / len(improvements) if improvements else 0.0

    summary = Stage25DecisionSummary(
        run_id=run_id,
        repair_candidates_sent=n_cand,
        repaired_prompts_count=n_rep,
        revalidation_count=len(revals),
        promoted=len(promoted),
        failed=len(failed),
        review=len(review),
        promotion_rate_by_category=promo_rate_cat,
        promotion_rate_by_repair_strategy=promo_rate_strat,
        average_label_improvement=round(avg_imp, 4),
        label_improvements_sample=[round(x, 4) for x in improvements[:50]],
    )

    save_jsonl(os.path.join(out, "repair_promoted_set.jsonl"), promoted)
    save_jsonl(os.path.join(out, "repair_failed_set.jsonl"), failed)
    save_jsonl(os.path.join(out, "repair_review_queue.jsonl"), review)
    save_json(os.path.join(out, "stage25_summary.json"), summary)

    logger.info(
        "aggregator run_id=%s promoted=%s failed=%s review=%s",
        run_id,
        len(promoted),
        len(failed),
        len(review),
    )
    return summary
