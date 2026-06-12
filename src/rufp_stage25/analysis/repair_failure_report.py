"""
Post-run Stage 2.5 analysis: strategy/category funnel, deltas, narrative failure report.

Writes under ``artifacts/stage25/<run_id>/reports`` and ``.../metrics``.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from ..io import load_jsonl
from ..nodes.repair_decision_aggregator import extract_before_labels_from_snapshot, label_score_triple
from ..schemas import (
    RepairCandidate,
    RepairFailureRecord,
    RepairPlan,
    RepairPromotionRecord,
    RepairRevalidationResult,
    RepairReviewRecord,
    RepairedPrompt,
    Stage25DecisionSummary,
)


def _safe_read_jsonl(path: str, model: Any) -> List[Any]:
    if not os.path.isfile(path):
        return []
    return load_jsonl(path, model)


def _load_summary(path: str) -> Optional[Stage25DecisionSummary]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return Stage25DecisionSummary.model_validate_json(f.read())


def _load_lineage_dicts(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_repair_strategy_stats(
    repaired: List[RepairedPrompt],
    promoted: List[RepairPromotionRecord],
    failed: List[RepairFailureRecord],
    review: List[RepairReviewRecord],
    summary: Optional[Stage25DecisionSummary],
) -> Dict[str, Any]:
    by_strat: Dict[str, Dict[str, Any]] = {}
    for rp in repaired:
        s = rp.repair_strategy.value
        if s not in by_strat:
            by_strat[s] = {
                "repaired_count": 0,
                "promoted": 0,
                "failed": 0,
                "review": 0,
                "label_improvements": [],
            }
        by_strat[s]["repaired_count"] += 1

    for p in promoted:
        s = p.repair_strategy.value
        by_strat.setdefault(
            s,
            {
                "repaired_count": 0,
                "promoted": 0,
                "failed": 0,
                "review": 0,
                "label_improvements": [],
            },
        )
        by_strat[s]["promoted"] += 1
        by_strat[s]["label_improvements"].append(p.label_improvement_score)

    for f in failed:
        s = f.repair_strategy.value
        by_strat.setdefault(
            s,
            {
                "repaired_count": 0,
                "promoted": 0,
                "failed": 0,
                "review": 0,
                "label_improvements": [],
            },
        )
        by_strat[s]["failed"] += 1

    for r in review:
        s = r.repair_strategy.value
        by_strat.setdefault(
            s,
            {
                "repaired_count": 0,
                "promoted": 0,
                "failed": 0,
                "review": 0,
                "label_improvements": [],
            },
        )
        by_strat[s]["review"] += 1

    out: Dict[str, Any] = {}
    for s, v in sorted(by_strat.items()):
        rc = v["repaired_count"] or 1
        imps = v["label_improvements"]
        out[s] = {
            "repaired_count": v["repaired_count"],
            "promoted": v["promoted"],
            "failed": v["failed"],
            "review": v["review"],
            "promotion_rate": round(v["promoted"] / max(v["repaired_count"], 1), 4),
            "avg_label_improvement": round(sum(imps) / len(imps), 4) if imps else 0.0,
        }

    if summary:
        out["_summary_ref"] = {
            "average_label_improvement": summary.average_label_improvement,
            "promotion_rate_by_repair_strategy": summary.promotion_rate_by_repair_strategy,
        }
    return {"by_strategy": out}


def build_category_repair_funnel(
    candidates: List[RepairCandidate],
    plans: List[RepairPlan],
    repaired: List[RepairedPrompt],
    promoted: List[RepairPromotionRecord],
    failed: List[RepairFailureRecord],
    review: List[RepairReviewRecord],
) -> Dict[str, Any]:
    """Per-category counts at each stage."""

    def key_cat(x: Any) -> str:
        return getattr(x, "category", None) or "unknown"

    funnel: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {
            "candidates": 0,
            "plans": 0,
            "repaired": 0,
            "promoted": 0,
            "failed": 0,
            "review": 0,
        }
    )

    for c in candidates:
        funnel[key_cat(c)]["candidates"] += 1
    for p in plans:
        oc = next((c for c in candidates if c.original_prompt_id == p.original_prompt_id), None)
        cat = key_cat(oc) if oc else "unknown"
        funnel[cat]["plans"] += 1
    for rp in repaired:
        funnel[key_cat(rp)]["repaired"] += 1
    for x in promoted:
        funnel[key_cat(x)]["promoted"] += 1
    for x in failed:
        funnel[key_cat(x)]["failed"] += 1
    for x in review:
        funnel[key_cat(x)]["review"] += 1

    repairable_rate: Dict[str, float] = {}
    for cat, v in funnel.items():
        c0 = v["candidates"] or 1
        repairable_rate[cat] = round(v["repaired"] / c0, 4)

    return {"funnel_by_category": dict(funnel), "repair_completion_rate_by_category": repairable_rate}


def build_revalidation_delta_stats(
    candidates: List[RepairCandidate],
    revals: List[RepairRevalidationResult],
) -> Dict[str, Any]:
    """Before (Stage 2 snapshot) vs after (revalidation) label scores."""

    by_orig = {c.original_prompt_id: c for c in candidates}
    deltas: List[float] = []
    degradations: List[Dict[str, Any]] = []
    improvements: List[Dict[str, Any]] = []

    for r in revals:
        cand = by_orig.get(r.original_prompt_id)
        before_raw = cand.stage2_labels_snapshot if cand else {}
        before = extract_before_labels_from_snapshot(before_raw)
        sb = label_score_triple(
            before.get("safety", ""),
            before.get("naturalness", ""),
            before.get("borderline", ""),
        )
        sa = label_score_triple(
            r.safety_label.value,
            r.naturalness_label.value,
            r.borderline_label.value,
        )
        d = sa - sb
        deltas.append(d)
        row = {
            "repaired_prompt_id": r.repaired_prompt_id,
            "original_prompt_id": r.original_prompt_id,
            "delta": round(d, 4),
            "before_labels": before,
            "after_labels": {
                "safety": r.safety_label.value,
                "naturalness": r.naturalness_label.value,
                "borderline": r.borderline_label.value,
            },
        }
        if d < -0.01:
            degradations.append(row)
        elif d > 0.01:
            improvements.append(row)

    return {
        "count": len(revals),
        "mean_delta": round(sum(deltas) / len(deltas), 4) if deltas else 0.0,
        "degradation_count": len(degradations),
        "improvement_count": len(improvements),
        "sample_degradations": degradations[:30],
        "sample_improvements": improvements[:30],
    }


def _rank_strategies(by_strat: Dict[str, Any]) -> List[Tuple[str, float, float]]:
    rows: List[Tuple[str, float, float]] = []
    for s, v in by_strat.items():
        if s.startswith("_"):
            continue
        pr = float(v.get("promotion_rate", 0.0))
        avg = float(v.get("avg_label_improvement", 0.0))
        rows.append((s, pr, avg))
    rows.sort(key=lambda x: (-x[1], -x[2]))
    return rows


def _reject_vs_repair_scores(funnel: Dict[str, Dict[str, int]]) -> List[Dict[str, Any]]:
    """Higher score => prefer reject over repair (many failed+review vs promoted)."""

    scored: List[Dict[str, Any]] = []
    for cat, v in funnel.items():
        p, f, rev = v["promoted"], v["failed"], v["review"]
        t = p + f + rev
        if t == 0:
            continue
        bad_ratio = (f + rev) / max(t, 1)
        scored.append(
            {
                "category": cat,
                "promoted": p,
                "failed": f,
                "review": rev,
                "bad_outcome_ratio": round(bad_ratio, 4),
            }
        )
    scored.sort(key=lambda x: (-x["bad_outcome_ratio"], -(x["failed"] + x["review"])))
    return scored


def render_failure_report_md(
    run_id: str,
    strategy_stats: Dict[str, Any],
    funnel_payload: Dict[str, Any],
    delta_stats: Dict[str, Any],
    lineage_rows: List[Dict[str, Any]],
    failed: List[RepairFailureRecord],
    promoted: List[RepairPromotionRecord],
    review: List[RepairReviewRecord],
    summary: Optional[Stage25DecisionSummary],
) -> str:
    by_strat = strategy_stats.get("by_strategy", {})
    top_strat = _rank_strategies(by_strat)

    funnel = funnel_payload.get("funnel_by_category", {})
    reject_scores = _reject_vs_repair_scores(funnel)

    fail_reasons = Counter(f.failure_reason for f in failed)
    remain_issues = Counter()
    for f in failed:
        for k, v in (f.after_labels or {}).items():
            remain_issues[f"after:{k}={v}"] += 1

    no_improvement = [r for r in lineage_rows if r.get("final_repair_decision") == "no_improvement"]
    pending = [r for r in lineage_rows if r.get("final_repair_decision") == "pending_revalidation"]

    lines = [
        f"# Stage 2.5 failure & repair effectiveness report (`{run_id}`)",
        "",
        "## Executive summary",
    ]
    if summary:
        lines.extend(
            [
                f"- Repair candidates (sent): **{summary.repair_candidates_sent}**",
                f"- Repaired prompts: **{summary.repaired_prompts_count}**",
                f"- Revalidations: **{summary.revalidation_count}**",
                f"- Promoted / failed / review: **{summary.promoted}** / **{summary.failed}** / **{summary.review}**",
                f"- Average label-score delta: **{summary.average_label_improvement}**",
                "",
            ]
        )

    lines.extend(
        [
            "## Which repair strategies work best",
            "Ranked by promotion rate, then average label improvement within strategy:",
            "",
        ]
    )
    for s, pr, avg in top_strat[:15]:
        lines.append(f"- **{s}**: promotion_rate={pr}, avg_label_improvement={avg}")
    lines.append("")

    lines.extend(
        [
            "## Which categories are most often repairable",
            "Higher `repaired / candidates` means more items reached the repairer:",
            "",
            "```json",
            json.dumps(funnel_payload.get("repair_completion_rate_by_category", {}), ensure_ascii=False, indent=2),
            "```",
            "",
            "## Where repair most often does not help",
            "Top failure reasons (aggregator):",
            "",
            "```json",
            json.dumps(dict(fail_reasons.most_common(25)), ensure_ascii=False, indent=2),
            "```",
            "",
            "## What problems remain after repair (failed bucket, label snapshot)",
            "",
            "```json",
            json.dumps(dict(remain_issues.most_common(30)), ensure_ascii=False, indent=2),
            "```",
            "",
            "## When repair degrades the prompt (label-score delta < 0)",
            f"Count: **{delta_stats.get('degradation_count', 0)}** (see `metrics/revalidation_delta_stats.json` for samples).",
            "",
            "## Categories where reject may beat repair",
            "Sorted by share of bad outcomes (failed+review) among terminal decisions:",
            "",
            "```json",
            json.dumps(reject_scores[:25], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Lineage: no improvement vs still pending",
            f"- Rows with `no_improvement`: **{len(no_improvement)}**",
            f"- Rows still `pending_revalidation` in lineage: **{len(pending)}** (finalize after aggregator if needed).",
            "",
            "## Review queue highlights",
            "",
            "```json",
            json.dumps(
                [{"id": r.repaired_prompt_id, "reason": r.review_reason} for r in review[:20]],
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
        ]
    )

    return "\n".join(lines)


def write_repair_failure_artifacts(run_id: str, stage25_root: Optional[str] = None) -> Dict[str, str]:
    """
    Write markdown report + JSON metrics. Returns map of logical name -> absolute path.

    Outputs:
      - reports/stage25_failure_report.md
      - metrics/repair_strategy_stats.json
      - metrics/category_repair_funnel.json
      - metrics/revalidation_delta_stats.json
    """
    base = stage25_root or os.path.join("artifacts", "stage25", run_id)
    reports_dir = os.path.join(base, "reports")
    metrics_dir = os.path.join(base, "metrics")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    cand_path = os.path.join(base, "repair_candidates.jsonl")
    plans_path = os.path.join(base, "repair_plans.jsonl")
    rep_path = os.path.join(base, "repaired_prompts.jsonl")
    rev_path = os.path.join(base, "repair_revalidation_results.jsonl")
    lin_path = os.path.join(base, "repair_lineage.jsonl")
    summary_path = os.path.join(base, "stage25_summary.json")

    candidates = _safe_read_jsonl(cand_path, RepairCandidate)
    plans = _safe_read_jsonl(plans_path, RepairPlan)
    repaired = _safe_read_jsonl(rep_path, RepairedPrompt)
    revals = _safe_read_jsonl(rev_path, RepairRevalidationResult)
    promoted = _safe_read_jsonl(os.path.join(base, "repair_promoted_set.jsonl"), RepairPromotionRecord)
    failed = _safe_read_jsonl(os.path.join(base, "repair_failed_set.jsonl"), RepairFailureRecord)
    review = _safe_read_jsonl(os.path.join(base, "repair_review_queue.jsonl"), RepairReviewRecord)
    summary = _load_summary(summary_path)
    lineage_rows = _load_lineage_dicts(lin_path)

    strategy_stats = build_repair_strategy_stats(repaired, promoted, failed, review, summary)
    funnel_payload = build_category_repair_funnel(candidates, plans, repaired, promoted, failed, review)
    delta_stats = build_revalidation_delta_stats(candidates, revals)

    paths: Dict[str, str] = {}

    p_strat = os.path.join(metrics_dir, "repair_strategy_stats.json")
    with open(p_strat, "w", encoding="utf-8") as f:
        json.dump(strategy_stats, f, ensure_ascii=False, indent=2)
    paths["repair_strategy_stats"] = p_strat

    p_fun = os.path.join(metrics_dir, "category_repair_funnel.json")
    with open(p_fun, "w", encoding="utf-8") as f:
        json.dump(funnel_payload, f, ensure_ascii=False, indent=2)
    paths["category_repair_funnel"] = p_fun

    p_delta = os.path.join(metrics_dir, "revalidation_delta_stats.json")
    with open(p_delta, "w", encoding="utf-8") as f:
        json.dump(delta_stats, f, ensure_ascii=False, indent=2)
    paths["revalidation_delta_stats"] = p_delta

    md = render_failure_report_md(
        run_id,
        strategy_stats,
        funnel_payload,
        delta_stats,
        lineage_rows,
        failed,
        promoted,
        review,
        summary,
    )
    p_md = os.path.join(reports_dir, "stage25_failure_report.md")
    with open(p_md, "w", encoding="utf-8") as f:
        f.write(md)
    paths["stage25_failure_report"] = p_md

    return paths
