"""Stage 3 final reporting: markdown summary, metrics JSON, dataset caveats."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Set

from ..policy.schema import Stage3Policy
from ..schemas import (
    HardSubsetRecord,
    QCLabelRecord,
    SplitAssignmentRecord,
    Stage3CandidateRecord,
    Stage3Summary,
)


def _count_distribution(rows: List[Stage3CandidateRecord], key: str) -> Dict[str, int]:
    c: Counter[str] = Counter()
    for r in rows:
        if key == "category":
            c[r.category] += 1
        elif key == "subtype":
            c[r.subtype or "_none_"] += 1
        elif key == "source_stage":
            ss = r.source_stages or []
            if not ss:
                c["unknown"] += 1
            elif len(ss) > 1:
                c["mixed"] += 1
            elif "stage25" in ss:
                c["stage25"] += 1
            elif "stage2" in ss:
                c["stage2"] += 1
            else:
                c[ss[0]] += 1
    return dict(sorted(c.items(), key=lambda x: (-x[1], x[0])))


def _repaired_split(rows: List[Stage3CandidateRecord]) -> Dict[str, int]:
    rep = 0
    non = 0
    for r in rows:
        if "repaired_accept" in set(r.source_statuses or []):
            rep += 1
        else:
            non += 1
    return {"repaired": rep, "non_repaired": non}


def build_probe_profile_summary(rows: List[Stage3CandidateRecord]) -> Dict[str, Any]:
    total_probes = 0
    refusal_hits = 0
    models: Counter[str] = Counter()
    for r in rows:
        pp = r.probe_profile or {}
        for pr in pp.get("probe_results") or []:
            if not isinstance(pr, dict):
                continue
            total_probes += 1
            lab = str(pr.get("response_label", "")).lower()
            if lab in ("refusal", "partial_refusal"):
                refusal_hits += 1
            m = str(pr.get("model_name") or "?")
            models[m] += 1
        for pr in pp.get("semantic_probe_results") or []:
            if not isinstance(pr, dict):
                continue
            total_probes += 1
            lab = str(pr.get("response_label", "")).lower()
            if lab in ("refusal", "partial_refusal"):
                refusal_hits += 1
            m = str(pr.get("model_name") or "?")
            models[m] += 1
    return {
        "total_probe_result_rows_across_pool": total_probes,
        "refusal_labeled_count": refusal_hits,
        "refusal_rate_in_probes": round(refusal_hits / total_probes, 6) if total_probes else 0.0,
        "models_frequency": dict(models.most_common(50)),
    }


def build_hard_subset_stats(hard: List[HardSubsetRecord]) -> Dict[str, Any]:
    if not hard:
        return {"count": 0, "mean_hard_score": None, "scores": []}
    scores = [h.hard_score for h in hard]
    return {
        "count": len(hard),
        "mean_hard_score": round(sum(scores) / len(scores), 4),
        "min_hard_score": round(min(scores), 4),
        "max_hard_score": round(max(scores), 4),
        "sample_why_hard": [h.why_hard[:200] for h in hard[:5]],
    }


def build_split_stats(
    splits: List[SplitAssignmentRecord],
    balanced: List[Stage3CandidateRecord],
    policy: Stage3Policy,
) -> Dict[str, Any]:
    by_id = {r.item_id: r for r in balanced}
    per_split: Dict[str, int] = defaultdict(int)
    fam_per_split: DefaultDict[str, Set[str]] = defaultdict(set)
    repaired_per: DefaultDict[str, int] = defaultdict(int)

    for s in splits:
        per_split[s.split] += 1
        fam_per_split[s.split].add(s.family_id or "_unknown")
        r = by_id.get(s.item_id)
        if r and "repaired_accept" in set(r.source_statuses or []):
            repaired_per[s.split] += 1

    # family -> set of splits (leakage if >1)
    fam_to_splits: DefaultDict[str, Set[str]] = defaultdict(set)
    for s in splits:
        fid = s.family_id or "_unknown"
        fam_to_splits[fid].add(s.split)
    leakage_families = [f for f, ss in fam_to_splits.items() if len(ss) > 1]

    cluster_to_splits: DefaultDict[str, Set[str]] = defaultdict(set)
    for s in splits:
        if s.cluster_id:
            cluster_to_splits[s.cluster_id].add(s.split)
    leakage_clusters = [c for c, ss in cluster_to_splits.items() if len(ss) > 1]

    return {
        "per_split_row_counts": dict(per_split),
        "unique_families_per_split": {k: len(v) for k, v in fam_per_split.items()},
        "repaired_rows_per_split": dict(repaired_per),
        "family_across_multiple_splits": sorted(leakage_families),
        "cluster_across_multiple_splits": sorted(leakage_clusters),
        "leakage_note": (
            "No family/cluster should span splits when split_export enforcement is on. "
            "Non-empty lists indicate policy violation or legacy run without constraints."
        ),
        "policy_enforce_family": policy.split_export.enforce_single_split_per_family,
        "policy_enforce_cluster": policy.split_export.enforce_single_split_per_dedup_cluster,
    }


def _write_json(path: str, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_stage3_analysis_bundle(
    base_dir: str,
    paths: Dict[str, str],
    summary: Stage3Summary,
    policy: Stage3Policy,
    pool: List[Stage3CandidateRecord],
    qc_labels: List[QCLabelRecord],
    dedup_cluster_record_count: int,
    balanced_pool: List[Stage3CandidateRecord],
    hard_rows: List[HardSubsetRecord],
    splits: List[SplitAssignmentRecord],
    node_metrics: Dict[str, Dict[str, Any]],
) -> None:
    """Writes ``stage3_final_report.md``, ``metrics/*.json``, and enriched ``stage3_summary.json``."""
    base = Path(base_dir)
    mdir = base / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)

    qc_total = len(qc_labels)
    qc_pass = sum(1 for q in qc_labels if q.qc_pass)
    qc_fail = qc_total - qc_pass

    cat_bal = _count_distribution(balanced_pool, "category")
    sub_bal = _count_distribution(balanced_pool, "subtype")
    src_bal = _count_distribution(balanced_pool, "source_stage")
    bf = _balance_fairness(cat_bal)
    hs = build_hard_subset_stats(hard_rows)
    ss = build_split_stats(splits, balanced_pool, policy)

    _write_json(str(mdir / "category_distribution.json"), cat_bal)
    _write_json(str(mdir / "subtype_distribution.json"), sub_bal)
    _write_json(str(mdir / "source_stage_distribution.json"), src_bal)
    _write_json(str(mdir / "probe_profile_summary.json"), build_probe_profile_summary(balanced_pool))
    _write_json(str(mdir / "hard_subset_stats.json"), hs)
    _write_json(str(mdir / "split_stats.json"), ss)

    extended = {
        "run_id": summary.run_id,
        "candidate_pool_rows": summary.pool_size,
        "qc": {
            "labels_total": qc_total,
            "pass": qc_pass,
            "fail_or_drop": qc_fail,
            "note": "Rows with qc_pass=false include exact/near duplicates and QC failures.",
        },
        "dedup_cluster_records": dedup_cluster_record_count,
        "balanced_pool_rows": summary.balanced_size,
        "balance_fairness": bf,
        "repaired_vs_non_repaired_balanced": _repaired_split(balanced_pool),
        "node_metrics": node_metrics,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_out = _merge_summary_model(summary, extended)
    _write_json(paths.get("summary", str(base / "stage3_summary.json")), summary_out)

    report_path = paths.get("report", str(base / "stage3_final_report.md"))
    write_rich_final_report(
        report_path,
        summary=summary,
        policy=policy,
        extended=extended,
        cat_dist=cat_bal,
        hard_stats=hs,
        split_stats=ss,
        qc_fail=qc_fail,
    )


def _merge_summary_model(summary: Stage3Summary, extended: Dict[str, Any]) -> Dict[str, Any]:
    d = summary.model_dump(mode="json")
    d.update({k: v for k, v in extended.items() if k not in ("run_id",)})
    return d


def _balance_fairness(cat_bal: Dict[str, int]) -> Dict[str, Any]:
    if not cat_bal:
        return {"max_share": 0.0, "min_share": 0.0, "underrepresented": [], "dominant": []}
    total = sum(cat_bal.values())
    shares = {k: round(v / total, 4) for k, v in cat_bal.items()}
    mx = max(shares.values()) if shares else 0
    mn = min(shares.values()) if shares else 0
    thr_low = 1.0 / max(len(cat_bal), 1) * 0.3
    under = [k for k, s in shares.items() if s < thr_low and cat_bal[k] > 0]
    dom = [k for k, s in shares.items() if s >= 0.4]
    return {
        "max_category_share": mx,
        "min_category_share": mn,
        "underrepresented_categories": sorted(under),
        "dominant_categories": sorted(dom, key=lambda k: -shares[k]),
    }


def write_rich_final_report(
    path: str,
    *,
    summary: Stage3Summary,
    policy: Stage3Policy,
    extended: Dict[str, Any],
    cat_dist: Dict[str, int],
    hard_stats: Dict[str, Any],
    split_stats: Dict[str, Any],
    qc_fail: int,
) -> None:
    lines = [
        f"# Stage 3 final report — `{summary.run_id}`",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}_",
        "",
        "## 1. Pipeline throughput",
        "",
        "| Stage | Count |",
        "|-------|-------|",
        f"| Candidate pool (merged) | **{summary.pool_size}** |",
        f"| After QC + dedup (survivors) | **{summary.qc_passed}** |",
        f"| Dropped at QC/dedup (labels with qc_pass=false) | **{qc_fail}** |",
        f"| Balanced pool | **{summary.balanced_size}** |",
        f"| Hard subset (selected) | **{summary.hard_subset_size}** |",
        f"| Dedup cluster records (multi-member groups) | **{summary.dedup_clusters}** |",
        "",
        "## 2. Benchmark balance (balanced pool)",
        "",
        "- **Category distribution** (see `metrics/category_distribution.json`).",
        f"- Max category share ≈ **{extended.get('balance_fairness', {}).get('max_category_share', 'n/a')}**.",
        f"- Dominant categories: `{extended.get('balance_fairness', {}).get('dominant_categories', [])}`.",
        f"- Possibly underrepresented: `{extended.get('balance_fairness', {}).get('underrepresented_categories', [])}`.",
        "",
        "## 3. Repaired vs non-repaired (balanced pool)",
        "",
        "```json",
        json.dumps(extended.get("repaired_vs_non_repaired_balanced", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## 4. Hard subset",
        "",
        "```json",
        json.dumps(hard_stats, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 5. Splits & leakage checks",
        "",
        "```json",
        json.dumps(
            {
                "per_split_row_counts": split_stats.get("per_split_row_counts"),
                "families_spanning_multiple_splits": split_stats.get("family_across_multiple_splits"),
                "clusters_spanning_multiple_splits": split_stats.get("cluster_across_multiple_splits"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        f"- Dev / test / review_holdout: **{summary.dev_size}** / **{summary.test_size}** / **{summary.review_holdout_size}**",
        "",
        "## 6. Caveats",
        "",
        "- Counts depend on Stage 2/2.5 inputs and `dry_run` caps.",
        "- Near/exact dedup and family locks are only as good as upstream cluster IDs and `family_id`.",
        "- Hard subset uses probe-derived refusal signals; empty probes may exclude rows unless policy allows.",
        "- **Leakage lists** above should be empty when `split_export` enforcement is enabled and the pipeline completed successfully.",
        "",
        "## 7. Node timing",
        "",
        "```json",
        json.dumps(extended.get("node_metrics", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## 8. Exports",
        "",
        f"- Export directory: `{summary.exports_dir}`",
        "- See also `export_manifest.json`, `metrics/*.json`.",
        "",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_final_report(path: str, summary: Stage3Summary, extra: Dict[str, Any]) -> None:
    """Legacy thin wrapper (kept for imports); prefer ``write_stage3_analysis_bundle`` from orchestrator."""
    lines = [
        f"# Stage 3 final report (`{summary.run_id}`)",
        "",
        "## Counts",
        f"- Candidate pool: **{summary.pool_size}**",
        f"- QC passed: **{summary.qc_passed}**",
        f"- Dedup clusters: **{summary.dedup_clusters}**",
        f"- Balanced pool rows: **{summary.balanced_size}**",
        f"- Hard subset: **{summary.hard_subset_size}**",
        f"- Dev / test / review_holdout: **{summary.dev_size}** / **{summary.test_size}** / **{summary.review_holdout_size}**",
        "",
        "## Exports",
        f"- Directory: `{summary.exports_dir}`",
        "",
        "## Extra",
        "```json",
        json.dumps(extra, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
