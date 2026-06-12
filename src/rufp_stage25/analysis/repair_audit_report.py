"""
Post-run audit metrics for Stage 2.5: no-improvement loops, failure patterns, unstable categories.

Reads repair_lineage.jsonl and writes JSON + markdown snippets for reports/.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List


def load_lineage(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_audit_metrics(lineage_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    no_improvement = [r for r in lineage_rows if r.get("final_repair_decision") == "no_improvement"]
    by_reason = Counter(r.get("repair_reason", "unknown") for r in lineage_rows)
    by_decision = Counter(r.get("final_repair_decision", "unknown") for r in lineage_rows)
    by_cat_decision: Dict[str, Counter] = defaultdict(Counter)
    for r in lineage_rows:
        cat = r.get("category", "unknown")
        by_cat_decision[cat][r.get("final_repair_decision", "unknown")] += 1

    # Unstable categories: high rate of no_improvement or mixed decisions
    unstable: List[Dict[str, Any]] = []
    cats = set(r.get("category", "unknown") for r in lineage_rows)
    for cat in cats:
        sub = [r for r in lineage_rows if r.get("category") == cat]
        if not sub:
            continue
        n = len(sub)
        ni = sum(1 for r in sub if r.get("final_repair_decision") == "no_improvement")
        unstable.append(
            {
                "category": cat,
                "count": n,
                "no_improvement_rate": round(ni / n, 4),
                "decision_mix": dict(Counter(r.get("final_repair_decision") for r in sub)),
            }
        )
    unstable.sort(key=lambda x: (-x["no_improvement_rate"], -x["count"]))

    repeated_failures = Counter(
        (r.get("repair_reason"), r.get("final_repair_decision")) for r in lineage_rows
    )

    return {
        "counts": {
            "total_lineage_records": len(lineage_rows),
            "no_improvement": len(no_improvement),
        },
        "by_repair_reason": dict(by_reason),
        "by_final_decision": dict(by_decision),
        "repeated_failure_patterns": {f"{a}|{b}": c for (a, b), c in repeated_failures.items() if c > 1},
        "unstable_categories": unstable[:50],
        "repair_loops_no_improvement_sample": [
            {
                "repaired_prompt_id": r.get("repaired_prompt_id"),
                "original_prompt_id": r.get("original_prompt_id"),
                "repair_reason": r.get("repair_reason"),
            }
            for r in no_improvement[:20]
        ],
    }


def write_audit_artifacts(run_id: str, lineage_path: str, reports_dir: str) -> None:
    os.makedirs(reports_dir, exist_ok=True)
    rows = load_lineage(path=lineage_path) if os.path.isfile(lineage_path) else []
    metrics = build_audit_metrics(rows)
    metrics_path = os.path.join(reports_dir, "repair_audit_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    md_path = os.path.join(reports_dir, "repair_audit_digest.md")
    lines = [
        f"# Stage 2.5 failure / audit report (`{run_id}`)",
        "",
        "## Summary",
        f"- Total lineage records: **{metrics['counts']['total_lineage_records']}**",
        f"- No improvement (repair_changed=false path): **{metrics['counts']['no_improvement']}**",
        "",
        "## Repeated failure patterns (reason × decision, count>1)",
        "```json",
        json.dumps(metrics.get("repeated_failure_patterns", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Unstable categories (high no_improvement rate)",
        "```json",
        json.dumps(metrics.get("unstable_categories", [])[:15], ensure_ascii=False, indent=2),
        "```",
        "",
        "## By final decision",
        "```json",
        json.dumps(metrics.get("by_final_decision", {}), ensure_ascii=False, indent=2),
        "```",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
