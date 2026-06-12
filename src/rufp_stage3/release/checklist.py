"""Release checklist markdown for Stage 3 drops."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def build_release_checklist_markdown(
    *,
    run_id: str,
    exports_dir: Path,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Practical pre-flight list for publishing a Stage 3 benchmark drop."""
    x = extra or {}
    policy_ok = x.get("policy_lint_ok", "—")
    lines = [
        f"# Release checklist — Stage 3 `{run_id}`",
        "",
        "Use this list before tagging a public / frozen benchmark version.",
        "",
        "## Artifact completeness",
        "",
        f"- [ ] Run directory exists: `artifacts/stage3/{run_id}/`",
        "- [ ] `stage3_balanced_pool.jsonl` present and non-empty",
        "- [ ] `split_assignment.jsonl` present",
        "- [ ] `stage3_exports/benchmark_{dev,test,review_holdout}.jsonl` present",
        "- [ ] `stage3_exports/final_lineage.jsonl` present (cross-stage trace)",
        "- [ ] `stage3_exports/dataset_card.md` reviewed",
        "- [ ] `stage3_exports/metadata.json` matches export row counts",
        "- [ ] `export_manifest.json` lists all shipped files + checksums where applicable",
        "",
        "## Quality & leakage",
        "",
        "- [ ] Spot-check **empty** `family_across_multiple_splits` / `cluster_across_multiple_splits` in `metrics/split_stats.json`",
        "- [ ] Confirm **review_holdout** not used in any training / early stopping for reported numbers",
        "- [ ] Re-run or spot-check QC: random sample of `qc_pass=false` rows are true drops",
        "",
        "## Reproducibility",
        "",
        "- [ ] `stage3_policy_resolved.json` archived with the run",
        "- [ ] `stage3_run_manifest.json` has `started_at` / `finished_at` and node timings",
        f"- [ ] Policy / config lint: {policy_ok}",
        "",
        "## Documentation",
        "",
        "- [ ] Dataset card purpose + FP/over-refusal wording approved",
        "- [ ] Caveats section matches known pipeline limits for this drop",
        "- [ ] Version string / DOI / internal ticket linked in release notes (if applicable)",
        "",
        "## Publishing",
        "",
        f"- [ ] Bundle `stage3_exports/` (path: `{exports_dir}`) for distribution",
        "- [ ] Optional: attach `stage3_final_report.md` + `metrics/*.json` for auditors",
        "",
        "## Sign-off",
        "",
        "- [ ] Owner: _______________  Date: _______________",
        "",
    ]
    return "\n".join(lines)
