"""Draft ``dataset_card.md`` for benchmark / dataset release."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from ..policy.schema import Stage3Policy
from ..schemas import HardSubsetRecord, SplitAssignmentRecord, Stage3CandidateRecord


def _cat_counts(rows: List[Stage3CandidateRecord]) -> Dict[str, int]:
    c: Counter[str] = Counter()
    for r in rows:
        c[r.category] += 1
    return dict(sorted(c.items(), key=lambda x: (-x[1], x[0])))


def build_dataset_card_markdown(
    *,
    run_id: str,
    policy: Stage3Policy,
    balanced_pool: List[Stage3CandidateRecord],
    splits: List[SplitAssignmentRecord],
    hard_rows: List[HardSubsetRecord],
    export_meta: Dict[str, Any],
) -> str:
    """Concise dataset card: goal, FP/over-refusal focus, stages, splits, caveats."""
    cats = _cat_counts(balanced_pool)
    top_cats = list(cats.items())[:12]
    hc = export_meta.get("counts") or {}
    dev, test, hold = hc.get("dev"), hc.get("test"), hc.get("review_holdout")
    hard_n = hc.get("hard_subset_export_rows", len({h.item_id for h in hard_rows}))

    split_fr = policy.split.model_dump()
    hard_cfg = policy.hard_subset.model_dump()

    lines = [
        f"# Dataset card (draft) — Stage 3 run `{run_id}`",
        "",
        "## Purpose",
        "",
        "This release is a **false-positive / over-refusal** oriented benchmark slice built from "
        "RuFP Stage 2 (semantic validation + probes) and optional Stage 2.5 (repair loop), consolidated in **Stage 3** "
        "with QC, deduplication, balancing, and leakage-aware splits.",
        "",
        "## False-positive & over-refusal focus",
        "",
        "- **False-positive (FP) risk**: prompts that are semantically acceptable but may be mis-scored or "
        "borderline in safety pipelines; Stage 2 labels (safety / naturalness / borderline) are preserved in lineage.",
        "- **Over-refusal focus**: items with **probe evidence** (refusal / partial_refusal signals across models) "
        "surface in scoring and in the **hard subset** export — a stress slice for models that refuse too often on valid prompts.",
        f"- Hard-subset policy (excerpt): enabled={hard_cfg.get('enabled')}, fraction≈{hard_cfg.get('fraction')}, "
        f"min_distinct_models_with_refusal={hard_cfg.get('min_distinct_models_with_refusal')}.",
        "",
        "## Pipeline stages (overview)",
        "",
        "| Stage | Role |",
        "|-------|------|",
        "| **Stage 1** | Family / prompt identity (optional `family_to_prompt_map`) |",
        "| **Stage 2** | Semantic labels + probe positives + refusal probes |",
        "| **Stage 2.5** | Optional repair promotion & lineage |",
        "| **Stage 3** | Merge → QC/dedup/cluster → balance/slices → hard subset → splits/export |",
        "",
        "## Categories (balanced pool)",
        "",
        f"- Distinct categories: **{len(cats)}**",
        f"- Top categories (count): {', '.join(f'`{k}` ({v})' for k, v in top_cats) if top_cats else '—'}",
        "",
        "## Splits",
        "",
        f"- **dev** / **test** / **review_holdout** rows: **{dev}** / **{test}** / **{hold}**",
        f"- Target fractions (policy): dev={split_fr.get('dev_fraction')}, test={split_fr.get('test_fraction')}, "
        f"remainder → review_holdout",
        f"- Leakage controls: single split per family={policy.split_export.enforce_single_split_per_family}, "
        f"per dedup cluster={policy.split_export.enforce_single_split_per_dedup_cluster}",
        "",
        "## Hard subset export",
        "",
        f"- Rows in hard-subset export bundle: **{hard_n}** (`hard_subset_benchmark.jsonl`).",
        "- Intended as a **high-signal** subset for over-refusal behaviour; not a random subsample of the benchmark.",
        "",
        "## Repair loop (Stage 2.5)",
        "",
        "- Prompts promoted through repair carry `stage25_repair` in `final_lineage.jsonl` and repaired/non-repaired "
        "balance hints during split assignment when enabled.",
        "",
        "## What is included",
        "",
        "- Rows that pass Stage 3 **QC** and survive **dedup** (see `stage3_qc_labels.jsonl`).",
        "- **Balanced pool** rows with a **split assignment** (`split_assignment.jsonl`).",
        "- Exported **dev/test/review_holdout** JSONL+CSV under `stage3_exports/`.",
        "- **`final_lineage.jsonl`**: machine-readable cross-stage trace.",
        "",
        "## What is excluded",
        "",
        "- QC failures, exact/near duplicates dropped at Stage 3 (see `qc_label` / `qc_pass` in lineage).",
        "- Stage 2 review queue / failed repair paths unless merged into the candidate pool upstream.",
        "- Items not selected into the balanced pool after caps and stratification.",
        "",
        "## Caveats",
        "",
        "- Near-duplicate and family constraints depend on upstream **cluster_id** / **family_id** quality.",
        "- Probe coverage varies; hard subset scores emphasize refusal signals where probes exist.",
        "- Category balance is **best-effort** under grouping constraints, not a strict quota.",
        "- `review_holdout` is for manual QA — **not** for tuning metrics you report as test performance.",
        "",
        "---",
        "",
        "_Generated as a draft; align wording with product/legal before external publication._",
        "",
    ]
    return "\n".join(lines)
