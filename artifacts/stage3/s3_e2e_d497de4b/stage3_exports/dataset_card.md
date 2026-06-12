# Dataset card (draft) — Stage 3 run `s3_e2e_d497de4b`

## Purpose

This release is a **false-positive / over-refusal** oriented benchmark slice built from RuFP Stage 2 (semantic validation + probes) and optional Stage 2.5 (repair loop), consolidated in **Stage 3** with QC, deduplication, balancing, and leakage-aware splits.

## False-positive & over-refusal focus

- **False-positive (FP) risk**: prompts that are semantically acceptable but may be mis-scored or borderline in safety pipelines; Stage 2 labels (safety / naturalness / borderline) are preserved in lineage.
- **Over-refusal focus**: items with **probe evidence** (refusal / partial_refusal signals across models) surface in scoring and in the **hard subset** export — a stress slice for models that refuse too often on valid prompts.
- Hard-subset policy (excerpt): enabled=True, fraction≈0.15, min_distinct_models_with_refusal=2.

## Pipeline stages (overview)

| Stage | Role |
|-------|------|
| **Stage 1** | Family / prompt identity (optional `family_to_prompt_map`) |
| **Stage 2** | Semantic labels + probe positives + refusal probes |
| **Stage 2.5** | Optional repair promotion & lineage |
| **Stage 3** | Merge → QC/dedup/cluster → balance/slices → hard subset → splits/export |

## Categories (balanced pool)

- Distinct categories: **1**
- Top categories (count): `unknown` (1)

## Splits

- **dev** / **test** / **review_holdout** rows: **0** / **0** / **1**
- Target fractions (policy): dev=0.7, test=0.2, remainder → review_holdout
- Leakage controls: single split per family=True, per dedup cluster=True

## Hard subset export

- Rows in hard-subset export bundle: **0** (`hard_subset_benchmark.jsonl`).
- Intended as a **high-signal** subset for over-refusal behaviour; not a random subsample of the benchmark.

## Repair loop (Stage 2.5)

- Prompts promoted through repair carry `stage25_repair` in `final_lineage.jsonl` and repaired/non-repaired balance hints during split assignment when enabled.

## What is included

- Rows that pass Stage 3 **QC** and survive **dedup** (see `stage3_qc_labels.jsonl`).
- **Balanced pool** rows with a **split assignment** (`split_assignment.jsonl`).
- Exported **dev/test/review_holdout** JSONL+CSV under `stage3_exports/`.
- **`final_lineage.jsonl`**: machine-readable cross-stage trace.

## What is excluded

- QC failures, exact/near duplicates dropped at Stage 3 (see `qc_label` / `qc_pass` in lineage).
- Stage 2 review queue / failed repair paths unless merged into the candidate pool upstream.
- Items not selected into the balanced pool after caps and stratification.

## Caveats

- Near-duplicate and family constraints depend on upstream **cluster_id** / **family_id** quality.
- Probe coverage varies; hard subset scores emphasize refusal signals where probes exist.
- Category balance is **best-effort** under grouping constraints, not a strict quota.
- `review_holdout` is for manual QA — **not** for tuning metrics you report as test performance.

---

_Generated as a draft; align wording with product/legal before external publication._
