# Dataset card (draft)

This benchmark split was produced by RuFP Stage 3 (`split_builder_exporter`).

## Composition
- Balanced pool size: **1**
- dev / test / review_holdout: **0** / **0** / **1**
- Hard subset export rows: **0** (separate JSONL/CSV; over-refusal stress slice)

## Leakage controls
- Single split per family: **True**
- Single split per deduplication cluster: **True**

## Caveats
- Splits respect declared dedup clusters from Stage 3 Node 2; near/exact clusters should not span dev+test when enforcement is on.
- Category/subtype balance is best-effort at the **constraint-group** level; tiny families may skew counts.
- Repaired vs original balance is a soft ordering hint when enabled; quotas remain approximate.

## Schema
- Primary rows: `Stage3CandidateRecord` JSONL (see repo `schemas.py`).
