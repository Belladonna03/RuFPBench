# Stage 3 final report — `s3_cli_d41ee785`

_Generated: 2026-04-20T15:30:04.664695+00:00_

## 1. Pipeline throughput

| Stage | Count |
|-------|-------|
| Candidate pool (merged) | **10** |
| After QC + dedup (survivors) | **1** |
| Dropped at QC/dedup (labels with qc_pass=false) | **9** |
| Balanced pool | **1** |
| Hard subset (selected) | **0** |
| Dedup cluster records (multi-member groups) | **1** |

## 2. Benchmark balance (balanced pool)

- **Category distribution** (see `metrics/category_distribution.json`).
- Max category share ≈ **1.0**.
- Dominant categories: `['unknown']`.
- Possibly underrepresented: `[]`.

## 3. Repaired vs non-repaired (balanced pool)

```json
{
  "repaired": 1,
  "non_repaired": 0
}
```

## 4. Hard subset

```json
{
  "count": 0,
  "mean_hard_score": null,
  "scores": []
}
```

## 5. Splits & leakage checks

```json
{
  "per_split_row_counts": {
    "review_holdout": 1
  },
  "families_spanning_multiple_splits": [],
  "clusters_spanning_multiple_splits": []
}
```

- Dev / test / review_holdout: **0** / **0** / **1**

## 6. Caveats

- Counts depend on Stage 2/2.5 inputs and `dry_run` caps.
- Near/exact dedup and family locks are only as good as upstream cluster IDs and `family_id`.
- Hard subset uses probe-derived refusal signals; empty probes may exclude rows unless policy allows.
- **Leakage lists** above should be empty when `split_export` enforcement is enabled and the pipeline completed successfully.

## 7. Node timing

```json
{
  "build_pool": {
    "duration_ms": 2.705,
    "status": "ok",
    "error": null,
    "outputs": {
      "candidate_pool_rows": 10
    }
  },
  "qc_dedup_cluster": {
    "duration_ms": 0.329,
    "status": "ok",
    "error": null,
    "outputs": {
      "qc_label_rows": 10,
      "dedup_cluster_records": 1,
      "survivor_rows": 1
    }
  },
  "balance": {
    "duration_ms": 0.486,
    "status": "ok",
    "error": null,
    "outputs": {
      "balanced_pool_rows": 1
    }
  },
  "hard_subset": {
    "duration_ms": 0.129,
    "status": "ok",
    "error": null,
    "outputs": {
      "hard_subset_rows": 0
    }
  },
  "split_builder_exporter": {
    "duration_ms": 4.707,
    "status": "ok",
    "error": null,
    "outputs": {
      "split_assignment_rows": 1,
      "dev": 0,
      "test": 0,
      "review_holdout": 1
    }
  }
}
```

## 8. Exports

- Export directory: `artifacts/stage3/s3_cli_d41ee785/stage3_exports`
- See also `export_manifest.json`, `metrics/*.json`.
