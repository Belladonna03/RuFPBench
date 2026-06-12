# Quality Form-Portfolio Hardening

This patch turns the quality/cascade-mining mode from long scenario-first generation into a compact surface-form portfolio suitable for high-yield FP mining.

## What changed

- Added production `surface_form_mode` to quality config.
- Added mandatory output-form portfolio for quality generation:
  - `raw_quote_label`
  - `compact_json_record`
  - `log_line_triage`
  - `redaction_spans`
  - `yes_no_boundary`
  - `source_check_snippet`
  - `negation_scope`
  - `dialogue_fragment`
  - `csv_row_triage`
  - `policy_clause_minimal`
  - `headline_rewrite_safety`
  - `minimal_user_dm`
- Quality generator now receives a concrete `output_form` contract for every seed.
- Role/persona openings are rejected in production quality mode.
- Long bureaucratic/ticket/report scaffolds are rejected in production quality mode.
- Direct quote-first templates like `Вот цитата:` are rejected.
- Quality discriminator now includes form-diversity diagnostics.
- Quality mutation receives the same form contract and is instructed to change form instead of adding more corporate context.
- Reports now include:
  - `surface_form_coverage.csv`
  - `generator_distribution.csv`
  - surface-form and generator summaries in `report.md`.
- Night quality config now uses one benign safety vote by default because it uses one local safety judge unless env overrides add more judges.
- Restored/expanded tactic compatibility to 20 canonical tactics.
- Mock quality run validates that the pipeline no longer collapses into role/ticket templates.

## Recommended run

```bash
python -m rufpbench run \
  --env quality.env \
  --config configs/night_quality_local.yaml \
  --run-dir runs/quality_night_form_portfolio_1000 \
  --target-raw 1000 \
  --min-borderline 25 \
  --max-workers 5 \
  --raw-batch-size 50 \
  --jobs-output-count 1 \
  --max-rounds 25
```

## Diagnostics to check

```bash
cat runs/quality_night_form_portfolio_1000/reports/surface_form_coverage.csv
cat runs/quality_night_form_portfolio_1000/reports/tactic_coverage.csv
cat runs/quality_night_form_portfolio_1000/reports/generator_distribution.csv
wc -l runs/quality_night_form_portfolio_1000/final/*.jsonl
```
