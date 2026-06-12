# Quote minimization patch

This patch makes direct quotation a rare/mostly rejected surface in quality mode.

Why:
- Full quoted fragments were collapsing the dataset into quote-classification prompts.
- The desired funnel should produce paraphrased, structured, and artifact-summary prompts instead of `«...». classify` patterns.

Changes:
- `configs/night_quality_local.yaml` sets `forbid_direct_quotes: true`, `max_quote_ratio: 0.12`, `max_quoted_spans: 1`.
- `QualityConfig` defaults are aligned with the no-full-quote production mode.
- Replaced `raw_quote_label` with `risk_signal_label`.
- Rewrote all `QUALITY_OUTPUT_FORMS` to use summaries/placeholders/typed spans instead of full quotes.
- `form_diversity_failure_reason` rejects direct quotes, quote dominance and too many quoted spans in form mode.
- Generator/mutator prompts explicitly forbid full direct quotes and `raw_text` / `quote` / `цитата` fields.

Preferred alternatives:
- `signal_summary`
- `claim_summary`
- `private_span_types`
- `evidence_type`
- placeholders such as `<ADDRESS>`, `<PHONE>`, `<HANDLE>`
- compact JSON/log/CSV summary rows
