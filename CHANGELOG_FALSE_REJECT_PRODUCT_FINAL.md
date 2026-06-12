# FalseReject product-query final patch

- Replaced artifact-heavy surface forms with user-like query forms.
- Added query realism validator.
- Rejected field-only JSON/CSV/dataset dumps without natural user ask.
- Kept direct quote minimization.
- Updated generator/mutator prompts to produce realistic standalone user queries.
- Updated mock quality outputs to match product-query style.
- Updated quality.env.example to stable oss generation/discrimination defaults.
- Added FINAL_FALSE_REJECT_PIPELINE.md with final architecture and blockers.
