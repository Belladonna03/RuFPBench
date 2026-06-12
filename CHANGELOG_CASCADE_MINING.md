# Cascade mining changelog

## Added

- New `cascade_mining` pipeline mode in `rufpbench/cascade.py`.
- Large-funnel generation with cheap QC and fast prompt filter.
- Short target scout pool with `target_response_scout.max_tokens=128`.
- GigaChat Max/Ultra exploration slice via `cascade_exploration_rate`.
- Promotion queue: only refusal-positive or high-friction candidates go to full validation.
- Full prompt-safety ensemble only for promoted candidates and small controls.
- Scout-response reuse during final validation.
- Model-specific hard subsets: `rufpbench_<model>_hard.jsonl`.
- Regex-first refusal classifier to avoid LLM judge calls for obvious refusals/compliance.
- Per-step `min_max_tokens`/`max_tokens_cap` route-only options to prevent global 8k token floors from inflating target calls.
- Tests for cascade promotion, fast filter, token overrides and mock end-to-end run.

## Kept

- Legacy `evolutionary` mode remains available.
- Existing bucketizer/reporting outputs remain compatible.

## Default target

- Config default aims at `8000` raw prompts and `500` strict `safe_refused_borderline` examples, with top-up to `16000` raw attempts if yield is low.
