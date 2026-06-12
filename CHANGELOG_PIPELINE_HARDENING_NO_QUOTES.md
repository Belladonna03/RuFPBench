# Pipeline hardening: scenario-first / no direct quotes

This patch fixes the quality-mining pipeline after the no-quotes/scenario-first debugging cycle.

## Key fixes

- `SCENARIO_FIRST_QUALITY_CONTRACT` is now actually interpolated into quality generator / discriminator / mutator / structural-repair prompts.
- Added hard no-direct-quotes enforcement in `quality.py`:
  - angle quotes are rejected;
  - raw_text / quote / quoted_text / exact_phrase style fields are rejected;
  - direct phrase markers such as “пользователь написал”, “вот цитата”, “точная фраза” are rejected;
  - JSON schema keys are still allowed.
- Added missing `QualityConfig` fields so YAML settings are no longer silently ignored:
  - `forbid_direct_quotes`;
  - `max_scenario_chars`;
  - `enable_structural_repair`.
- Added `quality_structural_repair` stage before discriminator/targets. It tries one repair pass for structurally broken outputs instead of wasting whole seed iterations.
- Production `configs/night_quality_local.yaml` now pins structured JSON-critical quality steps to `oss`, not `minimax-m2.7`:
  - generator;
  - mutator;
  - discriminator;
  - response generator;
  - safety/refusal judges.
- GigaChat/GLM remain target models only.
- Mock quality data was updated to obey no-direct-quotes policy.
- Tests were updated to treat quote-bearing scenarios as rejected and no-quote long scenarios as valid.

## Why

Previous runs had three architectural issues:

1. The no-quotes contract was present in source but not interpolated into several system prompts.
2. `forbid_direct_quotes` existed in YAML but was not part of `QualityConfig`, so config loading ignored it.
3. The pipeline relied on prompt wording but had no hard structural gate for direct quotes.

The new version enforces no-quotes at both prompt and code level.

## Suggested probe

```bash
python -m rufpbench run \
  --env .quality.env \
  --config configs/night_quality_local.yaml \
  --run-dir runs/quality_noquotes_clean_probe_300 \
  --target-raw 300 \
  --min-borderline 3 \
  --max-workers 5 \
  --raw-batch-size 50 \
  --jobs-output-count 1 \
  --max-rounds 10
```

Check:

```bash
grep -R "«\|»\|raw_text\|quoted_text\|точная фраза\|пользователь написал" runs/quality_noquotes_clean_probe_300/raw runs/quality_noquotes_clean_probe_300/final | head
cat runs/quality_noquotes_clean_probe_300/state.json
wc -l runs/quality_noquotes_clean_probe_300/quality/*.jsonl runs/quality_noquotes_clean_probe_300/final/*.jsonl
```
