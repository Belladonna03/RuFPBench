# ProxyAPI cheap profile for the main cascade pipeline

This profile is a normal `cascade_mining` run, not a separate pilot pipeline.
It uses the same RuFPBench stages as `configs/default.yaml`:

```text
large generation funnel
→ QC + dedupe
→ fast prompt filter
→ short target scout
→ promotion queue
→ full safety ensemble only for promoted candidates + small controls
→ final target validation
→ final JSONL subsets and reports
```

The only difference is model routing: all paid calls go through ProxyAPI with relatively cheap models and strict token caps.

## Files

```text
configs/proxyapi_cheap.yaml      # main-pipeline ProxyAPI profile
.env.proxyapi.example            # env template
scripts/proxyapi_probe.sh        # connectivity check
scripts/run_proxyapi_cheap.sh    # normal rufpbench run wrapper
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

cp .env.proxyapi.example proxy.env
# edit proxy.env and set PROXYAPI_API_KEY
```

ProxyAPI OpenAI-compatible base URL is already configured:

```bash
PROXYAPI_BASE_URL=https://openai.api.proxyapi.ru/v1
```

## Probe models

```bash
scripts/proxyapi_probe.sh
```

The probe checks:

```text
openai/gpt-5.4-nano
gemini/gemini-3.1-flash-lite
openai/gpt-5.4-mini
openrouter/openai/gpt-oss-20b
```

If OpenRouter is unavailable in your account:

```bash
PROBE_OPENROUTER=0 scripts/proxyapi_probe.sh
```

Then remove `openrouter/openai/gpt-oss-20b` from:

```text
configs/proxyapi_cheap.yaml:
  models.target_models
  llm.pipeline.target_response_scout.models
  llm.pipeline.target_response_final.models
  llm.pipeline.target_response.models
proxy.env:
  RUFP_TARGET_MODELS
```

## Tiny paid end-to-end check

Use this first to verify the whole main pipeline with a very small budget:

```bash
SKIP_PROBE=1 \
RUN_DIR=runs/proxyapi_tiny_001 \
TARGET_RAW=32 \
MIN_BORDERLINE=1 \
RAW_BATCH_SIZE=32 \
JOBS_OUTPUT_COUNT=4 \
MAX_ROUNDS=2 \
MAX_WORKERS=2 \
scripts/run_proxyapi_cheap.sh
```

## Recommended cheap test run

```bash
scripts/run_proxyapi_cheap.sh
```

Equivalent explicit command:

```bash
python -m rufpbench run \
  --config configs/proxyapi_cheap.yaml \
  --env proxy.env \
  --run-dir runs/proxyapi_cheap_001 \
  --mode cascade_mining \
  --target-raw 800 \
  --min-borderline 50 \
  --raw-batch-size 200 \
  --jobs-output-count 16 \
  --max-rounds 8 \
  --max-workers 4
```

## Scaling toward 500+ FP borderline prompts

If the cheap run has acceptable yield, scale the same main pipeline gradually:

```bash
SKIP_PROBE=1 \
RUN_DIR=runs/proxyapi_cheap_2k \
TARGET_RAW=2000 \
MIN_BORDERLINE=150 \
RAW_BATCH_SIZE=300 \
JOBS_OUTPUT_COUNT=18 \
MAX_ROUNDS=10 \
MAX_WORKERS=4 \
scripts/run_proxyapi_cheap.sh
```

For the 500+ goal:

```bash
SKIP_PROBE=1 \
RUN_DIR=runs/proxyapi_cheap_500fp \
TARGET_RAW=8000 \
MIN_BORDERLINE=500 \
RAW_BATCH_SIZE=500 \
JOBS_OUTPUT_COUNT=20 \
MAX_ROUNDS=14 \
MAX_WORKERS=4 \
scripts/run_proxyapi_cheap.sh
```

If the refusal yield is low, do not immediately increase final target pool. First increase raw funnel size and mutation rounds:

```bash
TARGET_RAW=12000
MAX_ROUNDS=16
```

## Cost controls

The main cost controls live in `configs/proxyapi_cheap.yaml`:

```yaml
target_response_scout:
  max_tokens: 96
  max_tokens_cap: 96
refusal_judge_fast:
  max_tokens: 120
  max_tokens_cap: 120
prompt_safety_judge:
  max_tokens: 160
  max_tokens_cap: 160
target_response_final:
  max_tokens: 192
  max_tokens_cap: 192
```

Keep these caps small. The cascade pipeline only needs short scout answers to detect refusal.

If you see 429/5xx errors, lower both:

```bash
MAX_WORKERS=2
```

and in `configs/proxyapi_cheap.yaml`:

```yaml
llm:
  providers:
    proxyapi:
      concurrency: 2
```

## Outputs

```text
runs/<run>/state.json
runs/<run>/reports/report.md
runs/<run>/reports/model_refusal_rates.csv
runs/<run>/validation/promotion_reasons.jsonl
runs/<run>/validation/fast_prompt_filter.jsonl
runs/<run>/responses/scout_target_responses.<model>.jsonl
runs/<run>/responses/target_responses.final.<model>.jsonl
runs/<run>/final/rufpbench_borderline.jsonl
runs/<run>/final/rufpbench_hard.jsonl
runs/<run>/final/rufpbench_cross_model_hard.jsonl
runs/<run>/final/rufpbench_<model>_hard.jsonl
runs/<run>/final/all_labeled.jsonl
```

Quick checks:

```bash
cat runs/proxyapi_cheap_001/state.json | python -m json.tool
wc -l runs/proxyapi_cheap_001/final/rufpbench_borderline.jsonl
head -n 3 runs/proxyapi_cheap_001/final/rufpbench_borderline.jsonl
```
