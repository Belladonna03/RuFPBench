# ProxyAPI ultra-cheap cost tuning

This profile runs the normal `cascade_mining` pipeline through ProxyAPI, but avoids the expensive bulk path observed in paid runs.

## What changed

- Removed `openai/gpt-5.4-mini` from bulk generation/rewrite. Bulk generation now uses `openai/gpt-4o-mini` with `openai/gpt-5.4-nano` as fallback.
- Removed Gemini from hot scout/safety path. `gemini/gemini-3.1-flash-lite` was useful but too expensive for a large funnel.
- Set retries to `0` for paid ProxyAPI steps. This prevents double billing on empty/length/content-filter responses.
- Increased OpenRouter OSS scout budget from 96 to 192 tokens, because 96 often ended with empty `finish_reason=length`.
- Treat empty `finish_reason=content_filter` as a refusal signal for target scout/final calls instead of retrying.
- Lowered full-validation pressure: fewer promoted candidates and fewer controls per round.
- Lowered `PROXYAPI_MAX_TOKENS_CAP` to 900 in `proxy.env` as a hard provider-level ceiling.

## Recommended commands

Stop the old paid run first with `Ctrl+C`.

Tiny smoke:

```bash
SKIP_PROBE=1 \
RUN_DIR=runs/proxyapi_tiny_ultra_001 \
TARGET_RAW=32 \
MIN_BORDERLINE=1 \
RAW_BATCH_SIZE=32 \
JOBS_OUTPUT_COUNT=8 \
MAX_ROUNDS=2 \
MAX_WORKERS=2 \
bash scripts/run_proxyapi_cheap.sh
```

Affordable run:

```bash
SKIP_PROBE=1 \
RUN_DIR=runs/proxyapi_ultra_cheap_001 \
TARGET_RAW=800 \
MIN_BORDERLINE=25 \
RAW_BATCH_SIZE=300 \
JOBS_OUTPUT_COUNT=24 \
MAX_ROUNDS=8 \
MAX_WORKERS=4 \
bash scripts/run_proxyapi_cheap.sh
```

If OpenRouter still returns many `finish_reason=length` empty outputs, either remove it from `RUFP_TARGET_MODELS` or increase only `target_response_scout.max_tokens` to 256 while keeping `retries: 0`.

## Current hot-path models

- Generator/rewrite: `openai/gpt-4o-mini`, fallback `openai/gpt-5.4-nano`
- Safety judge: `openai/gpt-5.4-nano`
- Targets: `openai/gpt-5.4-nano`, `openrouter/openai/gpt-oss-20b`

