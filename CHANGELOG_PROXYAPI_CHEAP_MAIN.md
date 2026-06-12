# ProxyAPI cheap main-pipeline profile

## Changed

- Replaced the separate `proxyapi_pilot_cheap` profile with a normal main-pipeline profile: `configs/proxyapi_cheap.yaml`.
- Added `.env.proxyapi.example` for the main ProxyAPI run.
- Added `scripts/run_proxyapi_cheap.sh`, which calls the regular `rufpbench run --mode cascade_mining` command.
- Updated `scripts/proxyapi_probe.sh` to use `configs/proxyapi_cheap.yaml` and `.env.proxyapi`.
- Removed pilot-specific docs/scripts/configs to avoid a second pipeline path.

## Model profile

- Generation/rewrite: `openai/gpt-5.4-mini`, `gemini/gemini-3.1-flash-lite`.
- Cheap critic/judges: `openai/gpt-5.4-nano`.
- Safety ensemble: `openai/gpt-5.4-nano`, `gemini/gemini-3.1-flash-lite`.
- Scout/final targets: `openai/gpt-5.4-nano`, `gemini/gemini-3.1-flash-lite`, `openrouter/openai/gpt-oss-20b`.

## Speed/cost guards

- Scout target max output: 96 tokens.
- Refusal judge max output: 120 tokens.
- Safety judge max output: 160 tokens.
- Final target max output: 192 tokens.
- Full validation still only runs after scout refusal/friction promotion.


## Proxy env template fix
- Added root-level `proxy.env` so scripts can run without copying the example first.
- `scripts/run_proxyapi_cheap.sh` and `scripts/proxyapi_probe.sh` now default to `ENV_FILE=proxy.env`.
