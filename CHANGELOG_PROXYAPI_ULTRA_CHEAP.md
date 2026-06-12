# ProxyAPI ultra-cheap hot-path fix

- Replaced expensive `openai/gpt-5.4-mini` bulk generation with `openai/gpt-4o-mini`.
- Removed Gemini from default hot-path safety/target calls in `configs/proxyapi_cheap.yaml`.
- Set paid ProxyAPI retries and JSON parse retries to zero by default.
- Added `empty_content_as_refusal_finish_reasons` handling for `content_filter` target responses.
- Raised OpenRouter OSS scout token budget from 96 to 192 to reduce empty length finishes.
- Updated `proxy.env`, `scripts/proxyapi_probe.sh`, and `scripts/run_proxyapi_cheap.sh` defaults.
- Added tests for content-filter empty response handling and updated ProxyAPI cheap config tests.

Validation: `pytest -q` -> 38 passed.
