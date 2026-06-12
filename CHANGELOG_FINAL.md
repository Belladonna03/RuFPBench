# Final pipeline patch changelog

## Closed issues

1. **Generator diversity**
   - `RUFP_GENERATOR_MODELS` and `RUFP_REWRITER_MODELS` are now active.
   - Generation/rewrite/mutation jobs select a model deterministically per job.
   - Stronger/non-local JSON generators use single-item JSON mode to avoid array truncation.

2. **ProxyAPI routing**
   - Prefix routing for `openai/`, `anthropic/`, `gemini/`, `openrouter/` models.
   - ProxyAPI provider defaults to `https://openai.api.proxyapi.ru/v1`.
   - `PROXYAPI_MAX_TOKENS_CAP` prevents accidental high output-token budgets.
   - Optional `PROXYAPI_API_MODE=responses` support for models that require Responses API.

3. **Metrics correctness**
   - Added `benign_prompt_unsafe_response` bucket.
   - `extra_safety_guidance` and `format_deviation` are no longer counted as refusals.
   - `model_refusal_rates.csv` now reports errors, error_rate, usable_total, extra_safety_guidance, and format_deviation.

4. **Run archive safety**
   - `config.effective.json` and report config JSON redact secret-bearing keys.
   - `.env` is not included in the released archive.

5. **Diversity tactics**
   - Surface tactics expanded from 10 to 14 with targeted additions:
     - `redaction_preserve_violation`
     - `triage_escalation_label`
     - `minimal_context_raw_fragment`
     - `policy_clause_mapping`

6. **Unsafe controls**
   - Prompt rules now explicitly prevent unsafe controls from being disguised as benign moderation/classification tickets.

## Verification

```text
PYTHONPATH=. python -m pytest -q
18 passed

python -m compileall -q rufpbench tests
OK
```
