# Scenario-first quality mining patch

This patch changes RuFPBench quality mode from quote/raw-fragment-first prompts to
FalseReject-style long realistic scenarios.

Key changes:

- `HARD_GENERATOR_SYSTEM` now requires scenario-first prompts, 900–1800 chars,
  with role/context/artifact/constraints/expected safe output.
- Quote-only prompts are explicitly discouraged; at most one short quote may be
  used as part of a broader scenario.
- Added deterministic `SCENARIO_BLUEPRINTS` rotation in `quality.py`:
  - workplace incident memo
  - evidence timeline / chain-of-custody
  - source attribution factcheck
  - compact schema with context
  - support/HR/safeguarding triage story
  - defensive analysis brief
- `QUALITY_MUTATOR_SYSTEM` now expands short quote-like parents into richer
  scenarios and uses feedback/target summaries to move near-miss cases toward
  harder but still benign prompts.
- Quality payloads now include `scenario_blueprint`; outputs keep
  `scenario_blueprint_id` in candidate metadata.
- Quality QC length window now supports longer prompts:
  - `min_chars: 120`
  - `max_chars: 2200`
- Quality target final/local response budgets for GLM are raised to 8192 tokens
  to avoid `finish_reason=length` empty responses.
- Quality target scout now sends ambiguous/mixed responses to the refusal judge;
  regex-only ambiguous labels can no longer inflate strict FP in quality mode.

Goal: produce diverse long Russian safe-but-sensitive scenarios closer to the
FalseReject mining style instead of repetitive quoted fragments.
