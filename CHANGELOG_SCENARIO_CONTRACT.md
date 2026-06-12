# Scenario-first contract hardening

This patch closes the quote-only / short-fragment failure mode in quality mining.

## Why

Previous quality generations could still collapse into prompts like:

```text
Вот цитата: «...». Это угроза? Верни JSON.
```

Those are easy to generate, repetitive, and unlike FalseReject-style long
realistic scenarios.

## Changes

- Added a shared `SCENARIO_FIRST_QUALITY_CONTRACT` to quality generator,
  discriminator, and mutator system prompts.
- Quality generator now requires a multi-layer user scenario with:
  - user/organization role;
  - real context;
  - artifact/document/log/policy/timeline;
  - uncertainty or conflict;
  - bounded safe output format;
  - at least three safe requested operations.
- Quote/raw fragments are hard-limited: they may appear only as short evidence
  fields inside a broader scenario.
- Added strict scenario-first gates in `quality.py`:
  - minimum scenario length: 900 characters;
  - minimum scenario signals: 5;
  - quote ratio cap: 10%;
  - at most one quoted span;
  - minimum requested operations: 3.
- Added structural rejection logging to `quality/structural_rejects.jsonl` with
  scenario metrics and prompt previews.
- Added tests that reject quote-only prompts and accept multi-layer scenarios.

## Expected effect

The quality miner should spend target calls on rich, realistic, FalseReject-style
cases rather than short quote/classification fragments.
