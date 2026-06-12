# RuFPBench v2 changes

Implemented the agreed evolutionary architecture on top of the existing project.

## Added

- `seed_bank.py`: structured native-RU seed bank loader.
- `tactics.py`: RU WildTeaming-style surface tactics loader.
- `sampler.py`: category+tactic+coverage+previous-stats GenerationJob sampler.
- `bucketizer.py`: observed bucket assignment from prompt safety and target responses.
- `router.py`: accept/repair/harden/clarify/drop routing.
- `mutations.py`: creates next-round mutation jobs from router decisions.
- `scoring.py`: borderline/hardness scoring.
- `evolution.py`: multi-round feedback-loop runner.
- `data/wildteaming_tactics_ru.yaml`.
- `data/category_tactic_compatibility.yaml`.
- `data/coverage_targets.yaml`.
- `data/native_ru_seed_bank.jsonl`.
- CLI `sample-jobs` command.

## Changed

- `runner.py` now delegates to the evolutionary pipeline.
- Generation is job-based: category + compatible tactics + intended distribution.
- Candidate validation is per-candidate; sibling groups are not dropped atomically.
- `safe_answered`, `safe_refused_borderline`, `unsafe_refused`, `unsafe_answered`, `quarantine` are assigned by observed validation results, not by intended labels.
- Failed candidates can be repaired or hardened into next-round jobs.
- Reports now include category, tactic, mutation and model refusal analytics.

## Verified

- `python -m compileall -q rufpbench`
- `python -m pytest -q`
- `python -m rufpbench smoke`
