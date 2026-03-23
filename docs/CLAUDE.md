# RuFPBench-MVP workspace guide

This repository is organized so each pipeline stage has a dedicated agent module, shared helpers, and thin entrypoints.

## Always start here

1. Read `README.md` for the public project contract.
2. Prefer editing **agent logic** in `agents/` and **orchestration** in `pipeline/` instead of duplicating code in entry scripts.
3. Shared configuration, paths, and LLM wiring live under `shared/`.
4. CLI implementation lives in `cli/`; root `run_pipeline.py` / `run_agent.py` are thin wrappers.

## Agent map

| Agent | Module |
|-------|--------|
| `DataCollectionAgent` | `agents/data_collection_agent.py` |
| `BorderlineRewriteAgent` | `agents/rewrite_agent.py` |
| `DataQualityAgent` | `agents/data_quality_agent.py` |
| `AnnotationAgent` | `agents/annotation_agent.py` |
| `ActiveLearningAgent` | `agents/active_learning_agent.py` |

## Ground rules

- Import agent classes from the `agents` package (e.g. `from agents.data_quality_agent import DataQualityAgent`).
- The project supports both:
  - **full pipeline run**: `python run_pipeline.py --config config.yaml`
  - **single-step debug**: `python run_agent.py --agent <name> ...`
- Store intermediate tables on disk with deterministic paths from `config.yaml`.
- If a step pauses for HITL, write a clear artifact and stop instead of silently continuing.
- Do not mix orchestration code with agent logic inside the same functions when a split keeps debugging easier.

## Debug defaults

- Collection writes to `data/raw/`
- Rewrite and quality write to `data/interim/`
- Annotation and HITL write to `data/labeled/`
- Reports go to `reports/`
- Final model goes to `models/`
