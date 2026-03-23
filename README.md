# RuFPBench-MVP — Data Pipeline (Course Project)

End-to-end, reproducible data pipeline for **«Сбор и обработка данных»**: collect seed data, optional borderline rewrite, quality cleaning, weak auto-labeling with HITL, active-learning analysis, and a baseline classifier.

## Layout

| Path | Role |
|------|------|
| `agents/` | Five agents (business logic only): collection, rewrite, quality, annotation, active learning |
| `pipeline/` | Orchestration (`orchestrator.py`), IO, HITL merge, training, reporting |
| `shared/` | Config loading, paths, schema constants, utilities, `llm.py`, `logging_utils.py`, `mediawiki_wikitext.py` |
| `cli/` | CLI implementation (`run_pipeline.py`, `run_agent.py`) |
| `docs/CLAUDE.md` | Workspace notes for contributors (agents map, debug defaults) |
| `run_pipeline.py` | Thin wrapper at repo root; delegates to `cli/run_pipeline.py` |
| `run_agent.py` | Thin wrapper at repo root; delegates to `cli/run_agent.py` |
| `config.yaml` | Single source of truth for paths and hyperparameters |

## Logging

Console logging uses the standard library `logging` module. Configure the default **INFO** level or set verbosity:

```bash
python run_pipeline.py --config config.yaml --log-level INFO
python run_agent.py --agent quality --config config.yaml --log-level DEBUG
```

Log lines look like: `LEVEL [component] message` (e.g. `[pipeline]` for orchestration, `[agents.data_collection]` for an agent). **Secrets and API keys are never logged.**

## Install

```bash
python3 -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set secrets (e.g. `HF_TOKEN` for Hugging Face Hub, **`DATA_COLLECTION_PROXYAPI_API_KEY`** for collection-stage LLM EDA interpretation). On startup, `load_config()` loads the first `.env` found next to `config.yaml` (walking up parent directories) or `./.env` in the current working directory, so `datasets` / `huggingface_hub` receive `HF_TOKEN` via `os.environ`. You can also `export HF_TOKEN=...` in the shell.

## Run the full pipeline

```bash
python run_pipeline.py --config config.yaml
```

Stages (in order): **collection → rewrite (optional) → quality → annotation → HITL gate (optional stop) → merge corrections → AL batch export (optional) → AL curves (optional) → training (optional) → reports.**

If `hitl.human_mode = stop_if_missing` and `data/labeled/review_queue.csv` exists with rows but `data/labeled/review_queue_corrected.csv` is missing or incomplete, the pipeline **exits with code 2** and prints what to do next.

After editing the corrected queue:

```bash
python run_pipeline.py --config config.yaml
```

### Typical artifacts

- Raw / merged: `data/raw/merged_raw.parquet` (+ per-source files when enabled)
- **Collection EDA** (after merge): `reports/collection_eda/` — tables and plots (`class_distribution*.csv/png`, `source_distribution*.csv/png`, `text_length_*`, `top20_words*.csv/png`, `source_label_crosstab.csv`, `eda_summary.md`) are **computed in code**. **`eda_llm_summary.md`** is optional text interpretation via ProxyAPI (`collection.llm` in `config.yaml`); if the API is unavailable, the collection step still finishes and code-based EDA remains.
- Interim: `data/interim/rewrite.parquet`, `data/interim/clean.parquet`
- Labeled: `data/labeled/auto_labeled.parquet`, `data/labeled/final_dataset.parquet`, review queues
- Reports: `reports/final_report.md`, `reports/quality_prescan.json`, optional `reports/learning_curve.png`
- Model (if `training.enabled`): `models/baseline_logreg.pkl`

## Debug one agent

```bash
python run_agent.py --agent collection --config config.yaml
python run_agent.py --agent rewrite   --config config.yaml --input data/raw/merged_raw.parquet
python run_agent.py --agent quality   --config config.yaml --input data/interim/rewrite.parquet
python run_agent.py --agent annotation --config config.yaml --input data/interim/clean.parquet
python run_agent.py --agent al        --config config.yaml --input data/labeled/final_dataset.parquet
```

Aliases: `data_collection`, `borderline_rewrite`, `data_quality`, `active_learning`.

## Imports (for notebooks)

```python
from agents.data_collection_agent import DataCollectionAgent
from agents.data_quality_agent import DataQualityAgent
from agents.annotation_agent import AnnotationAgent
from agents.active_learning_agent import ActiveLearningAgent
from agents.rewrite_agent import BorderlineRewriteAgent
```

## Optional appendix (post-collection rewrite)

```bash
python agents/post_collection_appendix.py --config config.yaml \
  --input data/raw/merged_raw.parquet \
  --output data/raw/merged_with_appendix.parquet
```

Uses `rewrite` settings from `config.yaml` (`rewrite.enabled`, `rewrite.mode`, etc.).

## Hugging Face / API / MediaWiki collection

- HF sources need **internet** and the `datasets` package.
- HTTP API sources (e.g. Wiktionary `action=query`) need a proper **User-Agent**; set `WIKIMEDIA_CONTACT_EMAIL` (and optionally `WIKIMEDIA_USER_AGENT_APP`) in the environment — see `.env.example`. Do not put contact email in `config.yaml`.
- Wiktionary and other wikis are read via **`https://…/w/api.php`** (`type: api` for category lists, `type: mediawiki_page` for `action=parse` + wikitext). Wikitext parsing lives in `shared/mediawiki_wikitext.py`.
- For **rewrite** LLM calls, `config.yaml` uses **`REWRITE_AGENT_PROXYAPI_*`** (ProxyAPI OpenRouter endpoint, default base `https://api.proxyapi.ru/openrouter/v1`, default model `qwen/qwen3-8b`). Copy `.env.example` and set at least **`REWRITE_AGENT_PROXYAPI_API_KEY`**. Other agents are unchanged; global **`OPENROUTER_*`** / **`PROXYAPI_*`** still work when `rewrite.llm` does not override them (see `shared/llm.py`).
- Rewrite uses **`rewrite.runtime`** (429 backoff, optional `parallel_enabled` / `max_concurrency`, `inter_request_delay_s`, `debug_max_selected_rows`) and logs endpoint/model/`api_key_env` (never the key). **`openrouter/free`** remains rate-limited if you point `rewrite.llm` at it.

### Collection `sources` types

| `type` | Role |
|--------|------|
| `hf_dataset` | Hugging Face `datasets` |
| `api` | JSON HTTP API (`endpoint`, `params`, `records_path`, `text_field`, …). If `params` is `action=query` + `list=categorymembers` (or `api_mode: categorymembers`), the agent fetches **all** pages via MediaWiki `continue` pagination and uses `text_field` (e.g. `title`) on each member. |
| `mediawiki_page` | One MediaWiki page via `action=parse` + `prop=wikitext` (`params.page` or `params.title`), then `parse_mode` (e.g. `mediawiki_wikitext_list`) on `parse.wikitext["*"]`. Optional `wikitext_parser_config`. |

`DataCollectionAgent.run()` dispatches by `sources[].type`: `hf_dataset` → `_collect_hf`, `api` → `_collect_api`, `mediawiki_page` → `_collect_mediawiki_page`.

### Row limits (merged dataset)

After each source is collected, its dataframe may be **trimmed** before concatenation into `merged_raw`:

| Config key | Meaning |
|------------|---------|
| `collection.max_rows_per_source` | Default cap applied to any source that does **not** set its own `max_rows`. |
| `sources[].max_rows` | Optional per-source override. If this key is present, it wins over `max_rows_per_source`. |

`null` (YAML) or omission of a cap means **no limit** at that level: e.g. `max_rows_per_source: null` and no `max_rows` on a source → that source keeps all rows. Explicit `max_rows: null` on a source also means **unlimited** for that source (useful when the collection default is a positive integer).

### Example: Wiktionary category (`api` + categorymembers)

```yaml
# Under collection.sources:
- type: api
  name: wiktionary_ru_phraseologisms_category
  endpoint: https://ru.wiktionary.org/w/api.php
  params:
    action: query
    list: categorymembers
    cmtitle: Категория:Фразеологизмы/ru
    cmlimit: 200
    format: json
  text_field: title
  label: native_ru_seed
  language: ru
  meta:
    seed_role: native_ru_seed
```

`records_path` is optional for categorymembers (ignored when pagination is used); you can omit it or keep `[query, categorymembers]` for documentation.

### Example: Wiktionary appendix page (`mediawiki_page`)

```yaml
- type: mediawiki_page
  name: wiktionary_ru_idioms_page
  endpoint: https://ru.wiktionary.org/w/api.php
  params:
    action: parse
    page: Приложение:Список_фразеологизмов_русского_языка
    prop: wikitext
    format: json
  parse_mode: mediawiki_wikitext_list
  wikitext_parser_config:
    include_nonlist_lines: false
  label: native_ru_seed
  language: ru
  meta:
    seed_role: native_ru_seed
```

## Smoke checks (no Hugging Face)

After `pip install -r requirements.txt`:

```bash
python3 -c "from shared.config import load_config; print(load_config('config.yaml')['project']['name'])"
python3 -c "from agents import DataCollectionAgent, ActiveLearningAgent; print('agents OK')"
python3 run_pipeline.py --help && python3 run_agent.py --help
```

End-to-end collection requires network access to download datasets.

## License / safety

Collected text may include toxic material used as **donors** for research. Use only for benchmarking and safety research.
