# RuFPBench-MVP — Data Pipeline (Course Project)

End-to-end, reproducible data pipeline for **«Сбор и обработка данных»**: collect seed data, optional borderline rewrite, quality cleaning, weak auto-labeling with HITL, active-learning analysis, and a baseline classifier.

## Версии проекта

Код разных этапов RuFPBench лежит в отдельных ветках. Переключение:

```bash
git checkout ru_fp_bench_vN
```

| Ветка | Описание |
|-------|----------|
| `master` (текущая) | Исходный MVP data pipeline: agents + orchestrator |
| `ru_fp_bench_v2` | Pseudo-graph extraction (GLiNER + KeyBERT) |
| `ru_fp_bench_v3` | Stage 1: generation pipeline |
| `ru_fp_bench_v4` | Agentic search data gen + unsafe topics taxonomy |
| `ru_fp_bench_v5` | Cascade miner (ранняя версия) |
| `ru_fp_bench_v6` | Cascade miner v6 |
| `ru_fp_bench_v7` | Cascade miner + quality hardening |
| `ru_fp_bench_v8` | Cascade miner + scenario-first |
| `ru_fp_bench_v9` | Cascade miner + pipeline hardening |
| `ru_fp_bench_v10` | Cascade miner (false-reject product final) |

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

Copy `.env.example` to `.env` and set secrets (e.g. `HF_TOKEN` for Hugging Face Hub and **`PROXYAPI_API_KEY`** for ProxyAPI OpenRouter free models). On startup, `load_config()` loads the first `.env` found next to `config.yaml` (walking up parent directories) or `./.env` in the current working directory, so `datasets` / `huggingface_hub` receive `HF_TOKEN` via `os.environ`. You can also `export HF_TOKEN=...` in the shell.

## Run the full pipeline

```bash
python run_pipeline.py --config config.yaml
```

Stages (in order): **collection → rewrite (optional) → quality → annotation → HITL gate (optional stop) → merge corrections → AL batch export (optional) → AL curves (optional) → training (optional) → reports.**

If `hitl.human_mode = stop_if_missing` and `data/labeled/review_queue.jsonl` exists with rows but `data/labeled/review_results.jsonl` is missing or incomplete, the pipeline **exits with code 2** and prints what to do next.

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
python run_agent.py --agent annotation --config config.yaml --review-console --input data/labeled/review_queue.jsonl --output data/labeled/review_results.jsonl
python run_agent.py --agent al        --config config.yaml --input data/labeled/final_dataset.parquet
```

Aliases: `data_collection`, `borderline_rewrite`, `data_quality`, `active_learning`.

## Rewrite Modes

`BorderlineRewriteAgent` supports two generation modes controlled by `rewrite.defaults.n_variants_per_input` and optional per-policy overrides in `rewrite.policies[*].n_variants_per_input`.

- `single-variant`: `n_variants_per_input: 1`
- `multi-variant`: `n_variants_per_input: > 1`

Both modes now use the same prompt contract:

- `rewrite.borderline_definition`
- `rewrite.policies[*].generation_goal`
- `rewrite.policies[*].prompt_style.instructions`
- strategy-specific task rules inside `agents/rewrite_agent.py`

The difference is only the output shape:

- `single-variant` asks for the best single candidate
- `multi-variant` asks for several distinct candidates separated by `rewrite.modes.default.variant_separator`

### Rewrite Config Knobs

Core generation controls in `config.yaml`:

- `rewrite.defaults.n_variants_per_input`: global default for single vs multi mode
- `rewrite.policies[*].n_variants_per_input`: per-policy override
- `rewrite.modes.single_variant.best_candidate_bias`: extra guidance for picking one strongest answer
- `rewrite.modes.multi_variant.diversity_bias`: extra guidance for making variants meaningfully different
- `rewrite.modes.default.variant_separator`: separator expected in multi-output mode
- `rewrite.validation.*`: lightweight generation guardrails (format / artifact cleanup), not the full downstream quality stage
- `rewrite.policies[*].prompt_style.instructions`: task tuning for each strategy
- `rewrite.policies[*].strategy`: one of `generate_borderline_ru`, `rewrite_unsafe_to_borderline_ru`, `translate_and_adapt_to_ru_borderline`
- `rewrite.runtime.*`: concurrency, progress logs, retry behavior
- `rewrite.llm.*`: model, base URL, timeout, token limit

### Minimal Single-Variant Example

```yaml
rewrite:
  defaults:
    n_variants_per_input: 1
  policies:
    - name: wiki_native_to_ru_borderline
      strategy: generate_borderline_ru
      n_variants_per_input: 1
```

Use this mode when you want one best candidate per input row.

### Minimal Multi-Variant Example

```yaml
rewrite:
  defaults:
    n_variants_per_input: 3
  modes:
    default:
      variant_separator: ---
    multi_variant:
      diversity_bias:
        - Spread variants across framing, context, or speech register.
        - Avoid near-duplicates.
  policies:
    - name: wiki_native_to_ru_borderline
      strategy: generate_borderline_ru
      n_variants_per_input: 3
```

Use this mode when you want several candidate rewrites per input and will select or score them later in the pipeline.

### Strategy-Specific Prompt Tuning

`generate_borderline_ru`

- Best for idioms, phraseologisms, slangy or noisy lexical seeds.
- Tune `prompt_style.instructions` toward meaning, usage, tone, context, and example requests.
- Avoid dictionary-style outputs and bare lexical repeats.

`rewrite_unsafe_to_borderline_ru`

- Best for unsafe Russian donor texts that should become safe-but-borderline prompts.
- Tune `prompt_style.instructions` toward safe landing zones: explanation, neutral analysis, harmless context, technical or language clarification.
- Keep thematic proximity, but remove explicit harmful intent.

`translate_and_adapt_to_ru_borderline`

- Best for safe English seeds that need natural Russian adaptation.
- Tune `prompt_style.instructions` toward idiomatic Russian phrasing and pragmatic adaptation.
- Prefer meaning transfer over literal translation.

### Validation Layer

`rewrite.validation` performs lightweight generation validation inside the rewrite step:

- rejects literal `__SKIP__`
- rejects obvious meta-output like `вот вариант`
- rejects list-like or dictionary-like outputs when the result should be a user query
- rejects multi-answer blobs in `single-variant`
- normalizes multi-output splitting and deduplicates variants

This layer is only an operational guardrail for generation. Full candidate filtering, ranking, and evaluation still belong to downstream stages.

### Run Rewrite Only

Single-agent debug run:

```bash
python run_agent.py --agent rewrite --config config.yaml --input data/raw/merged_raw.parquet
```

Custom output path:

```bash
python run_agent.py --agent rewrite --config config.yaml --input data/raw/merged_raw.parquet --output data/interim/rewrite_multi.parquet
```

Switch between `single-variant` and `multi-variant` by editing `n_variants_per_input` in `config.yaml`; no code change is needed.

### Run Inside Full Pipeline

The full pipeline uses the same rewrite settings:

```bash
python run_pipeline.py --config config.yaml
```

Pipeline order remains:

- `collection -> rewrite -> quality -> annotation -> ...`

If you want to compare single vs multi generation, create two config variants and run the pipeline separately for each.

## Imports (for notebooks)

```python
from agents.data_collection_agent import DataCollectionAgent
from agents.data_quality_agent import DataQualityAgent
from agents.annotation_agent import AnnotationAgent
from agents.active_learning_agent import ActiveLearningAgent
from agents.rewrite_agent import BorderlineRewriteAgent
```

## Assignment 1: DataCollectionAgent

`DataCollectionAgent` is the coursework agent for multi-source collection. It exposes the following public API:

- `scrape(url, selector) -> pd.DataFrame`
- `fetch_api(endpoint, params) -> pd.DataFrame`
- `load_dataset(name, source="hf" | "kaggle") -> pd.DataFrame`
- `merge(sources: list[pd.DataFrame]) -> pd.DataFrame`
- `run(sources: list[dict] | None = None) -> pd.DataFrame`

Example:

```python
from agents.data_collection_agent import DataCollectionAgent

agent = DataCollectionAgent(config="config.yaml")
df = agent.run(
    sources=[
        {
            "type": "hf_dataset",
            "name": "imdb",
            "split": "train",
            "text_column": "text",
            "label_column": "label",
            "language": "en",
        },
        {
            "type": "scrape",
            "name": "example_quotes",
            "url": "https://example.com",
            "selector": "article.quote",
            "label": "plain_benign_control",
            "language": "en",
        },
    ]
)
```

The merged dataset is normalized to the common schema (`text/audio/image`, `label`, `source`, `collected_at`, plus metadata fields). The coursework notebook for this assignment is `notebooks/eda.ipynb`.

## Assignment 2: DataQualityAgent

`DataQualityAgent` is implemented as a compact observe → decide → act → evaluate loop for text classification data.

- `detect_issues(df) -> dict`
- `choose_strategy(report, df) -> dict`
- `fix(df, strategy) -> pd.DataFrame`
- `compare(df_before, df_after) -> pd.DataFrame`
- `run(df) -> dict`

Example:

```python
from agents.data_quality_agent import DataQualityAgent

agent = DataQualityAgent(task_type="text_classification")
result = agent.run(df)

report_before = result["report_before"]
strategy = result["chosen_strategy"]
df_clean = result["df_clean"]
comparison = result["comparison"]
```

For text classification, the agent focuses on empty `text`, empty `label`, duplicate texts, and text-length outliers. The coursework notebook for this assignment is `notebooks/data_quality.ipynb`.

## Assignment 3: AnnotationAgent

`AnnotationAgent` is the weak-supervision and human-in-the-loop labeling agent for the RuFPBench borderline-prompt task.

- `auto_label(df, modality="text") -> pd.DataFrame`
- `generate_spec(df, task) -> Path`
- `check_quality(df_labeled) -> dict`
- `export_to_labelstudio(df) -> Path`
- `build_review_queue(df) -> pd.DataFrame | None`

Example:

```python
from agents.annotation_agent import AnnotationAgent

agent = AnnotationAgent(modality="text", config="config.yaml")
df_labeled = agent.auto_label(df)
spec_path = agent.generate_spec(df_labeled, task="ru_fpbench_borderline_prompt_classification")
metrics = agent.check_quality(df_labeled)
labelstudio_path = agent.export_to_labelstudio(df_labeled)
```

Notes:

- The agent uses rule-based weak supervision tailored to `candidate_benign_borderline`, `plain_benign_control`, and `unsafe_or_not_suitable`.
- It produces `predicted_label`, `confidence`, `label_reason`, and `label_signals`.
- Low-confidence or otherwise suspicious examples are automatically routed to a review queue for HITL.
- Main HITL path is now console-first: `auto_label -> flag_for_review -> export_review_queue -> review_in_console -> merge_review_decisions`.
- Label Studio export is still available for compatibility, but it is no longer the primary operational path.
- The coursework notebook for this assignment is `notebooks/annotation_agent.ipynb`.

### Console-First HITL Workflow

After `annotation` step, the pipeline writes:

- `data/labeled/review_queue.jsonl` — review subset with predicted label, confidence, reason codes, and empty human decision fields
- `data/labeled/review_results.jsonl` — reviewer decisions created by the console review loop
- `data/labeled/labelstudio_import.json` — full compatibility export for Label Studio
- `data/labeled/labelstudio_low_confidence.json` — optional Label Studio export of the review subset

Primary workflow:

1. Run annotation:

```bash
python run_agent.py --agent annotation --config config.yaml --input data/interim/clean.parquet
```

2. Review flagged samples in console:

```bash
python run_agent.py --agent annotation --config config.yaml --review-console --input data/labeled/review_queue.jsonl --output data/labeled/review_results.jsonl
```

3. Re-run the full pipeline:

```bash
python run_pipeline.py --config config.yaml
```

The review loop supports:

- `a` — accept auto label
- `1..N` — assign one of the configured labels
- `s` — skip sample
- `n` — add a note
- `q` — save and quit

Repeated runs resume from existing decisions in `review_results.jsonl`.

## Assignment 4: ActiveLearningAgent

`ActiveLearningAgent` selects the next most useful examples for human annotation after the first weak-labeling / review cycle.

- `fit(labeled_df) -> model`
- `query(pool_df, strategy="entropy" | "margin" | "random", batch_size=..., model=None) -> indices`
- `evaluate(labeled_df, test_df, model=None) -> dict`
- `select_batch(pool_df, labeled_df, strategy="entropy", batch_size=...) -> pd.DataFrame`
- `run_cycle(labeled_df, pool_df, test_df, ...) -> list[dict]`
- `report(histories, output_path=...) -> Path | None`
- `export_candidates(df, output_csv=..., output_labelstudio=...)`

Example:

```python
from agents.active_learning_agent import ActiveLearningAgent

agent = ActiveLearningAgent(config="config.yaml")
model = agent.fit(labeled_df)
metrics = agent.evaluate(labeled_df, test_df, model=model)
indices = agent.query(
    pool_df=pool_df,
    strategy="margin",
    batch_size=50,
    model=model,
)
batch = agent.select_batch(
    pool_df=pool_df,
    labeled_df=labeled_df,
    strategy="entropy",
    batch_size=50,
)
```

Notes:

- The baseline selector uses TF-IDF + logistic regression.
- `entropy`, `margin`, and `random` are supported query strategies.
- `entropy` / `margin` selection is enriched with uncertainty metadata (`uncertainty`, `margin`, `predicted_label`) and a lightweight diversity filter.
- Candidates can be exported to CSV and Label Studio JSON for the next annotation round.
- `run_cycle()` now tracks `n_labeled`, `accuracy`, and `f1_macro` per iteration.
- The coursework notebook for this assignment is `notebooks/al_experiment.ipynb`.

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
- LLM calls use **ProxyAPI OpenRouter** in OpenAI-compatible `chat.completions` mode. Main envs are **`PROXYAPI_API_KEY`**, **`PROXYAPI_OPENROUTER_BASE_URL`** (default `https://api.proxyapi.ru/openrouter/v1`) and **`FREE_MODEL`** (default `openrouter/free`). You can also point `FREE_MODEL` to a specific free model such as `meta-llama/llama-3.3-70b-instruct:free`.
- Model selection supports an explicit switch via **`PROXYAPI_MODEL_MODE`**:
  - `free` — use `FREE_MODEL` and ignore legacy section-specific model envs
  - `specific` — use `PROXYAPI_MODEL` (or legacy section-specific model envs) for a fixed model
- Current recommended default is `PROXYAPI_MODEL_MODE=free`.
- The client logs both the requested model and the actual model returned in `response.model`, because `openrouter/free` may route to different free backends.
- Error handling distinguishes `429` rate limits, `402` balance / credits issues, and temporary `5xx` upstream failures. `429` and `5xx` are retried with exponential backoff; `402` raises an explicit error explaining that even free models may still require non-negative balance / credits on the OpenRouter side.

### Collection `sources` types

| `type` | Role |
|--------|------|
| `hf_dataset` | Hugging Face `datasets` |
| `scrape` | HTML page scraping via `requests` + CSS selector (`selector`) |
| `api` | JSON HTTP API (`endpoint`, `params`, `records_path`, `text_field`, …). If `params` is `action=query` + `list=categorymembers` (or `api_mode: categorymembers`), the agent fetches **all** pages via MediaWiki `continue` pagination and uses `text_field` (e.g. `title`) on each member. |
| `mediawiki_page` | One MediaWiki page via `action=parse` + `prop=wikitext` (`params.page` or `params.title`), then `parse_mode` (e.g. `mediawiki_wikitext_list`) on `parse.wikitext["*"]`. Optional `wikitext_parser_config`. |

`DataCollectionAgent.run()` dispatches by `sources[].type`: `hf_dataset` → `load_dataset()`, `scrape` → `scrape()`, `api` → `fetch_api()`, `mediawiki_page` → `_collect_mediawiki_page()`.

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
