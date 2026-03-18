# RuFPBench-MVP — Final Data Pipeline (Course Project)

This repository contains an **end-to-end reproducible data pipeline** for the course project **"Сбор и обработка данных"**.

**Goal:** build an MVP dataset of Russian *benign-borderline* prompts (safe by intent, but lexically similar to risky prompts) — **candidate** false positives for safety systems — with **human-in-the-loop** review, Active Learning analysis, and a trained baseline model.

## One-command pipeline

```bash
pip install -r requirements.txt
python run_pipeline.py --config config.yaml
```

### Human-in-the-loop (HITL)

If `hitl.human_mode = stop_if_missing` (default), the first run will generate:

- `data/labeled/review_queue.csv` (also copied to `./review_queue.csv`)

Edit labels in that file and save it as:

- `data/labeled/review_queue_corrected.csv`

Then re-run the same command:

```bash
python run_pipeline.py --config config.yaml
```

Artifacts:
- Final dataset: `data/labeled/final_dataset.parquet`
- Data card: `data/labeled/DATA_CARD.md`
- Reports: `reports/final_report.md` (+ quality/annotation/AL/train reports)
- Model: `models/baseline_logreg.pkl`

---

# RuFPBench-MVP — DataCollectionAgent (Assignment 1)

This repo contains **DataCollectionAgent** for the course **"Сбор и обработка данных"**.

## Goal (ML task context)

The broader project goal is to build a dataset of **Russian benign-borderline prompts** (safe by intent, but lexically similar to risky prompts), which are **candidates** for false-positive refusals in safety systems.

**Assignment 1 scope:** only **data collection**. We collect *seed corpora* from multiple sources and unify them into a single dataset.  
Later pipeline stages (cleaning, auto-labeling, rewriting toxic → benign-borderline, human review, AL, training) are implemented in subsequent assignments.

> **Optional appendix step (post-collection):**
> This repo also includes a modular post-collection rewrite step (`agents/post_collection_appendix.py`).
> It can be enabled via `appendix_rewrite:` section in `config.yaml`.
> The appendix is **not required for Assignment 1**, but is useful for RuFPBench-MVP.

---

## What the agent does

- Collects data from **2+ sources**:
  - Open dataset from Hugging Face / Kaggle
  - Scraping **or** API source
- Maps all sources to a **fixed unified schema**
- Saves per-source and merged outputs into `data/raw/`

---

## Output schema (fixed columns)

The agent always returns a `pandas.DataFrame` with these columns:

| column | type | description |
|---|---|---|
| `uid` | str | stable hash id |
| `text` | str \| None | text content (for this project) |
| `audio` | str \| None | path/url (unused here) |
| `image` | str \| None | path/url (unused here) |
| `label` | any | **weak label** at collection stage (ok for Assignment 1) |
| `source` | str | source name |
| `collected_at` | str | UTC timestamp (ISO) |
| `language` | str \| None | `ru` / `en` |
| `source_type` | str | `hf_dataset` / `kaggle_dataset` / `scrape` / `api` |
| `url` | str \| None | origin url |
| `meta` | str \| None | JSON string with extra metadata |

---

## Quickstart

### 1) Install

```bash
pip install -r requirements.txt
```

### 2) Configure sources

Edit `config.yaml`.

### 3) Run collection

```bash
python -c "from agents.data_collection_agent import DataCollectionAgent; df = DataCollectionAgent('config.yaml').run(); print(df.head()); print(df.shape)"
```

Outputs:
- `data/raw/<source_name>.parquet` for each source
- `data/raw/merged_raw.parquet` and `data/raw/merged_raw.csv`

---

## EDA

Open and run:

- `notebooks/eda.ipynb`

The notebook contains:
- class distribution
- text length distribution
- top-20 tokens (overall and per label)

---

## Optional appendix: unsafe → benign-borderline rewrite (post-collection)

This step generates **candidate safe-but-borderline** Russian prompts from unsafe donor texts.
It is inspired by over-refusal benchmarks that rewrite toxic prompts into "seemingly toxic but safe" ones (e.g., OR-Bench).

1) Enable it in `config.yaml`:

```yaml
appendix_rewrite:
  enabled: true
  mode: rule     # or hybrid/llm
```

2) Run after collection:

```bash
python agents/post_collection_appendix.py --config config.yaml \
  --input data/raw/merged_raw.parquet \
  --output data/raw/merged_with_appendix.parquet
```

If you use `mode: llm` or `mode: hybrid`, set an OpenAI-compatible key:

```bash
export OPENAI_API_KEY="..."
```

---

## Notes on reproducibility

- Hugging Face sources require internet access.
- Kaggle sources (optional) require `kaggle` CLI credentials:
  - `~/.kaggle/kaggle.json` **or** env vars `KAGGLE_USERNAME` / `KAGGLE_KEY`.

---

## License / Safety

This project may collect text that contains toxic language (used as **donors** for later rewriting and filtering).  
Do not use collected data for harassment or harm; only for research/benchmarking and safety evaluation.


---

## DataQualityAgent (Assignment 2)

This repo also includes `DataQualityAgent` ("Data Detective") for Assignment 2.

### Quick usage

```bash
python - << 'PY'
import pandas as pd
from data_quality_agent import DataQualityAgent

df = pd.read_parquet('data/raw/merged_raw.parquet')
agent = DataQualityAgent()
report = agent.detect_issues(df)
print(report['duplicates'])

clean = agent.fix(df, strategy={
    'missing': 'fill',
    'duplicates': 'drop',
    'outliers': 'drop_iqr'
})
comparison = agent.compare(df, clean)
print(comparison['table'][:5])
PY
```

### Notebook

- `notebooks/quality_eda.ipynb` — визуализации проблем качества + 2 стратегии чистки + сравнение до/после.



---

## AnnotationAgent (Assignment 3)

Implements automatic labeling for **text** modality, generates an annotation specification, computes quality metrics, and exports tasks to Label Studio.

### Quick usage

```bash
python - << 'PY'
import pandas as pd
from annotation_agent import AnnotationAgent

# load data collected by Assignment 1
try:
    df = pd.read_parquet('data/raw/merged_with_appendix.parquet')
except Exception:
    df = pd.read_parquet('data/raw/merged_raw.parquet')

agent = AnnotationAgent(modality='text', config={'confidence_threshold': 0.7})
df_labeled = agent.auto_label(df)

spec_path = agent.generate_spec(df_labeled, task='ru_fpbench_borderline_prompt_classification')
print('spec:', spec_path)

metrics = agent.check_quality(df_labeled)
print(metrics)

ls_path = agent.export_to_labelstudio(df_labeled)
print('labelstudio:', ls_path)
PY
```

### Outputs

- `reports/annotation_spec.md`
- `data/labeled/labelstudio_import.json`
- `data/labeled/review_queue.csv` (low-confidence HITL queue)
- `data/labeled/labelstudio_low_confidence.json`

### Notebook

- `notebooks/annotation_demo.ipynb`



---

## ActiveLearningAgent (Assignment 4)

Track A: Active Learning agent for smart data selection.

### Quick usage

```bash
python - << 'PY'
import pandas as pd
from al_agent import ActiveLearningAgent
from sklearn.model_selection import train_test_split

# Load any collected dataset
try:
    df = pd.read_parquet('data/raw/merged_with_appendix.parquet')
except Exception:
    df = pd.read_parquet('data/raw/merged_raw.parquet')

# Minimal oracle label for AL simulation (you can replace with human labels later)
# NOTE: For real HITL, the selected batch would be exported for annotation.

df['label'] = df.get('label')  # if already labeled

# Use a small split for demo
train_pool, test_df = train_test_split(df, test_size=0.2, random_state=42)
labeled_df, pool_df = train_test_split(train_pool, train_size=50, random_state=42)

agent = ActiveLearningAgent(model='logreg')
h_entropy = agent.run_cycle(labeled_df=labeled_df, pool_df=pool_df, test_df=test_df,
                            strategy='entropy', n_iterations=5, batch_size=20)
h_random = agent.run_cycle(labeled_df=labeled_df, pool_df=pool_df, test_df=test_df,
                           strategy='random', n_iterations=5, batch_size=20)

agent.report({'entropy': h_entropy, 'random': h_random}, metric='f1',
             output_path='reports/learning_curve.png',
             title='Active Learning: entropy vs random (macro F1)')
print('Saved learning curve to reports/learning_curve.png')
PY
```

### Notebook

- `notebooks/al_experiment.ipynb` — full experiment: N=50 start, 5 iterations × 20, entropy vs random, label savings.

