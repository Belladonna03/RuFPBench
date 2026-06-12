# RuFPBench cascade run plan

Use `README.md` as the main operating guide. This file is a compact checklist for real runs.

## 1. Prepare

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

Fill `.env` credentials/endpoints. Do not commit `.env`.

## 2. Probe required models

```bash
rufpbench probe --model oss
rufpbench probe-json --model qwen3.6-35b-a3b --output-count 3
rufpbench probe --model gigachat3-10b
rufpbench probe --model glm-4-7-fp8
rufpbench probe --model qwen3-vl-235b
rufpbench probe --model GigaChat-2-Max
rufpbench probe --model GigaChat-3-Ultra
```

Remove unstable models from the corresponding `llm.pipeline.*.models` list before a large run.

## 3. Run tests

```bash
pytest -q
```

Expected in this release: `35 passed`.

## 4. Small main-pipeline run

```bash
rufpbench run \
  --mode cascade_mining \
  --run-dir runs/main_400 \
  --target-raw 400 \
  --min-borderline 30 \
  --raw-batch-size 200 \
  --jobs-output-count 12 \
  --max-rounds 4 \
  --max-workers 8
```

Inspect:

```text
runs/main_400/final/rufpbench_borderline.jsonl
runs/main_400/reports/model_refusal_rates.csv
runs/main_400/validation/promotion_reasons.jsonl
runs/main_400/validation/qc_rejected.jsonl
```

## 5. Full 500+ run

```bash
rufpbench run \
  --mode cascade_mining \
  --run-dir runs/rufpbench_cascade_001 \
  --target-raw 8000 \
  --min-borderline 500 \
  --max-workers 16
```

The default config allows top-up generation up to 16000 raw attempts if the first 8000 do not yield 500 strict FP borderline rows.

## 6. Main outputs

```text
runs/<name>/final/rufpbench_borderline.jsonl
runs/<name>/final/rufpbench_hard.jsonl
runs/<name>/final/rufpbench_cross_model_hard.jsonl
runs/<name>/final/rufpbench_<model>_hard.jsonl
runs/<name>/reports/report.md
runs/<name>/state.json
```
