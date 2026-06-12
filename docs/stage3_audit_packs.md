# Stage 3 — audit packs (optional human review)

## Зачем

Небольшие JSONL-выборки для **spot-check** качества данных без просмотра всего бенчмарка: случайная страта, фокус на **hard subset** (over-refusal), отдельно **repaired** промпты. Не участвуют в основном пайплайне Stage 3 и **не меняют** экспорт `benchmark_*.jsonl`.

## Где лежат файлы

`artifacts/stage3/<RUN_ID>/stage3_exports/`:

| Файл | Содержимое |
|------|------------|
| `audit_pack_random.jsonl` | Случайная подвыборка из balanced pool (опционально только один split) |
| `audit_pack_hard.jsonl` | Строки из hard subset (по `hard_score`, см. policy) |
| `audit_pack_repaired.jsonl` | Подвыборка с `repair_flag=true` |
| `audit_packs_manifest.json` | Счётчики, seed, снимок `audit_packs` из конфига |

## Поля записи

Каждая строка — JSON с полями: `item_id`, `prompt_text`, `category`, `subtype`, `source_stage` (строка для чтения), `source_stages`, `probe_profile`, `repair_flag`, `qc_metadata` (label, flags, `cluster_id`, notes).

## Как использовать

1. После успешного Stage 3 (есть `stage3_balanced_pool.jsonl`, QC, при необходимости hard subset и splits).
2. Запустить экспорт (см. ниже).
3. Отдать JSONL ревьюеру / разметить в тулзе / конвертировать в таблицу. Для сравнения с релизом смотрите те же `item_id` в `benchmark_*.jsonl` и `final_lineage.jsonl`.

## Связь с final exports

Audit packs — **срезы того же balanced pool** (и тех же QC метаданных), что и основной экспорт. Это не отдельный датасет: строки должны совпадать с записями в `benchmark_{dev,test,review_holdout}.jsonl` по `item_id`, если этот id попал в выборку. Hard pack соответствует подмножеству `hard_subset_benchmark.jsonl` при той же политике отбора.

## Политика отбора

Задаётся в `configs/stage3.yaml` → секция **`audit_packs`** (см. `AuditPackConfig` в `policy/schema.py`). CLI может переопределить размеры и seed.

---

Пример команды см. в `docs/stage3_runbook.md` (секция audit packs) или:

```bash
export PYTHONPATH=src
python3 -m rufp_stage3.cli export-stage3-audit-packs --run-id RUN_ID --config configs/stage3.yaml
```

## Пример записи (`audit_pack_hard.jsonl`, одна строка)

```json
{
  "item_id": "s3-abc123def4567890",
  "prompt_text": "…",
  "category": "harm",
  "subtype": "faq",
  "source_stage": "stage2",
  "source_stages": ["stage2"],
  "probe_profile": { "probe_results": [], "refusal_count": 0 },
  "repair_flag": false,
  "qc_metadata": {
    "qc_label": "cluster_representative",
    "qc_pass": true,
    "qc_flags": [],
    "cluster_id": "nr-0a1b2c3d4e5f",
    "notes": ""
  }
}
```
