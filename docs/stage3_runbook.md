# Stage 3 — runbook & troubleshooting

Предполагается: репозиторий в корне, пакет доступен как `rufp_stage3` (`PYTHONPATH=src` или editable install). Артефакты: `RUFP_ARTIFACTS_ROOT` (по умолчанию текущая директория).

---

## 1. Полный прогон

```bash
export RUFP_ARTIFACTS_ROOT=/path/to/repo
export PYTHONPATH=src

python3 -m rufp_stage3.cli run-stage3 \
  --stage3-run-id MY_S3_RUN \
  --stage2-run-id MY_S2_RUN \
  --stage25-run-id MY_S25_RUN \
  --config configs/stage3.yaml \
  --resume
```

`--dry-run` — урезать объём по источникам (`dry_run.max_rows_per_source`).  
`--resume` — пропускать уже заполненные шаги (см. `resume.skip_completed_nodes` в конфиге).

Выход: `artifacts/stage3/MY_S3_RUN/` (pool, QC, баланс, сплиты, `stage3_exports/`, отчёты).

---

## 2. Отдельные ноды (тот же `RUN_ID`)

Порядок зависимостей: **1 → 2 → 3 → 4 → 5**.

| Шаг | Команда |
|-----|---------|
| 1. Candidate pool | `python3 -m rufp_stage3.cli build-stage3-candidate-pool --stage2-run-id … --stage3-run-id RUN_ID [--stage25-run-id …] --config configs/stage3.yaml` |
| 2. QC + dedup | `python3 -m rufp_stage3.cli run-stage3-qc --run-id RUN_ID --config configs/stage3.yaml` |
| 3. Balancer | `python3 -m rufp_stage3.cli run-stage3-balancer --run-id RUN_ID --config configs/stage3.yaml` |
| 4. Hard subset | `python3 -m rufp_stage3.cli run-stage3-hard-subset --run-id RUN_ID --config configs/stage3.yaml` |
| 5. Splits + export | `python3 -m rufp_stage3.cli run-stage3-splits --run-id RUN_ID --config configs/stage3.yaml` |

После полного прогона или правки артефактов — пересобрать release-доки:

```bash
python3 -m rufp_stage3.cli prepare-stage3-release --run-id RUN_ID --config configs/stage3.yaml
```

**Опционально — audit packs** (малые JSONL для ручной ревью, не влияют на основной экспорт):

```bash
python3 -m rufp_stage3.cli export-stage3-audit-packs --run-id RUN_ID --config configs/stage3.yaml
# переопределения: --random-n 40 --hard-n 20 --repaired-n 15 --seed 7 --random-split review_holdout --hard-order score_desc
```

Подробнее: `docs/stage3_audit_packs.md`.

---

## 3. Summary (`stage3_summary.json`)

Файл: `artifacts/stage3/<RUN_ID>/stage3_summary.json`.

Смотрите: `pool_size`, `qc_passed`, `dedup_clusters`, `balanced_size`, `hard_subset_size`, размеры `dev` / `test` / `review_holdout`, `exports_dir`. Расширенные поля (QC, баланс, `node_metrics`) могут дублироваться в том же файле после финального бандла анализа.

---

## 4. Final report (`stage3_final_report.md`)

Файл: `artifacts/stage3/<RUN_ID>/stage3_final_report.md`.

Там: поток по этапам, баланс категорий, repaired vs non-repaired, hard subset (JSON-блок), сплиты и **списки leakage** (семьи/кластеры на нескольких сплитах), тайминги нод, ссылка на `metrics/`.

---

## 5. Проверка leakage

1. Откройте `metrics/split_stats.json` → `family_across_multiple_splits`, `cluster_across_multiple_splits` (должны быть пустыми при strict policy и корректном прогоне).
2. В `stage3_final_report.md` — секция про splits / leakage.
3. Убедитесь, что `split_export.family_leakage_policy` и `cluster_leakage_policy` = `strict` в `configs/stage3.yaml` для релизных прогонов.

---

## 6. Hard subset stats

- `metrics/hard_subset_stats.json` — размер, средний/ min / max score, короткие примеры `why_hard`.
- Экспорт: `stage3_exports/hard_subset_benchmark.jsonl` (+ csv при включённом формате).

---

## Troubleshooting

### Слишком много дубликатов

- Поднять `near_duplicate.threshold` (меньше агрессивный near-dup) или сменить `near_duplicate.metric` / `normalization`.
- Проверить `dedup.method` и нормализацию текста в Stage 2.
- Узкий `clustering.dedup_strata` (например только `category`) увеличивает слияние внутри категории — при «слишком много» near-dup попробовать явно `[category, subtype]`.

### «Взрыв» числа кластеров

- Часто из-за низкого near-threshold или слабой нормализации — слегка **увеличить** `near_duplicate.threshold`.
- Проверить выбросы: очень короткие промпты (`qc_thresholds.min_text_len`).

### Дисбаланс категорий

- `balance.max_per_category`, `target_pool_size`, `category_balance_weights` в `balance`.
- Исключить/урезать редкие категории на входе (Stage 2 / отбор пула).

### Доминирование семей

- `balance.max_rows_per_family_per_category`, **`max_items_per_family`** (глобальный потолок).
- `split_export` не решает баланс категорий — только утечки между сплитами.

### Hard subset схлопнулся в пару категорий

- Поднять `hard_subset.min_categories_in_subset`, снизить `max_fraction_per_category`.
- В `hard_subset.scoring` скорректировать веса refusal vs benchmark.
- Проверить eligibility: `min_distinct_models_with_refusal`, `key_probe_refusal_alone_ok`.

### Обнаружен split leakage

- Убедиться, что `split_export.enforce_single_split_per_family` / `enforce_single_split_per_dedup_cluster` включены и leakage policies = `strict`.
- Пересобрать Node 5 после исправленных `stage3_dedup_clusters.jsonl` / пула.
- Если leakage в отчёте не пустой — не использовать run как финальный benchmark без разбора.

### Repaired промпты перегружают сплиты

- `split_export.balance_repaired_across_splits`, `repaired_original_mixing` (`interleave` vs блоки).
- На уровне пула: пересмотреть долю Stage 2.5 во входе.

---

## Рекомендуемые следующие шаги

| Симптом | Действие |
|---------|----------|
| Общая форма датасета не та | **Подкрутить `configs/stage3.yaml`** (секции `qc_thresholds`, `near_duplicate`, `balance`, `hard_subset`, `split`, `split_export`). |
| Странный микс repair | **Пересмотреть стратегию Stage 2.5** (что попадает в promoted / repaired_accept). |
| Шумные/пустые категории | **Урезать low-value категории** на входе Stage 2 или через `balance` / отбор пула. |
| Одна семья давит метрики | **Ужесточить family cap** (`max_items_per_family`, `max_rows_per_family_per_category`). |
| Hard subset не отражает цель | **Настроить `hard_subset.scoring`** и пороги `min_score` / `top_k` / `fraction`. |

---

## Быстрый чеклист после прогона

```bash
ls artifacts/stage3/RUN_ID/stage3_summary.json
ls artifacts/stage3/RUN_ID/stage3_final_report.md
ls artifacts/stage3/RUN_ID/metrics/split_stats.json
ls artifacts/stage3/RUN_ID/metrics/hard_subset_stats.json
```

```bash
python3 -c "import json; d=json.load(open('artifacts/stage3/RUN_ID/metrics/split_stats.json')); print('families', len(d.get('family_across_multiple_splits',[])), 'clusters', len(d.get('cluster_across_multiple_splits',[])))"
```
