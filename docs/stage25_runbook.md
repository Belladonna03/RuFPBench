# Stage 2.5 — runbook & troubleshooting

Префикс для всех команд (из корня репозитория):

```bash
export PYTHONPATH=src
```

Модуль CLI: `python -m rufp_stage25.cli …`

Артефакты: `artifacts/stage25/<run_id>/`. Конфиг политики: `configs/stage25.yaml` или `RUFP_STAGE25_CONFIG=/path/to/stage25.yaml`.

---

## 1. Dry-run (mock, без реальных LLM/судей)

Быстрая проверка цепочки и артефактов:

```bash
python -m rufp_stage25.cli run-stage25 \
  --stage2-run-id <STAGE2_RUN_ID> \
  --run-id <STAGE25_RUN_ID> \
  --dry-run \
  --config configs/stage25.yaml
```

`--dry-run` принудительно включает mock-клиенты и помечает логи. Дополнительно: `--force-mock` (перекрывает `runtime.use_mock_clients` в YAML).

---

## 2. Real run (реальные клиенты)

В `configs/stage25.yaml` выставите `runtime.use_mock_clients: false` **или** передайте:

```bash
python -m rufp_stage25.cli run-stage25 \
  --stage2-run-id <STAGE2_RUN_ID> \
  --run-id <STAGE25_RUN_ID> \
  --real \
  --config configs/stage25.yaml
```

`--real` и `--force-mock` вместе недопустимы.

Переопределение probe-моделей:

```bash
python -m rufp_stage25.cli run-stage25 ... --models probe_a,probe_b
```

---

## 3. Отдельные ноды (ручная сборка)

Порядок: **1 → 2 → 3 → 4 → 5**. Пути — под ваш `<STAGE25_RUN_ID>`.

| Нода | Команда |
|------|---------|
| 1 Selector | `python -m rufp_stage25.cli run-repair-selector --stage2-run-id <S2> --run-id <S25> --config configs/stage25.yaml` |
| 2 Planner | `python -m rufp_stage25.cli run-repair-planner --input-path artifacts/stage25/<S25>/repair_candidates.jsonl --run-id <S25> --mock --config configs/stage25.yaml` |
| 3 Repairer | `python -m rufp_stage25.cli run-prompt-repairer --plans-path artifacts/stage25/<S25>/repair_plans.jsonl --candidates-path artifacts/stage25/<S25>/repair_candidates.jsonl --run-id <S25> --stage2-run-id <S2> --mock` |
| 4 Revalidation | `python -m rufp_stage25.cli run-revalidation --input artifacts/stage25/<S25>/repaired_prompts.jsonl --run-id <S25> --mock --config configs/stage25.yaml` |
| 5 Aggregator | `python -m rufp_stage25.cli run-repair-aggregator --run-id <S25> --config configs/stage25.yaml` |

Подмножество revalidation: `--only-ids rufp-repaired-xxx,rufp-repaired-yyy`.

Resume полного прогона (пропуск готовых шагов): `--resume` у `run-stage25` (см. `resume.skip_completed_nodes` в YAML).

---

## 4. Summary

Файл: `artifacts/stage25/<run_id>/stage25_summary.json`

- `promoted`, `failed`, `review`, счётчики и `promotion_rate_by_category` / `promotion_rate_by_repair_strategy`.

Быстро в консоли:

```bash
python3 -c "import json; print(json.dumps(json.load(open('artifacts/stage25/<S25>/stage25_summary.json')), indent=2))"
```

CLI после `run-repair-aggregator` печатает строку `promoted=… failed=… review=…`.

---

## 5. Failure report и метрики

После полного `run-stage25` (без `--no-analysis`):

| Файл | Содержание |
|------|------------|
| `reports/stage25_failure_report.md` | Текстовый разбор: стратегии, категории, деградации |
| `metrics/repair_strategy_stats.json` | Статистика по `repair_strategy` |
| `metrics/category_repair_funnel.json` | Воронка по категориям |
| `metrics/revalidation_delta_stats.json` | До/после по label-score |
| `reports/repair_audit_digest.md` | Краткий audit по lineage |

```bash
less artifacts/stage25/<S25>/reports/stage25_failure_report.md
```

---

## 6. Lineage одного repaired prompt

После финализации lineage в полном прогоне в `repair_lineage.jsonl` есть решение и снимок revalidation.

```bash
python -m rufp_stage25.cli show-repair-lineage \
  --repaired-prompt-id rufp-repaired-XXXXXXXX \
  --run-id <S25>
```

Без `--run-id` поиск по всем прогонам под `artifacts/stage25/` (медленнее).

---

## 7. Troubleshooting

### Invalid JSON from LLM (planner / repairer)

**Симптомы:** в логах предупреждения planner/repairer, пустой или урезанный `repair_plans.jsonl`.

**Действия:** усилить инструкции в шаблонах `prompts/*.jinja2`; снизить температуру на стороне клиента; для planner — retry уже есть в коде; проверить, что в ответе один JSON без лишнего текста.

### Слишком много repaired всё ещё падают на revalidation

**Смотреть:** `repair_failed_set.jsonl`, `failure_reason` в summary-логике, `stage25_failure_report.md`.

**Действия:** сузить `repair_reasons.allow` в YAML; ужесточить `repair_strategies.allowed`; проверить Stage 2 входы (не тянуть заведомо плохие кейсы); поднять качество плана (промпты planner/repairer).

### Repair loop даёт bland / «плоские» промпты

**Симптомы:** `still_translatedese_or_clunky`, `mixed_improvement_weak_borderline`, низкий borderline после revalidation.

**Действия:** отключить стратегии, дающие укорочение/обезличивание; добавить в план `must_preserve` / усилить `prompt_repairer` шаблон; пересмотреть `repair_reason` для weak borderline.

### Category drift после repair

**Симптомы:** `still_off_category_signal`, странные `borderline.rationale`.

**Действия:** сузить `repair_strategies` (убрать `add_realistic_context` и т.п., если виноваты); проверить `category` в кандидатах; вручную ревью `repair_review_queue.jsonl`.

### Переполненная manual review (`repair_review_queue`)

**Симптомы:** большой `review` в `stage25_summary.json`.

**Действия:** ослабить триггеры review в `configs/stage25.yaml` → `aggregator.manual_review` (например `review_on_probe_disagreement`, пороги confidence); временно ослабить `probe_required_for_promotion`; разобрать типичные `review_reason` в JSONL и править политику/промпты точечно.

### Mock vs real расхождение

**Симптомы:** на mock всё promote, на real — провалы.

**Действия:** прогнать один и тот же `<S25>` с `--force-mock` и `--real` на подмножестве (`run-revalidation --only-ids`); сравнить `repair_revalidation_results.jsonl`; убедиться, что real-клиенты Stage 2 настроены и те же шаблоны судей; mock даёт детерминированный JSON — real нет: ожидайте разброс.

---

## 8. Рекомендуемые шаги после прогона

1. **Подкрутить config** — `repair_reasons`, `limits.max_repairs_per_prompt`, `revalidation.run_probes`, `aggregator.*`.
2. **Сменить strategy mapping** — в planner/repairer промптах или в политике `repair_strategies.allowed`.
3. **Отключить плохие стратегии** — убрать значения из `repair_strategies.allowed` и перезапустить planner+ниже.
4. **Разобрать воронку по категориям** — `metrics/category_repair_funnel.json` + `stage25_failure_report.md`.
5. **Передать инсайты в Stage 3** — промоутнутые ID из `repair_promoted_set.jsonl` + политика QC; явные отказы и review — отдельные очереди, не смешивать с «чистым» Stage 2 без ремонта.

---

## Структура документа (оглавление)

1. Dry-run  
2. Real run  
3. Отдельные ноды  
4. Summary  
5. Failure report  
6. Lineage  
7. Troubleshooting  
8. Рекомендуемые шаги после run  

---

## Шпаргалка команд

```bash
export PYTHONPATH=src

# E2E dry-run
python -m rufp_stage25.cli run-stage25 --stage2-run-id S2 --run-id S25 --dry-run --config configs/stage25.yaml

# E2E real
python -m rufp_stage25.cli run-stage25 --stage2-run-id S2 --run-id S25 --real --config configs/stage25.yaml

# Только selector
python -m rufp_stage25.cli run-repair-selector --stage2-run-id S2 --run-id S25 --config configs/stage25.yaml

# Lineage
python -m rufp_stage25.cli show-repair-lineage --repaired-prompt-id <ID> --run-id S25
```
