# RuFP Bench Stage 2.5: Repair Loop

Этот пакет отвечает за **контролируемое** исправление (repair) **пограничных** промптов, отобранных из артефактов Stage 2. Он **не** формирует финальный бенчмарк: после repair каждый промпт снова проходит **полную revalidation** (тот же контур судей и probes, что на Stage 2); только после этого решение агрегатора определяет, может ли запись идти к **Stage 3 (QC / shaping)**.

**Место в общем пайплайне:** см. корневой **[docs/pipeline_roadmap.md](../../docs/pipeline_roadmap.md)** (Stage 0 → 1 → 2 → **2.5** → 3).

## Policy / config (`configs/stage25.yaml`)

Поведение узлов задаётся YAML, валидируемым Pydantic (`rufp_stage25.policy`): допустимые `repair_reason`, исключения, разрешённые `repair_strategies`, лимиты, revalidation (в т.ч. `run_probes`), правила агрегатора (в т.ч. `probe_required_for_promotion` и пороги manual review), resume, `runtime.use_mock_clients` и уровень логов.  
Путь к файлу: переменная окружения **`RUFP_STAGE25_CONFIG`** или флаг CLI **`--config`**. При ошибке схемы выбрасывается **`Stage25ConfigError`** с понятным текстом.  
В артефакты прогона пишется копия: `artifacts/stage25/<run_id>/stage25_policy_resolved.json`.

## Основные этапы Stage 2.5
1. **Candidate Identification**: Отбор промптов из `review_queue` или `validated_semantic_set` (с низким borderline score).
2. **Repair Planning**: Анализ причин неудачи и выбор стратегии исправления.
3. **Repair Execution**: Применение правок с помощью LLM-редактора.
4. **Re-validation**: Повторный прогон через логику Stage 2 (safety, naturalness, probes).
5. **Promotion**: Перенос успешно исправленных промптов в финальные наборы.

## Использование (end-to-end)
```bash
PYTHONPATH=src python -m rufp_stage25.cli run-stage25 \
  --stage2-run-id <stage2_run_id> \
  --run-id <stage25_run_id> \
  --mock
```

Отдельные ноды по-прежнему можно вызывать из CLI (см. `--help`).

### Revalidation (Node 4, повторный Stage 2)
Переиспользуются узлы `rufp_stage2`: `SafetyJudgeNode`, `NaturalnessJudgeNode`, `BorderlineJudgeNode`, `RefusalProbeRunnerNode`.

```bash
PYTHONPATH=src python -m rufp_stage25.cli run-revalidation \
  --input artifacts/stage25/<run_id>/repaired_prompts.jsonl \
  --run-id <run_id> \
  --mock \
  --models probe_a,probe_b \
  --only-ids rufp-repaired-xxx,rufp-repaired-yyy   # опционально: подмножество
```

Артефакт: `artifacts/stage25/<run_id>/repair_revalidation_results.jsonl`.

### Decision aggregator (Node 5)
После revalidation: `repair_promoted_set.jsonl`, `repair_failed_set.jsonl`, `repair_review_queue.jsonl`, `stage25_summary.json`.

```bash
PYTHONPATH=src python -m rufp_stage25.cli run-repair-aggregator --run-id <run_id>
```

## Структура проекта
- `nodes/`: Логика планирования и выполнения ремонта.
- `prompts/`: Шаблоны для планировщика и исполнителя.
- `pipeline/`: Оркестрация цикла ремонта.
- `analysis/`: Анализ эффективности ремонта и audit-метрики (`repair_audit_report.py`).
- `audit/`: Человекочитаемый вывод цепочки (`show_lineage.py`).
- `utils/`: Хеши SHA-256 и краткие diff-summary для lineage.

## Lineage и аудит
- Каждый ремонт пишет строку в `repair_lineage.jsonl` (связь `stage1_prompt_id` ↔ `original_prompt_id` ↔ `repaired_prompt_id`, хеши до/после, снимки Stage 2 / плана / revalidation).
- `repaired_prompts.jsonl` дублирует ключевые поля для быстрых выборок.
- CLI: `python -m rufp_stage25.cli show-repair-lineage --repaired-prompt-id <id> [--run-id <run>]`
- Отчёты: `artifacts/stage25/<run_id>/reports/stage25_failure_report.md` (анализ repair / воронка), `repair_audit_digest.md` и `repair_audit_metrics.json` (краткий audit по lineage).
