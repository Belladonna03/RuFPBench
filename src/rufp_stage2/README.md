# RuFP Bench Stage 2: Gating & Validation

Этот пакет отвечает за валидацию кандидатов, сгенерированных на Stage 1, и сбор сигналов об отказах (refusals) от целевых моделей. Часть **repairable** кейсов затем может уйти в **Stage 2.5 (Repair Loop)** с повторной валидацией — см. **[docs/pipeline_roadmap.md](../../docs/pipeline_roadmap.md)**.

## Основные этапы Stage 2
1. **Semantic Gating**: Проверка промптов на безопасность (safety), естественность языка (naturalness) и силу пограничного эффекта (borderline).
2. **Refusal Probing**: Прогон валидных промптов через набор целевых LLM для фиксации случаев over-refusal.
3. **Aggregation**: Распределение промптов по итоговым наборам (validated_semantic, probe_positive, review_queue, reject_set).

## Использование
```bash
python -m src.rufp_stage2.cli --input-path artifacts/stage1/prompt_candidate_bank_raw.jsonl --mock
```

## Структура проекта
- `nodes/`: Логика отдельных этапов валидации.
- `prompts/`: Шаблоны для LLM-судей.
- `analysis/`: Скрипты для анализа результатов и формирования отчетов.
- `llm/`: Клиенты для взаимодействия с моделями.
