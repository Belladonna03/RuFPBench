# RuFP Bench Stage 1: Generation Pipeline

Этот пакет отвечает за генерацию сырых кандидатов (raw prompt candidates) для русскоязычного false-positive бенчмарка.

## Сквозной roadmap пайплайна

Полный путь от таксономии до финального датасета (включая **Stage 2 — validation**, **Stage 2.5 — repair loop** и **Stage 3 — QC / shaping**) описан в **[docs/pipeline_roadmap.md](docs/pipeline_roadmap.md)**.  
Кратко: **Stage 0** = taxonomy → **Stage 1** = generation → **Stage 2** = validation → **Stage 2.5** = controlled repair + **повторная** валидация → **Stage 3** = финальное shaping. Stage 2.5 **не** выпускает финальный бенчмарк.

## Структура Stage 1
- **Family Router**: Выбор стратегии расширения.
- **Candidate Generator**: Генерация коротких, средних и длинных промптов.
- **RU Naturalizer**: Приведение к естественному русскому языку.
- **Borderline Refiner**: Усиление пограничных свойств без нарушения безопасности.

## Установка
```bash
pip install -r requirements.txt
```

## Использование
Запуск пайплайна в режиме заглушки (mock):
```bash
python -m src.rufp_stage1.cli --input-path tiny_sample.json --mock
```

## Артефакты
Результаты сохраняются в `artifacts/stage1/`.
