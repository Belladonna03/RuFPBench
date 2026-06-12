# Stage 3 — QC & benchmark export (skeleton)

Собирает финальный пакет из артефактов Stage 2 и Stage 2.5. **Не** генерирует и **не** ремонтирует промпты.

## Структура пакета

```text
src/rufp_stage3/
  __init__.py
  __main__.py          # python -m rufp_stage3
  config.py            # пути к artifacts/stage3/<run_id>
  schemas.py           # Pydantic-модели строк JSONL
  io.py
  cli.py
  policy/              # YAML configs/stage3.yaml (Pydantic)
    schema.py
    loader.py
  nodes/               # шаги пайплайна (можно вызывать отдельно из кода)
    build_pool.py
    qc_pass.py
    dedup.py
    balance.py
    hard_subset.py
    splits.py
    export_package.py  # тонкая обёртка над exporters
  pipeline/
    orchestrator.py
  analysis/
    final_report.py
  exporters/
    benchmark_bundle.py
```

## Схемы (JSONL)

| Модель | Назначение |
|--------|------------|
| `Stage3CandidateRecord` | объединённый пул |
| `QCLabelRecord` | QC |
| `DedupClusterRecord` | кластеры дедупа |
| `BalancedRecord` | балансировка |
| `HardSubsetRecord` | hard subset |
| `SplitAssignmentRecord` | dev/test/review_holdout |
| `FinalExportRecord` | манифест экспорта |
| `Stage3RunManifest` | метаданные прогона |

## Запуск

```bash
export PYTHONPATH=src
python -m rufp_stage3 --run-id MY_RUN --config configs/stage3.yaml --dry-run
```

Выход: `artifacts/stage3/<run_id>/` (корень задаётся `RUFP_ARTIFACTS_ROOT`).

## Конфиг

`configs/stage3.yaml` — `inputs.stage2_dir`, опционально `inputs.stage25_dir`, политика dedup/balance/splits.
