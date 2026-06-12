# Pseudo-graph extraction (GLiNER + KeyBERT)

Локальный MVP: из **одной строки корпуса** (один текст) строится запись с сущностями (GLiNER), ключевыми фразами (KeyBERT), объединёнными узлами и **рёбрами совместной встречаемости** в пределах этого текста.

## Что делает пайплайн

1. Читает JSONL или CSV (колонка текста настраивается).
2. Для каждого документа: GLiNER → список span’ов; KeyBERT → список фраз с фильтрами и при возможности грубыми offset’ами в тексте.
3. Склеивает сущности и фразы в **узлы** (`merged_nodes`) с дедупом по нормализованной строке и опциональным soft-merge между label’ами.
4. Строит **рёбра** трёх типов: весь промпт, одно предложение (rule-based сегментация), окно в символах между span’ами с offset’ами.
5. Пишет JSONL: одна строка = один `DocumentGraph` (см. пример ниже).

## Чего пайплайн не делает

- Не извлекает **отношения** (who did what to whom); рёбра — не семантические связи, а эвристики совместной встречаемости.
- Не строит **knowledge graph**: нет онтологии, нет проверенных фактов, нет разрешения coreference на уровне KB.
- Не гарантирует качество NER/ключевых фраз: zero-shot GLiNER и KeyBERT дают шум, особенно на коротких и нетипичных текстах.
- Не делает идеальной **сегментации предложений**: только правила по `\n` и `.!?…`.

## Как читать узлы и рёбра

- **`merged_nodes`**: один узел = группа поверхностных упоминаний с одним и тем же нормализованным текстом (и при soft-merge — разные label’ы могут схлопнуться). `raw_mentions` хранит исходные span’ы/фразы. `node_id` детерминирован (hash от `doc_id` + текст + label’ы).
- **`edges`**: для каждой пары узлов и типа ребра не больше одного ребра; `source_node_id` ≤ `target_node_id` (лексикографически). **Веса — эвристики**, не вероятности P(relation).
  - `same_prompt`: узлы встретились в одном входном тексте.
  - `same_sentence`: есть offset’ы и оба попали в одно rule-based предложение (`evidence.sentence_ids`).
  - `nearby_window`: минимальный зазор между span’ами [start,end) не больше `--window-size`.

## Структура проекта

```
src/
  io_utils.py           # загрузка JSONL/CSV, запись JSONL и debug CSV
  normalization.py      # normalize_text, dedup, rough_word_tokens
  entity_extractor.py   # GLiNER, разрешение пересечений span’ов
  keyphrase_extractor.py    # KeyBERT, фильтры, поиск offset в тексте
  keyphrase_filter_rules.py # rule-based отсев низкоинформативных фраз
  merge_dedup.py            # слияние сущностей и фраз в узлы
  graph_builder.py      # предложения + рёбра
  schemas.py            # Pydantic-модели
  pipeline.py           # оркестрация и счётчики прогона
  cli.py                # Typer CLI
  config.py             # дефолты (labels, опц. blocklist для фраз)
tests/
requirements.txt
pytest.ini
```

## Установка

Требуется **Python 3.10+**, виртуальное окружение по желанию.

```bash
cd /path/to/ru_fp_bench_2
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Первый запуск скачает веса GLiNER и sentence-transformers (интернет, несколько ГБ на диске).

**Датасет Hugging Face `s-nlp/ru_paradetox`:** пошаговый экспорт в JSONL и команды CLI — [docs/run_ru_paradetox.md](docs/run_ru_paradetox.md).

## Запуск

Из корня проекта: `PYTHONPATH=.` (или см. `pytest.ini` для тестов).

```bash
PYTHONPATH=. python3 -m src.cli run \
  -i data/prompts.jsonl \
  -o data/pseudograph.jsonl \
  --text-column text \
  --id-column id \
  --gliner-model urchade/gliner_medium-v2.1 \
  --keybert-model sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
  --entity-threshold 0.35 \
  --top-n-keyphrases 10 \
  --window-size 64 \
  --use-mmr
```

Вход: файл UTF-8; выход: родительская директория создаётся при необходимости. Без `--save-debug-csv` результат **стримится** построчно в JSONL. Ошибки загрузки указывают путь и номер строки (JSONL).

| Флаг | Назначение |
|------|------------|
| `-i` / `-o` | Вход / выход |
| `--input-format` | `auto`, `jsonl`, `csv` |
| `--skip-bad-jsonl-lines` | Пропуск битых строк JSONL с предупреждениями |
| `--continue-on-error` | Пустой граф + `metadata.extra.pipeline_error` при сбое на документе |
| `--save-debug-csv` | Debug CSV (все графы в RAM до конца прогона) |
| `--disable-soft-merge` | Без слияния узлов с одним `normalized_text` и разными label |
| `--filter-overlaps-before-merge` | Жадное снятие пересечений GLiNER только для merge/графа |
| `--no-mmr` | KeyBERT без MMR |
| `--device` | `cuda`, `cpu`, `mps`, … |
| `--limit` | Первые N строк |
| `--log-level` | В т.ч. `DEBUG` для трассировок при `--continue-on-error` |

Сводка в логе: число документов, пустых текстов, без сущностей/фраз, «текст есть — извлечений нет», ошибки, средние узлы/рёбра на документ.

`python3 -m src.cli run --help`, `python3 -m src.cli version`.

## Пример входа (JSONL)

Поле `text` обязательно. Поля `id`, `source`, `category` — по желанию. **Все остальные ключи** строки попадают в `metadata.extra` без изменений (кроме имён колонок текста, id, source, category — они не дублируются в `extra`).

```json
{"id": "p1", "text": "Пример запроса на русском.", "category": "harmful", "source": "bench", "severity": 2}
```

## Пример выхода (JSONL)

Сокращённо:

```json
{
  "id": "p1",
  "original_text": "...",
  "normalized_text": "...",
  "entities": [],
  "keyphrases": [],
  "merged_nodes": [],
  "edges": [],
  "metadata": {"source": "bench", "category": "harmful", "num_entities": 0, "num_keyphrases": 0, "num_nodes": 0, "num_edges": 0, "extra": {"severity": 2}}
}
```

## Known limitations

- **GLiNER (zero-shot)**: качество и полнота сильно зависят от списка `entity_labels` и порога; возможны ложные срабатывания и пропуски.
- **KeyBERT**: шум на коротких текстах; пост-фильтр **rule-based** (структура текста + небольшие наборы служебных слов RU/EN в `keyphrase_filter_rules.py`), а не большой ручной blacklist. В `config.py` опционально `keyphrase_stop_phrases` / `DEFAULT_KEYPHRASE_EXTRA_BLOCKLIST` — только точечные фразы для удаления.
- **Offset’ы у keyphrases**: только литеральный / casefold поиск первого вхождения; при несовпадении с `original_text` остаётся `null` → нет `nearby_window` / хуже `same_sentence` для этих узлов.
- **Рёбра**: отражают **co-occurrence** и близость символов, а не истинные семантические отношения; веса для downstream лучше калибровать отдельно или трактовать как ранги.
- **Sentence splitting**: rule-based, ошибается на аббревиатурах, кавычках, «1. пункт» и т.д.
- **Soft merge**: узлы с одинаковым `normalized_text` и разными label схлопываются — разные смыслы одной формы могут смешаться.
- **Память / скорость**: последовательная обработка по документам; GPU ускоряет эмбеддинги; `--save-debug-csv` держит все графы в памяти.
- **CSV**: чтение через pandas, `utf-8-sig`; пустой файл → 0 строк и предупреждение.
- **JSONL**: строгий режим по умолчанию; «грязный» дамп — `--skip-bad-jsonl-lines`.

## Сознательные компромиссы MVP (trade-offs)

Это не недочёты «когда-нибудь починим», а явный выбор границы продукта:

- **Ошибки моделей не маскируются**: сбой `predict_entities` / `extract_keywords` прерывает документ (или даёт пустой граф только с `--continue-on-error`), а не тихий пустой список — проще отлаживать корпус.
- **`same_prompt` — полный граф по узлам**: любая пара узлов в одном документе получает ребро; на текстах с десятками узлов граф плотный — зато контракт простой для downstream.
- **Один `PipelineConfig` без пресетов**: сценарии (только NER, только keyphrases, другой язык) настраиваются флагами и правкой `config.py`, без профилей в коде.
- **`entities` в выходе vs merge**: при `--filter-overlaps-before-merge` в JSON остаётся полный список сущностей, а merge/граф используют отфильтрованный список — намеренно для трассировки и отладки.
- **Пакет не устанавливается как `pip install`**: запуск из корня с `PYTHONPATH=.` — меньше упаковочного boilerplate в MVP.

## Эвристики (кратко)

1. Нормализация текста: пробелы/переносы, без лемматизации; `canonical_text` в узлах **совпадает** с `normalized_text` (нижний регистр на слое dedup).
2. Узлы: дедуп по `(normalize_for_dedup(text), label)`; soft-merge по одному `normalized_text` между label’ами (если не отключено).
3. `node_id`: `n_` + 12 hex от SHA-256(`doc_id`, `canonical_text`, отсортированные `labels`).
4. Граф: см. модульный docstring в `src/graph_builder.py` (агрегация рёбер, сортировка, веса `same_prompt` / `nearby_window`).
5. Keyphrases после KeyBERT: правила в `keyphrase_filter_rules.py` — пустое/один символ после нормализации, длина, «только пунктуация/цифры», нет букв и цифр, фраза из одних служебных слов (RU+EN), один короткий токен; плюс опциональный точечный `extra_blocklist` по целой нормализованной фразе.

## Тесты

```bash
python3 -m pytest tests/ -v
```

## Лицензия

Укажите лицензию проекта при публикации; зависимости имеют свои лицензии.
