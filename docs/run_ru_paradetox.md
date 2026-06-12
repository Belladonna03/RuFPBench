# Запуск пайплайна на `s-nlp/ru_paradetox`

Пошаговая инструкция для локального MVP: **GLiNER → entities**, **KeyBERT → keyphrases**, **merge + dedup → pseudo-graph** (CLI `python -m src.cli run`).

## О датасете

**[s-nlp/ru_paradetox](https://huggingface.co/datasets/s-nlp/ru_paradetox)** — параллельный корпус для **детоксификации** русского текста (задача близка к перефразированию с переносом стиля: токсичная форма ↔ нейтральная). В каждой строке два поля:

| Поле | Смысл |
|------|--------|
| `ru_toxic_comment` | исходная (токсичная) формулировка |
| `ru_neutral_comment` | нейтральный параллельный вариант |

Для нашего пайплайна можно прогонять:

- только **токсичную** сторону (`ru_toxic_comment`);
- только **нейтральную** (`ru_neutral_comment`);
- **обе отдельно** — два экспорта и два прогона CLI (рекомендуется не смешивать в одном файле без явной цели: разный лексикон и шум).

**Рекомендация для первого запуска:** начать с **нейтральной** стороны (`--side neutral`). Обычно меньше оскорблений и «шумной» лексики → стабильнее NER/ключевые фразы; токсичную сторону имеет смысл сравнить вторым прогоном.

Корень проекта в примерах: `ru_fp_bench_2`. Подставьте свой путь или работайте из каталога репозитория.

---

## 1. Установка зависимостей

```bash
cd /Users/dekovaleva/PythonProjects/ru_fp_bench_2   # ваш путь
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install datasets        # нужно только для экспорта HF → JSONL
```

**Ожидаемый результат:** `pip` завершается без ошибок; при первом запуске пайплайна скачаются веса GLiNER и sentence-transformers (нужен интернет, несколько ГБ на диск).

---

## 2. Скачивание датасета

Явно «скачивать» архив не обязательно: при первом вызове скрипта экспорта (или `load_dataset`) данные подтянутся с Hugging Face в локальный кэш (`~/.cache/huggingface/datasets`).

Проверка из Python (опционально):

```bash
source .venv/bin/activate
python3 -c "from datasets import load_dataset; d=load_dataset('s-nlp/ru_paradetox','train'); print(len(d), list(d[0].keys()))"
```

**Ожидаемый результат:** для `train` обычно **~11k** строк (точное число зависит от ревизии датасета на Hub); ключи включают `ru_toxic_comment`, `ru_neutral_comment`. Есть также сплит **`validation`** (~1.1k строк).

---

## 3. Какое поле брать как входной текст

Пайплайн читает **одну строку = один документ**, поле текста задаётся флагом **`--text-column`** (по умолчанию `text`).

После экспорта скриптом ниже в JSONL всегда есть:

- **`text`** — выбранная сторона (`neutral` → содержимое `ru_neutral_comment`, `toxic` → `ru_toxic_comment`);
- **`ru_toxic_comment`** и **`ru_neutral_comment`** — дублируются в записи и попадут в **`metadata.extra`** в выходном графе (удобно для сопоставления пар).

Итого для CLI: **`--text-column text`**, **`--id-column id`**.

---

## 4. Экспорт в JSONL

Скрипт: **`scripts/export_ru_paradetox_to_jsonl.py`**.

Примеры (из корня репозитория):

```bash
source .venv/bin/activate
mkdir -p data/ru_paradetox

# Нейтральная сторона, train (рекомендуется для первого прогона)
python3 scripts/export_ru_paradetox_to_jsonl.py \
  --side neutral \
  --split train \
  -o data/ru_paradetox/train_neutral.jsonl

# Токсичная сторона, train
python3 scripts/export_ru_paradetox_to_jsonl.py \
  --side toxic \
  --split train \
  -o data/ru_paradetox/train_toxic.jsonl

# Validation (по желанию)
python3 scripts/export_ru_paradetox_to_jsonl.py \
  --side neutral \
  --split validation \
  -o data/ru_paradetox/validation_neutral.jsonl
```

**Ожидаемый результат:** в консоли строка вида `Wrote N line(s) to .../train_neutral.jsonl`; файл в UTF-8, одна JSON-объект на строку.

CSV пайплайн тоже умеет; JSONL удобнее для длинных строк и совпадает с примерами README. Конвертация в CSV не обязательна.

---

## 5. Запуск пайплайна (CLI)

Общий вид:

```bash
cd /Users/dekovaleva/PythonProjects/ru_fp_bench_2
source .venv/bin/activate
export PYTHONPATH=.

python3 -m src.cli run \
  -i data/ru_paradetox/train_neutral.jsonl \
  -o data/ru_paradetox/out/train_neutral_pseudograph.jsonl \
  --text-column text \
  --id-column id
```

**Куда пишется результат:** ровно в файл, указанный в **`-o`** (`--output`). Родительская папка создаётся при необходимости. Без **`--save-debug-csv`** выход **стримится** построчно (не держит все графы в RAM).

Дополнительные полезные флаги (как в README):

- **`--device cuda`** / **`mps`** / **`cpu`** — при необходимости;
- **`--continue-on-error`** — при сбое на одном документе пишется пустой граф с ошибкой в `metadata.extra`;
- **`--limit N`** — обработать только первые N строк **после** загрузки файла.

---

## 6. Debug CSV

Включение:

```bash
python3 -m src.cli run \
  -i data/ru_paradetox/train_neutral.jsonl \
  -o data/ru_paradetox/out/train_neutral_pseudograph.jsonl \
  --text-column text \
  --id-column id \
  --save-debug-csv
```

**Куда:** рядом с выходным JSONL, с тем же «stem», что у `-o`:

- `.../train_neutral_pseudograph_debug_nodes.csv`
- `.../train_neutral_pseudograph_debug_edges.csv`

**Важно:** для этого режима все графы накапливаются в памяти до конца прогона (см. `--help`). На полном train это обычно приемлемо по размеру, но для очень больших корпусов лучше без этого флага.

---

## 7. Smoke test (маленькая подвыборка)

**Вариант A — урезать при экспорте:**

```bash
python3 scripts/export_ru_paradetox_to_jsonl.py \
  --side neutral --split train --limit 32 \
  -o data/ru_paradetox/smoke_neutral.jsonl

PYTHONPATH=. python3 -m src.cli run \
  -i data/ru_paradetox/smoke_neutral.jsonl \
  -o data/ru_paradetox/out/smoke_neutral_pseudograph.jsonl \
  --text-column text \
  --id-column id
```

**Вариант B — полный экспорт, урезать в CLI:**

```bash
PYTHONPATH=. python3 -m src.cli run \
  -i data/ru_paradetox/train_neutral.jsonl \
  -o data/ru_paradetox/out/smoke_out.jsonl \
  --text-column text \
  --id-column id \
  --limit 32
```

**Ожидаемый результат:** прогресс tqdm по документам; в логе сводка `RunStats`; в выходе **32** строки JSONL (по одному `DocumentGraph` на документ).

---

## 8. Полный датасет (train)

```bash
# экспорт (если ещё не сделан)
python3 scripts/export_ru_paradetox_to_jsonl.py \
  --side neutral --split train \
  -o data/ru_paradetox/train_neutral.jsonl

mkdir -p data/ru_paradetox/out

PYTHONPATH=. python3 -m src.cli run \
  -i data/ru_paradetox/train_neutral.jsonl \
  -o data/ru_paradetox/out/train_neutral_pseudograph.jsonl \
  --text-column text \
  --id-column id
```

**Ожидаемый результат:** выходной JSONL с **числом строк = числу строк во входе** (если не использовали `--continue-on-error` и не было фатальных ошибок). Время — от минут до часов в зависимости от CPU/GPU.

Вторая полная прогонка для токсичной стороны — отдельный входной файл и отдельный `-o`:

```bash
python3 scripts/export_ru_paradetox_to_jsonl.py \
  --side toxic --split train \
  -o data/ru_paradetox/train_toxic.jsonl

PYTHONPATH=. python3 -m src.cli run \
  -i data/ru_paradetox/train_toxic.jsonl \
  -o data/ru_paradetox/out/train_toxic_pseudograph.jsonl \
  --text-column text \
  --id-column id
```

---

## Краткий чеклист

1. `pip install -r requirements.txt` и `pip install datasets`
2. `python3 scripts/export_ru_paradetox_to_jsonl.py --side neutral --split train -o data/ru_paradetox/train_neutral.jsonl`
3. `PYTHONPATH=. python3 -m src.cli run -i ... -o ... --text-column text --id-column id`
4. Результат: путь из **`-o`**; опционально `*_debug_*.csv` при **`--save-debug-csv`**
