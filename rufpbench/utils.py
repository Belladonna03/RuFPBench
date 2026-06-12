from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TypeVar

T = TypeVar("T")
U = TypeVar("U")

CYR_RE = re.compile(r"[А-Яа-яЁё]")
WORD_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9_]+", re.U)
SPACE_RE = re.compile(r"\s+")


def now_ms() -> int:
    return int(time.time() * 1000)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


SECRET_KEY_RE = re.compile(
    r"^(api[_-]?key|apiKey|password|pass|secret|credentials?|access[_-]?token|refresh[_-]?token|authorization|bearer)$",
    re.I,
)


def redact_secrets(obj: Any) -> Any:
    """Return a copy with obvious secret-bearing keys redacted.

    This protects config.effective.json / report.config.json when users share run
    archives for analysis. It intentionally redacts by key name only; prompt and
    response text are left untouched.
    """
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            key_s = str(key)
            # *_env keys contain environment variable names, not secret values.
            # Also keep diagnostic knobs such as max_tokens/min_max_tokens and
            # allow_empty_api_key; previous broad matching redacted them because
            # they contain the substrings "token"/"key".
            is_env_name = key_s.endswith("_env") or key_s.endswith("Env")
            if SECRET_KEY_RE.search(key_s) and not is_env_name:
                out[key_s] = "***REDACTED***" if value not in (None, "", [], {}) else value
            else:
                out[key_s] = redact_secrets(value)
        return out
    if isinstance(obj, list):
        return [redact_secrets(x) for x in obj]
    return obj


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(redact_secrets(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json_dumps(row) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json_dumps(row) + "\n")


def stable_id(prefix: str, *parts: Any, length: int = 16) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\0")
    return f"{prefix}_{h.hexdigest()[:length]}"


def normalize_text(text: str) -> str:
    text = text.replace("ё", "е").replace("Ё", "Е")
    text = text.lower()
    text = re.sub(r"[`'\"“”«»]+", "", text)
    text = re.sub(r"[^a-zа-я0-9_\s-]+", " ", text, flags=re.I)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def cyrillic_ratio(text: str) -> float:
    chars = [c for c in text if c.isalpha()]
    if not chars:
        return 0.0
    return len(CYR_RE.findall(text)) / max(1, len(chars))


def token_set(text: str) -> set[str]:
    return set(WORD_RE.findall(normalize_text(text)))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def is_near_duplicate(text: str, existing_token_sets: list[set[str]], threshold: float) -> bool:
    ts = token_set(text)
    if not ts:
        return True
    for other in existing_token_sets:
        if jaccard(ts, other) >= threshold:
            return True
    return False


def dedupe_prompts(rows: list[dict[str, Any]], prompt_key: str = "prompt", threshold: float = 0.86) -> list[dict[str, Any]]:
    seen_norm: set[str] = set()
    seen_sets: list[set[str]] = []
    out: list[dict[str, Any]] = []
    for row in rows:
        prompt = str(row.get(prompt_key, ""))
        norm = normalize_text(prompt)
        if not norm or norm in seen_norm:
            continue
        ts = token_set(prompt)
        if any(jaccard(ts, prev) >= threshold for prev in seen_sets):
            continue
        seen_norm.add(norm)
        seen_sets.append(ts)
        out.append(row)
    return out


def chunks(items: list[T], size: int) -> Iterator[list[T]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _progress(iterable: Iterable[T], *, total: int, desc: str | None = None) -> Iterable[T]:
    """Wrap an iterable with tqdm when enabled and available.

    Progress bars are intentionally optional so tests and non-interactive logs do
    not depend on tqdm rendering. Set RUFP_DISABLE_TQDM=1 to disable them.
    """
    if not desc or os.getenv("RUFP_DISABLE_TQDM", "0").lower() in {"1", "true", "yes"}:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(
            iterable,
            total=total,
            desc=desc,
            dynamic_ncols=True,
            mininterval=float(os.getenv("RUFP_TQDM_MININTERVAL", "0.5")),
            file=sys.stderr,
        )
    except Exception:
        return iterable


def map_parallel(
    fn: Callable[[T], U],
    items: list[T],
    max_workers: int,
    desc: str | None = None,
) -> list[U]:
    if not items:
        return []
    if max_workers <= 1:
        return [fn(item) for item in _progress(items, total=len(items), desc=desc)]

    results: list[U | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_idx = {ex.submit(fn, item): i for i, item in enumerate(items)}
        completed = as_completed(future_to_idx)
        for fut in _progress(completed, total=len(future_to_idx), desc=desc):
            idx = future_to_idx[fut]
            results[idx] = fut.result()
    return [r for r in results if r is not None]


def bounded_sleep(base: float, attempt: int, jitter: float = 0.4) -> None:
    time.sleep(base * (2 ** max(0, attempt - 1)) + random.random() * jitter)


def redact_if_needed(text: str, should_redact: bool) -> str:
    if not should_redact:
        return text
    return "[REDACTED_BY_RUFPBENCH]"


def getenv_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    return [x.strip() for x in raw.split(",") if x.strip()]


def getenv_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def getenv_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)
