"""JSONL / JSON helpers (Pydantic v2)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def load_jsonl(path: str, model: type[T]) -> List[T]:
    items: List[T] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(model.model_validate_json(line))
    return items


def load_jsonl_dicts(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: str, items: List[BaseModel]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(data, BaseModel):
            f.write(data.model_dump_json(indent=2))
        else:
            json.dump(data, f, indent=2, ensure_ascii=False)


def file_nonempty(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0
