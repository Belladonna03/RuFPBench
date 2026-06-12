from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml

from .config import AppConfig, resolve_path


def load_taxonomy(cfg: AppConfig) -> dict[str, Any]:
    path = resolve_path(cfg, cfg.paths.taxonomy_path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"categories": []}


def load_fewshots(cfg: AppConfig) -> list[dict[str, Any]]:
    path = resolve_path(cfg, cfg.paths.fewshot_path)
    if not path.exists():
        return []
    import json

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sample_categories(taxonomy: dict[str, Any], n: int, seed: int | None = None) -> list[dict[str, Any]]:
    cats = list(taxonomy.get("categories", []))
    if not cats:
        return []
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.append(rng.choice(cats))
    return out
