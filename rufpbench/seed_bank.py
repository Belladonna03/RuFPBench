from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import AppConfig, resolve_path
from .schemas import SeedIntent
from .taxonomy import load_taxonomy
from .utils import stable_id


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    return [str(value)] if str(value).strip() else []


def _seed_from_row(row: dict[str, Any], idx: int, source: str) -> SeedIntent:
    category = str(row.get("category") or row.get("id") or "unknown")
    risk_axis = str(row.get("risk_axis") or "unknown")
    benign = str(row.get("benign_use_case") or "")
    surface = str(row.get("sensitive_surface") or "")
    unsafe = str(row.get("unsafe_intent_summary") or "")
    phenomena = _as_list(row.get("ru_phenomena"))
    seed_id = stable_id("seed", source, idx, category, risk_axis, benign, surface)
    return SeedIntent(
        seed_id=seed_id,
        category=category,
        risk_axis=risk_axis,
        unsafe_intent_summary=unsafe or f"абстрактная небезопасная трактовка категории {category} без деталей",
        benign_use_case=benign or f"безопасная задача пользователя в категории {category}",
        sensitive_surface=surface or "чувствительная лексика и неоднозначная поверхность",
        ru_phenomena=phenomena or ["русский контекст"],
        source=source,
        metadata={"row_index": idx},
    )


def seeds_from_taxonomy(cfg: AppConfig) -> list[SeedIntent]:
    taxonomy = load_taxonomy(cfg)
    seeds: list[SeedIntent] = []
    idx = 0
    for cat in taxonomy.get("categories", []):
        category = str(cat.get("id") or cat.get("category") or "unknown")
        risk_axis = str(cat.get("risk_axis") or "unknown")
        use_cases = _as_list(cat.get("benign_use_cases")) or [f"безопасно разобрать чувствительную тему {category}"]
        surfaces = _as_list(cat.get("sensitive_surfaces")) or ["чувствительная лексика"]
        phenomena = _as_list(cat.get("ru_phenomena")) or ["русский контекст"]
        for benign in use_cases:
            for surface in surfaces:
                row = {
                    "category": category,
                    "risk_axis": risk_axis,
                    "benign_use_case": benign,
                    "unsafe_intent_summary": f"небезопасная трактовка по оси {risk_axis}, описанная только абстрактно",
                    "sensitive_surface": surface,
                    "ru_phenomena": phenomena,
                }
                seeds.append(_seed_from_row(row, idx, "taxonomy_ru"))
                idx += 1
    return seeds


def seeds_from_native_bank(cfg: AppConfig) -> list[SeedIntent]:
    path = resolve_path(cfg, cfg.paths.native_seed_bank_path)
    if not path.exists():
        return []
    seeds: list[SeedIntent] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                seeds.append(_seed_from_row(row, idx, "native_ru_seed_bank"))
    return seeds


def load_seed_bank(cfg: AppConfig) -> list[SeedIntent]:
    seeds = seeds_from_native_bank(cfg) + seeds_from_taxonomy(cfg)
    # Stable de-duplication by semantic key.
    seen: set[tuple[str, str, str, str]] = set()
    out: list[SeedIntent] = []
    for s in seeds:
        key = (s.category, s.risk_axis, s.benign_use_case, s.sensitive_surface)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out
