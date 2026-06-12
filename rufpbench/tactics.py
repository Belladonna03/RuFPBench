from __future__ import annotations

from typing import Any

import yaml

from .config import AppConfig, resolve_path
from .schemas import TacticSpec


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


def load_tactics(cfg: AppConfig) -> dict[str, TacticSpec]:
    path = resolve_path(cfg, cfg.paths.tactics_path)
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, TacticSpec] = {}
    for row in data.get("tactics", []):
        tid = str(row.get("tactic_id") or "").strip()
        if not tid:
            continue
        out[tid] = TacticSpec(
            tactic_id=tid,
            family=str(row.get("family") or "unknown"),
            ru_name=str(row.get("ru_name") or tid),
            use_for=_as_list(row.get("use_for")),
            safe_rule=str(row.get("safe_rule") or ""),
            forbidden_rule=str(row.get("forbidden_rule") or ""),
            mutation_goal=str(row.get("mutation_goal") or ""),
            source=str(row.get("source") or "wildteaming_ru_adapted"),
            metadata={k: v for k, v in row.items() if k not in {"tactic_id", "family", "ru_name", "use_for", "safe_rule", "forbidden_rule", "mutation_goal", "source"}},
        )
    return out


def load_compatibility(cfg: AppConfig) -> dict[str, Any]:
    path = resolve_path(cfg, cfg.paths.compatibility_path)
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def compatible_tactics(
    *,
    category: str,
    target_distribution: str,
    tactics: dict[str, TacticSpec],
    compatibility: dict[str, Any],
    strength: str = "strong",
) -> list[str]:
    row = compatibility.get(category) or compatibility.get("default") or {}
    if strength == "weak":
        preferred = _as_list(row.get("weak_tactics"))
    elif strength == "all":
        preferred = _as_list(row.get("strong_tactics")) + _as_list(row.get("weak_tactics"))
    else:
        preferred = _as_list(row.get("strong_tactics"))
    forbidden = set(_as_list(row.get("forbidden_tactics")))
    result = []
    for tid in preferred:
        spec = tactics.get(tid)
        if spec is None or tid in forbidden:
            continue
        if spec.use_for and target_distribution not in spec.use_for:
            continue
        result.append(tid)
    if not result and strength != "all":
        return compatible_tactics(
            category=category,
            target_distribution=target_distribution,
            tactics=tactics,
            compatibility=compatibility,
            strength="all",
        )
    return result
