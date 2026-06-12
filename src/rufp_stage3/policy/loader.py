"""Load ``configs/stage3.yaml``."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

from pydantic import ValidationError

from .schema import Stage3Policy

DEFAULT_PATH = os.path.join("configs", "stage3.yaml")


class Stage3ConfigError(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


def _fmt(err: ValidationError) -> str:
    lines = ["Stage 3 config validation failed:"]
    for e in err.errors():
        loc = ".".join(str(x) for x in e.get("loc", ()))
        lines.append(f"  - {loc}: {e.get('msg')} ({e.get('type')})")
    return "\n".join(lines)


def load_stage3_policy(path: Optional[Union[str, Path]] = None) -> Stage3Policy:
    raw = path or os.environ.get("RUFP_STAGE3_CONFIG") or DEFAULT_PATH
    p = Path(raw)
    if not p.is_file():
        raise Stage3ConfigError(f"Config not found: {p.resolve()}")
    try:
        import yaml
    except ImportError as e:
        raise Stage3ConfigError("Install PyYAML: pip install pyyaml") from e
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    try:
        return Stage3Policy.model_validate(data)
    except ValidationError as e:
        raise Stage3ConfigError(_fmt(e)) from e
