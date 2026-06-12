"""Load and validate ``configs/stage25.yaml``."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

from pydantic import ValidationError

from .schema import Stage25Policy

DEFAULT_REL_PATH = os.path.join("configs", "stage25.yaml")


class Stage25ConfigError(ValueError):
    """Raised when YAML is missing, invalid, or fails Pydantic validation."""

    def __init__(self, message: str, *, errors: Optional[list] = None) -> None:
        super().__init__(message)
        self.errors = errors


def _format_validation_error(exc: ValidationError) -> str:
    lines = ["Stage 2.5 config validation failed:"]
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()))
        msg = err.get("msg", "")
        typ = err.get("type", "")
        lines.append(f"  - {loc}: {msg} ({typ})")
    return "\n".join(lines)


def load_stage25_policy(path: Optional[Union[str, Path]] = None) -> Stage25Policy:
    """
    Load YAML from ``path`` or ``RUFP_STAGE25_CONFIG`` or ``configs/stage25.yaml`` under cwd.

    Raises:
        Stage25ConfigError: unreadable file, bad YAML, or schema errors.
    """
    raw_path = path or os.environ.get("RUFP_STAGE25_CONFIG") or DEFAULT_REL_PATH
    p = Path(raw_path)
    if not p.is_file():
        raise Stage25ConfigError(
            f"Stage 2.5 config file not found: {p.resolve()}. "
            f"Set RUFP_STAGE25_CONFIG or pass config_path=."
        )
    text = p.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError as e:
        raise Stage25ConfigError(
            "PyYAML is required to load Stage 2.5 config. Install: pip install pyyaml"
        ) from e

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise Stage25ConfigError(f"Invalid YAML in {p}: {e}") from e

    if data is None:
        data = {}

    try:
        return Stage25Policy.model_validate(data)
    except ValidationError as e:
        raise Stage25ConfigError(_format_validation_error(e), errors=e.errors()) from e
