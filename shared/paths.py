from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ProjectPaths:
    """Resolved filesystem locations from config `paths` and project root."""

    root: Path
    raw_dir: Path
    interim_dir: Path
    labeled_dir: Path
    reports_dir: Path
    models_dir: Path

    @classmethod
    def from_config(cls, root: Path, cfg: dict[str, Any]) -> ProjectPaths:
        p = cfg.get("paths") or {}
        return cls(
            root=root,
            raw_dir=root / p.get("raw_dir", "data/raw"),
            interim_dir=root / p.get("interim_dir", "data/interim"),
            labeled_dir=root / p.get("labeled_dir", "data/labeled"),
            reports_dir=root / p.get("reports_dir", "reports"),
            models_dir=root / p.get("models_dir", "models"),
        )

    def ensure(self) -> None:
        for d in (
            self.raw_dir,
            self.interim_dir,
            self.labeled_dir,
            self.reports_dir,
            self.models_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
