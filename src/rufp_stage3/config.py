"""Stage 3 defaults and artifact paths (no extra deps beyond stdlib)."""

from __future__ import annotations

import os
from pathlib import Path


def effective_artifacts_root() -> str:
    """Prefer ``RUFP_ARTIFACTS_ROOT``, then ``RUFP_S3_ARTIFACTS_ROOT``, else cwd."""
    return (
        os.environ.get("RUFP_ARTIFACTS_ROOT")
        or os.environ.get("RUFP_S3_ARTIFACTS_ROOT")
        or "."
    )


def stage3_run_dir(run_id: str) -> Path:
    """Output directory: ``<root>/artifacts/stage3/<run_id>``."""
    return Path(effective_artifacts_root()) / "artifacts" / "stage3" / run_id


def ensure_stage3_run_dir(run_id: str) -> Path:
    p = stage3_run_dir(run_id)
    p.mkdir(parents=True, exist_ok=True)
    return p
