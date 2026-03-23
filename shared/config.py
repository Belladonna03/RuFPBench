from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _try_load_dotenv(anchor: Path | None = None) -> None:
    """
    Load the first ``.env`` found walking up from ``anchor``'s directory (if given),
    then ``./.env`` (cwd), then ``.env`` next to the project root (parent of ``shared/``).

    Hugging Face Hub reads ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` from ``os.environ``;
    without this, a token only present in a file is never seen by ``datasets`` / ``huggingface_hub``.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if anchor is not None:
        ap = anchor.resolve()
        cur = ap.parent if ap.is_file() else ap
        for _ in range(12):
            env_file = cur / ".env"
            if env_file.is_file():
                load_dotenv(env_file, override=False)
                return
            if cur.parent == cur:
                break
            cur = cur.parent
    cwd_env = Path.cwd() / ".env"
    if cwd_env.is_file():
        load_dotenv(cwd_env, override=False)
    # Cwd-independent fallback: repo root is parent of ``shared/`` (where this file lives).
    repo_env = Path(__file__).resolve().parent.parent / ".env"
    if repo_env.is_file():
        load_dotenv(repo_env, override=False)


def _subst_env(s: str) -> str:
    def repl(m: re.Match[str]) -> str:
        key = m.group(1).strip()
        return os.environ.get(key, "")

    return _ENV_PATTERN.sub(repl, s)


def _walk_subst(obj: Any) -> Any:
    if isinstance(obj, str):
        return _subst_env(obj)
    if isinstance(obj, dict):
        return {k: _walk_subst(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_subst(x) for x in obj]
    return obj


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    _try_load_dotenv(p)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return _walk_subst(raw)


def as_config_dict(config: str | Path | dict[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        _try_load_dotenv(None)
        return config
    return load_config(config)
