from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_uid(*parts: str, length: int = 16) -> str:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return h[:length]


def dumps_meta(meta: dict[str, Any] | None) -> str | None:
    if not meta:
        return None
    return json.dumps(meta, ensure_ascii=False, sort_keys=True)


def loads_meta(meta: str | None) -> dict[str, Any]:
    if not meta:
        return {}
    try:
        out = json.loads(meta)
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        return {}


def seed_role_from_row(label: Any, meta_str: str | None) -> str | None:
    m = loads_meta(meta_str)
    sr = m.get("seed_role")
    if isinstance(sr, str) and sr:
        return sr
    if label is not None and str(label).strip():
        return str(label).strip()
    return None
