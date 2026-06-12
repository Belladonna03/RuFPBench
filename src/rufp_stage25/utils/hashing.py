"""Stable UTF-8 text hashing for lineage / audit."""

from __future__ import annotations

import hashlib


def text_sha256(text: str) -> str:
    """Return hex digest of SHA-256 over UTF-8 bytes (machine-readable, stable)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
