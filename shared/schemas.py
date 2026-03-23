"""Unified dataframe schema (course contract)."""

from __future__ import annotations

COLLECTION_COLUMNS: list[str] = [
    "uid",
    "text",
    "audio",
    "image",
    "label",
    "source",
    "collected_at",
    "language",
    "source_type",
    "url",
    "meta",
]
