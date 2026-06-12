"""Shared text normalization for Stage 3 dedup / near-duplicate (reused by dedup + qc_dedup_cluster)."""

from __future__ import annotations

import re
from typing import Literal

NearNormMode = Literal["whitespace_lower", "alnum_lower"]


def normalize_exact_text(text: str, *, normalize_whitespace: bool = True) -> str:
    """Whitespace normalization for exact-hash dedup (same contract as legacy ``dedup._norm``)."""
    t = text or ""
    if normalize_whitespace:
        t = re.sub(r"\s+", " ", t.strip())
    return t


def normalize_for_near_duplicate(text: str, mode: NearNormMode = "whitespace_lower") -> str:
    """
    Stronger normalization for similarity / near-duplicate (configurable).

    - ``whitespace_lower``: collapse spaces + case-fold.
    - ``alnum_lower``: Stage1-style — lower, strip punctuation, collapse tokens.
    """
    t = normalize_exact_text(text, normalize_whitespace=True)
    if mode == "whitespace_lower":
        return t.lower()
    if mode == "alnum_lower":
        t = t.lower()
        t = re.sub(r"[^\w\s]", "", t, flags=re.UNICODE)
        return " ".join(t.split())
    return t
