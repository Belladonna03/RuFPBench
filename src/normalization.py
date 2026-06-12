"""Lightweight text normalization (original text preserved elsewhere)."""

from __future__ import annotations

import re

_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """
    Trim, collapse repeated spaces, normalize newlines.
    Does not strip punctuation; does not lowercase (callers add lowercase for dedup layer).
    """
    if not text:
        return ""
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    s = _MULTI_NEWLINE.sub("\n\n", s)
    lines = s.split("\n")
    lines = [ln.strip() for ln in lines]
    s = "\n".join(lines)
    s = _MULTI_SPACE.sub(" ", s)
    return s.strip()


def normalize_for_dedup(text: str) -> str:
    """Trim + lowercase + collapse spaces (canonical/dedup layer)."""
    return normalize_text(text).lower()


# Edge punctuation stripped in rough_word_tokens (keyphrase stopword heuristics).
_TOKEN_EDGE_STRIP = ".,!?;:()[]{}'\"«»„“”…—–-*_^`"


def rough_word_tokens(text: str) -> list[str]:
    """
    Split normalized-ish text into lowercase tokens (whitespace-separated),
    with light stripping of common edge punctuation. Used for stopword-only checks.
    """
    if not text:
        return []
    out: list[str] = []
    for part in text.lower().split():
        w = part.strip(_TOKEN_EDGE_STRIP)
        if w:
            out.append(w)
    return out
