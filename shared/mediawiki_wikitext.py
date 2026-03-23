"""
Extract plain text lines from MediaWiki wikitext (no HTML DOM parsing).

Used with action=parse & prop=wikitext from the MediaWiki API.
"""

from __future__ import annotations

import re
from typing import Any

# Skip Category/File/interwiki-style link targets when extracting [[...]] as list items.
_NS_SKIP = frozenset(
    {
        "category",
        "категория",
        "file",
        "файл",
        "image",
        "приложение",
        "special",
        "участник",
        "media",
    }
)


def _strip_wiki_markup(s: str) -> str:
    s = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", s)
    s = re.sub(r"\[\[([^\]]+)\]\]", r"\1", s)
    s = re.sub(r"\{\{[^\}]+\}\}", "", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"''+", "", s)
    return s.strip()


def _link_title_namespace(title: str) -> str | None:
    """Return lowercase namespace before ':' if present, else None."""
    if ":" not in title:
        return None
    return title.split(":", 1)[0].strip().lower()


def _should_skip_link_target(title: str) -> bool:
    # Leading ":" means hide namespace in output; strip before NS checks.
    t = title.strip().lstrip(":").strip()
    if not t:
        return True
    ns = _link_title_namespace(t)
    if ns and ns in _NS_SKIP:
        return True
    # Interwiki / language prefix: [[en:...]], [[:en:...]]
    if re.match(r"^[a-z]{2,3}:", t, re.I):
        return True
    return False


def _first_meaningful_wikilink_text(line: str) -> str | None:
    """First [[...]] on the line whose target is not Category/File/interwiki."""
    for m in re.finditer(r"\[\[(?:([^|\]]+)\|)?([^\]]+)\]\]", line):
        left = m.group(1)
        right = m.group(2).strip()
        if left:
            title = left.strip()
            display = right
        else:
            title = right
            display = right
        if _should_skip_link_target(title):
            continue
        return _strip_wiki_markup(display)
    return None


def extract_list_items_from_wikitext(wikitext: str, **kwargs: Any) -> list[str]:
    """
    Parser mode ``mediawiki_wikitext_list``: lines that look like list items (* # : ;),
    wikitable rows with ``||`` / ``|`` cells, or standalone lines with meaningful ``[[...]]``.

    Optional kwargs:
    - ``include_nonlist_lines``: if True, also include non-empty lines that are not templates-only.
    """
    include_plain = bool(kwargs.get("include_nonlist_lines", False))

    lines = wikitext.splitlines()
    items: list[str] = []

    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("{|") or s.startswith("|}") or s.startswith("|-"):
            continue
        if re.match(r"^\s*!\s*", s):
            continue
        if re.match(r"^\{\{", s):
            continue
        if s.startswith("==") and s.endswith("=="):
            continue

        raw: str | None = None
        m = re.match(r"^[*#:;]+\s*(.+)$", s)
        if m:
            raw = m.group(1).strip()
        elif s.startswith("|") and "[[" in s:
            # Wikitable data row: ``|| [[phrase]] || gloss ...`` or ``| [[...]]``
            t = _first_meaningful_wikilink_text(s)
            if t:
                raw = t
        elif "[[" in s:
            t = _first_meaningful_wikilink_text(s)
            if t:
                raw = t
        elif include_plain and not s.startswith("|"):
            raw = s

        if raw is None:
            continue
        raw = _strip_wiki_markup(raw)
        if raw:
            items.append(raw)

    return items


WIKITEXT_MODES = {
    "mediawiki_wikitext_list": extract_list_items_from_wikitext,
}


def apply_wikitext_mode(mode: str, wikitext: str, **kwargs: Any) -> list[str]:
    if mode not in WIKITEXT_MODES:
        raise ValueError(f"Unknown wikitext parse_mode {mode!r}; use one of {list(WIKITEXT_MODES)}")
    return WIKITEXT_MODES[mode](wikitext, **kwargs)


__all__ = ["extract_list_items_from_wikitext", "apply_wikitext_mode", "WIKITEXT_MODES"]
