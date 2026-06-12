"""Simple human-readable diff summary between two strings."""

from __future__ import annotations


def simple_change_summary(original: str, revised: str, max_snippet: int = 120) -> str:
    """
    Short summary: length delta + whether identical after strip.
    Not a full diff — keeps lineage artifacts compact.
    """
    o = original or ""
    r = revised or ""
    if o == r:
        return "no_change"
    lo, lr = len(o), len(r)
    delta = lr - lo
    sign = "+" if delta > 0 else ""
    head_o = o[:max_snippet].replace("\n", " ")
    head_r = r[:max_snippet].replace("\n", " ")
    return (
        f"len_delta={sign}{delta}; "
        f"orig_preview={head_o!r}… → repaired_preview={head_r!r}…"
    )
