"""
Co-occurrence pseudo-graph: same_prompt, same_sentence, nearby_window.

Edge aggregation (MVP, deterministic):
- At most one edge per (unordered node pair, edge_type).
- same_prompt: uniform weight — all node pairs in the document are linked once (co-occurrence in
  the same user prompt). Weight is fixed 1.0 (explainable; not a probability).
- same_sentence: one edge per pair if mentions from both nodes share ≥1 sentence; evidence carries
  sorted unique sentence_ids; sentence_id is min(sentence_ids) for single-int consumers.
- nearby_window: one edge per pair if any mention interval is within `nearby_window_chars` gap
  (half-open spans, minimum gap over all mention pairs); char_distance is that minimum gap;
  weight = linear decay from 1.0 at gap 0 to a small floor at the window boundary.

Final edge list is sorted by (source_node_id, target_node_id, edge_type) for stable JSONL diffs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.schemas import Edge, EdgeEvidence, MergedNode, RawMention


@dataclass(frozen=True)
class SentenceSpan:
    sentence_id: int
    start: int
    end: int


def split_sentences_with_spans(text: str) -> list[SentenceSpan]:
    """
    Rule-based sentence spans over original character indices (MVP).

    Boundaries:
    - Newlines: sentence ends at newline; consecutive newlines collapse into one boundary (empty
      lines do not produce empty sentence records).
    - Punctuation runs: one or more of . ! ? … ending a sentence when followed by whitespace or EOF.
    - Trimming: leading/trailing whitespace inside a span is removed; wholly empty segments are dropped.
    - sentence_id is 0..N-1 in document order over non-empty segments only.
    """
    if not text:
        return []

    t = text.replace("\r\n", "\n").replace("\r", "\n")
    n = len(t)
    raw_ranges: list[tuple[int, int]] = []
    seg_start = 0
    i = 0

    def push_range(end_exc: int) -> None:
        nonlocal seg_start
        lo, hi = seg_start, end_exc
        while lo < hi and t[lo].isspace():
            lo += 1
        while lo < hi and t[hi - 1].isspace():
            hi -= 1
        if lo < hi:
            raw_ranges.append((lo, hi))
        seg_start = end_exc

    while i < n:
        c = t[i]
        if c == "\n":
            push_range(i)
            i += 1
            while i < n and t[i] == "\n":
                i += 1
            seg_start = i
            continue
        if c in ".!?…":
            j = i
            while j < n and t[j] in ".!?…":
                j += 1
            if j >= n or t[j].isspace():
                push_range(j)
                i = j
                while i < n and t[i].isspace():
                    i += 1
                seg_start = i
                continue
        i += 1

    push_range(n)

    if not raw_ranges and t.strip():
        lo, hi = 0, n
        while lo < hi and t[lo].isspace():
            lo += 1
        while lo < hi and t[hi - 1].isspace():
            hi -= 1
        if lo < hi:
            raw_ranges.append((lo, hi))

    return [
        SentenceSpan(sentence_id=k, start=a, end=b)
        for k, (a, b) in enumerate(raw_ranges)
    ]


def interval_gap_half_open(s1: int, e1: int, s2: int, e2: int) -> int:
    """Minimum characters between two half-open [start, end) spans; 0 if overlapping/touching."""
    if e1 <= s2:
        return s2 - e1
    if e2 <= s1:
        return s1 - e2
    return 0


def char_span_to_sentence(
    start: Optional[int],
    end: Optional[int],
    sentences: list[SentenceSpan],
) -> Optional[int]:
    """
    Map a character span to a sentence_id using maximum overlap with sentence spans;
    fallback to sentence containing the span midpoint, then nearest boundary.
    """
    if start is None or end is None or not sentences:
        return None
    s0, e0 = (start, end) if start <= end else (end, start)

    best_id: Optional[int] = None
    best_ov = -1
    for sp in sentences:
        ov = max(0, min(e0, sp.end) - max(s0, sp.start))
        if ov > best_ov:
            best_ov = ov
            best_id = sp.sentence_id
    if best_ov > 0 and best_id is not None:
        return best_id

    mid = (s0 + e0) // 2
    for sp in sentences:
        if sp.start <= mid < sp.end:
            return sp.sentence_id

    best_d: Optional[tuple[int, int]] = None
    for sp in sentences:
        d = min(abs(s0 - sp.start), abs(s0 - sp.end), abs(e0 - sp.start), abs(e0 - sp.end))
        if best_d is None or d < best_d[0]:
            best_d = (d, sp.sentence_id)
    return best_d[1] if best_d else None


def mention_sentence_map(
    mentions: list[RawMention],
    sentences: list[SentenceSpan],
) -> list[tuple[RawMention, Optional[int]]]:
    return [(m, char_span_to_sentence(m.start, m.end, sentences)) for m in mentions]


def _node_sentence_id_sets(
    nodes: list[MergedNode],
    sentences: list[SentenceSpan],
) -> dict[str, set[int]]:
    """Union of sentence_ids over all mentions of each node (with offsets)."""
    out: dict[str, set[int]] = {}
    for n in nodes:
        sids: set[int] = set()
        for m, sid in mention_sentence_map(n.raw_mentions, sentences):
            if sid is not None:
                sids.add(sid)
        out[n.node_id] = sids
    return out


def _mention_intervals_by_node(nodes: list[MergedNode]) -> dict[str, list[tuple[int, int]]]:
    """Half-open [start, end) intervals per node from mentions that have offsets."""
    by_id: dict[str, list[tuple[int, int]]] = {}
    for n in nodes:
        spans: list[tuple[int, int]] = []
        for m in n.raw_mentions:
            if m.start is not None and m.end is not None and m.end > m.start:
                spans.append((m.start, m.end))
        if spans:
            by_id[n.node_id] = spans
    return by_id


def _nearby_weight(min_gap: int, window: int) -> float:
    """Linear decay: gap 0 → 1.0; gap == window → floor; beyond window not called."""
    w = max(window, 1)
    # At gap 0 weight 1.0; at gap w weight 0.05 floor
    return max(0.05, 1.0 - (min_gap / w))


def _edge_sort_key(e: Edge) -> tuple[str, str, str]:
    a, b = sorted([e.source_node_id, e.target_node_id])
    return (a, b, e.edge_type)


def build_cooccurrence_edges(
    full_text: str,
    nodes: list[MergedNode],
    nearby_window_chars: int = 64,
) -> list[Edge]:
    sentences = split_sentences_with_spans(full_text)
    node_ids = sorted(n.node_id for n in nodes)

    node_sentences = _node_sentence_id_sets(nodes, sentences)
    intervals = _mention_intervals_by_node(nodes)

    edges: list[Edge] = []

    # --- same_prompt: one edge per unordered pair, fixed weight ---
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            u, v = node_ids[i], node_ids[j]
            edges.append(
                Edge(
                    source_node_id=u,
                    target_node_id=v,
                    edge_type="same_prompt",
                    weight=1.0,
                    evidence=EdgeEvidence(note="same_prompt"),
                )
            )

    # --- same_sentence: one edge per pair, aggregate all shared sentence indices ---
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            u, v = node_ids[i], node_ids[j]
            common = node_sentences.get(u, set()) & node_sentences.get(v, set())
            if not common:
                continue
            ids_sorted = sorted(common)
            edges.append(
                Edge(
                    source_node_id=u,
                    target_node_id=v,
                    edge_type="same_sentence",
                    weight=1.0,
                    evidence=EdgeEvidence(
                        sentence_ids=ids_sorted,
                        sentence_id=ids_sorted[0],
                        note="same_sentence",
                    ),
                )
            )

    # --- nearby_window: min gap over all mention-interval pairs, one edge per node pair ---
    wchars = max(1, nearby_window_chars)
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            u, v = node_ids[i], node_ids[j]
            spans_u = intervals.get(u, [])
            spans_v = intervals.get(v, [])
            if not spans_u or not spans_v:
                continue
            best: Optional[int] = None
            for s1, e1 in sorted(spans_u):
                for s2, e2 in sorted(spans_v):
                    g = interval_gap_half_open(s1, e1, s2, e2)
                    if g <= wchars:
                        if best is None or g < best:
                            best = g
            if best is None:
                continue
            edges.append(
                Edge(
                    source_node_id=u,
                    target_node_id=v,
                    edge_type="nearby_window",
                    weight=_nearby_weight(best, wchars),
                    evidence=EdgeEvidence(
                        char_distance=best,
                        note="nearby_window",
                    ),
                )
            )

    edges.sort(key=_edge_sort_key)
    return edges
