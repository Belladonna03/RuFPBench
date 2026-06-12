"""Merge entities + keyphrases into MergedNode list with layered dedup."""

from __future__ import annotations

import hashlib
from typing import Iterable

from src.schemas import MergedNode, RawEntity, RawKeyphrase, RawMention


def stable_node_id(doc_id: str, canonical_text: str, labels: list[str]) -> str:
    """
    Deterministic id: same doc + canonical surface + label set → same node_id.
    Short hex suffix from SHA-256 (readable, stable across runs).
    """
    label_key = "|".join(sorted(labels))
    payload = f"{doc_id}\n{canonical_text}\n{label_key}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"n_{digest}"


def _mention_from_entity(e: RawEntity) -> RawMention:
    return RawMention(
        text=e.text,
        label=e.label,
        extraction_source=e.extraction_source,
        score=e.score,
        start=e.start,
        end=e.end,
    )


def _mention_from_keyphrase(k: RawKeyphrase) -> RawMention:
    return RawMention(
        text=k.text,
        label=k.label,
        extraction_source=k.extraction_source,
        score=k.score,
        start=k.start,
        end=k.end,
    )


def _build_node(norm_text: str, label: str, mentions: list[RawMention]) -> MergedNode:
    labels = sorted({label})
    sources = sorted({m.extraction_source for m in mentions})
    max_score = max((m.score for m in mentions), default=0.0)
    return MergedNode(
        node_id="",
        canonical_text=norm_text,
        normalized_text=norm_text,
        raw_mentions=list(mentions),
        labels=labels,
        sources=sources,
        max_score=max_score,
        mention_count=len(mentions),
    )


def _sort_nodes_for_stable_output(nodes: list[MergedNode]) -> list[MergedNode]:
    """Stable order in JSONL: by normalized text, then joined labels."""
    return sorted(nodes, key=lambda n: (n.normalized_text, "|".join(n.labels)))


def _assign_stable_node_ids(doc_id: str, nodes: list[MergedNode]) -> list[MergedNode]:
    return [
        n.model_copy(
            update={"node_id": stable_node_id(doc_id, n.canonical_text, n.labels)}
        )
        for n in nodes
    ]


def group_by_normalized_and_label(
    entities: Iterable[RawEntity],
    keyphrases: Iterable[RawKeyphrase],
) -> list[MergedNode]:
    """First dedup level: (normalized_text, label) groups."""
    from src.normalization import normalize_for_dedup

    buckets: dict[tuple[str, str], list[RawMention]] = {}

    for e in entities:
        nt = normalize_for_dedup(e.text)
        key = (nt, e.label)
        buckets.setdefault(key, []).append(_mention_from_entity(e))

    for k in keyphrases:
        nt = normalize_for_dedup(k.text)
        key = (nt, k.label)
        buckets.setdefault(key, []).append(_mention_from_keyphrase(k))

    nodes: list[MergedNode] = []
    for (norm_text, label), mentions in buckets.items():
        nodes.append(_build_node(norm_text, label, mentions))
    return nodes


def soft_merge_by_normalized_text(nodes: list[MergedNode]) -> list[MergedNode]:
    """
    Optional second level: merge nodes sharing normalized_text across labels.
    Preserves all raw_mentions; unions labels and sources.
    """
    by_norm: dict[str, list[MergedNode]] = {}
    for n in nodes:
        by_norm.setdefault(n.normalized_text, []).append(n)

    merged: list[MergedNode] = []
    for norm, group in by_norm.items():
        if len(group) == 1:
            merged.append(group[0])
            continue
        all_mentions: list[RawMention] = []
        label_set: set[str] = set()
        source_set: set[str] = set()
        for n in group:
            all_mentions.extend(n.raw_mentions)
            label_set.update(n.labels)
            source_set.update(n.sources)
        max_score = max((m.score for m in all_mentions), default=0.0)
        merged.append(
            MergedNode(
                node_id="",
                canonical_text=norm,
                normalized_text=norm,
                raw_mentions=all_mentions,
                labels=sorted(label_set),
                sources=sorted(source_set),
                max_score=max_score,
                mention_count=len(all_mentions),
            )
        )
    return merged


def merge_entities_and_keyphrases(
    entities: Iterable[RawEntity],
    keyphrases: Iterable[RawKeyphrase],
    doc_id: str,
    soft_merge_same_normalized: bool = True,
) -> list[MergedNode]:
    nodes = group_by_normalized_and_label(entities, keyphrases)
    if soft_merge_same_normalized:
        nodes = soft_merge_by_normalized_text(nodes)
    nodes = _sort_nodes_for_stable_output(nodes)
    return _assign_stable_node_ids(doc_id, nodes)
