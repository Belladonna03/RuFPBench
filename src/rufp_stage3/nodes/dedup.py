"""Exact dedup by normalized text hash."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

from ..policy.schema import DedupConfig, Stage3Policy
from ..schemas import DedupClusterRecord, Stage3CandidateRecord
from ..text_normalization import normalize_exact_text


def _norm(text: str, cfg: DedupConfig) -> str:
    return normalize_exact_text(text, normalize_whitespace=cfg.normalize_whitespace)


def run_dedup(
    rows: List[Stage3CandidateRecord],
    policy: Stage3Policy,
) -> Tuple[List[Stage3CandidateRecord], List[DedupClusterRecord]]:
    cfg = policy.dedup
    if cfg.method in ("none", "skip"):
        return rows, []

    seen: Dict[str, str] = {}
    clusters: List[DedupClusterRecord] = []
    kept: List[Stage3CandidateRecord] = []
    members_map: Dict[str, List[str]] = {}

    for r in rows:
        key = hashlib.sha256(_norm(r.text, cfg).encode("utf-8")).hexdigest()
        if key not in seen:
            seen[key] = r.item_id
            members_map[key] = [r.item_id]
            kept.append(r)
        else:
            members_map[key].append(r.item_id)

    for key, rep in seen.items():
        mids = members_map.get(key, [rep])
        clusters.append(
            DedupClusterRecord(
                cluster_id=f"cl-{key[:12]}",
                representative_item_id=rep,
                member_item_ids=sorted(set(mids)),
                method="sha256_text",
            )
        )
    return kept, clusters
