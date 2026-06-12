"""Stage 3 Node 2: QC, exact dedup, near-duplicate clustering within category/subtype."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, DefaultDict, Dict, List, Set, Tuple

from ..policy.schema import Stage3Policy
from ..schemas import DedupClusterRecord, QCLabelRecord, Stage3CandidateRecord
from ..text_normalization import normalize_exact_text, normalize_for_near_duplicate

logger = logging.getLogger(__name__)

QcLabel = str


def _policy_fingerprint(fragment: Dict[str, Any]) -> str:
    raw = json.dumps(fragment, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _qc_heuristic(
    r: Stage3CandidateRecord,
    min_len: int,
    max_repeat: int,
    injection_patterns: List[str],
) -> Tuple[QcLabel, List[str]]:
    flags: List[str] = []
    text = (r.prompt_text or "").strip()
    if not text:
        return "low_value", ["empty_text"]
    if len(text) < min_len:
        flags.append("too_short")
        return "low_value", flags
    if max_repeat > 0:
        pattern = r"(.)\1{" + str(max_repeat - 1) + r",}"
        if re.search(pattern, text):
            flags.append("long_char_repeat")
            return "noisy", flags
    for i, pat in enumerate(injection_patterns):
        if not pat:
            continue
        try:
            if re.search(pat, text):
                flags.append(f"injection_pattern[{i}]")
                return "noisy", flags
        except re.error as e:
            logger.warning("invalid qc_dedup.injection_patterns[%d]: %s", i, e)
    if r.source and "review_queue" in r.source:
        flags.append("from_review_queue")
    return "keep", flags


def _dedup_bucket_key(r: Stage3CandidateRecord, strata: List[str]) -> Tuple[str, ...]:
    parts: List[str] = []
    for f in strata:
        if f == "category":
            parts.append(r.category)
        elif f == "subtype":
            parts.append(r.subtype or "")
        else:
            parts.append("")
    return tuple(parts)


def _similarity(a: str, b: str, metric: str) -> float:
    if metric == "jaccard_words":
        wa, wb = set(a.split()), set(b.split())
        if not wa and not wb:
            return 1.0
        u = wa | wb
        if not u:
            return 1.0
        return len(wa & wb) / len(u)
    return SequenceMatcher(None, a, b).ratio()


class _DSU:
    __slots__ = ("p",)

    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _finalize(
    rows: List[Stage3CandidateRecord],
    final_labels: Dict[str, QcLabel],
    flags_by_id: Dict[str, List[str]],
    clusters_out: List[DedupClusterRecord],
) -> Tuple[List[QCLabelRecord], List[DedupClusterRecord], List[Stage3CandidateRecord]]:
    cluster_for_item: Dict[str, str] = {}
    for c in clusters_out:
        for mid in c.member_item_ids:
            cluster_for_item[mid] = c.cluster_id

    qc_rows: List[QCLabelRecord] = []
    for r in rows:
        lbl = final_labels.get(r.item_id, "keep")
        fl = flags_by_id.get(r.item_id, [])
        qc_rows.append(
            QCLabelRecord(
                item_id=r.item_id,
                qc_label=lbl,
                qc_flags=fl,
                notes=",".join(fl) if fl else lbl,
                original_prompt_id=r.original_prompt_id,
                cluster_id=cluster_for_item.get(r.item_id),
            )
        )

    pass_ids = {q.item_id for q in qc_rows if q.qc_pass}
    pool = [r for r in rows if r.item_id in pass_ids]

    logger.info("qc_dedup_cluster: labels=%d clusters=%d pool_out=%d", len(qc_rows), len(clusters_out), len(pool))
    return qc_rows, clusters_out, pool


def run_qc_dedup_cluster(
    rows: List[Stage3CandidateRecord],
    policy: Stage3Policy,
) -> Tuple[List[QCLabelRecord], List[DedupClusterRecord], List[Stage3CandidateRecord]]:
    """
    QC labels for every input row (with ``original_prompt_id``), multi-member dedup clusters,
    and filtered pool (``keep`` + ``cluster_representative``).
    """
    cfg = policy.qc_dedup
    dedup_cfg = policy.dedup
    fp = _policy_fingerprint(
        {
            "qc_dedup": cfg.model_dump(),
            "dedup": {"method": dedup_cfg.method, "normalize_whitespace": dedup_cfg.normalize_whitespace},
        }
    )
    seed = cfg.clustering_seed
    strata = list(cfg.dedup_strata) if cfg.dedup_strata else ["category", "subtype"]
    inj = list(cfg.injection_patterns) if cfg.injection_patterns else [r"(?i)\bignore previous\b"]

    final_labels: Dict[str, QcLabel] = {}
    flags_by_id: Dict[str, List[str]] = {}
    for r in rows:
        lbl, fl = _qc_heuristic(r, cfg.min_text_len, cfg.max_repeat_char_run, inj)
        final_labels[r.item_id] = lbl
        flags_by_id[r.item_id] = fl

    structural = dedup_cfg.method not in ("none", "skip", "")
    clusters_out: List[DedupClusterRecord] = []

    if not structural:
        return _finalize(rows, final_labels, flags_by_id, clusters_out)

    active = [r for r in rows if final_labels[r.item_id] == "keep"]
    buckets: DefaultDict[Tuple[str, ...], List[Stage3CandidateRecord]] = defaultdict(list)
    for r in active:
        buckets[_dedup_bucket_key(r, strata)].append(r)

    near_mode = cfg.near_normalization
    if near_mode not in ("whitespace_lower", "alnum_lower"):
        near_mode = "whitespace_lower"
    metric = cfg.near_duplicate_metric
    if metric not in ("difflib_ratio", "jaccard_words"):
        metric = "difflib_ratio"
    thr = cfg.near_duplicate_threshold

    was_exact_multi_rep: Set[str] = set()

    for key in sorted(buckets.keys(), key=lambda x: x):
        bucket_rows = buckets[key]
        if not bucket_rows:
            continue
        r0 = bucket_rows[0]
        cat = r0.category
        sub = (r0.subtype or None) if "subtype" in strata else None
        exact_groups: DefaultDict[str, List[Stage3CandidateRecord]] = defaultdict(list)
        for r in bucket_rows:
            etext = normalize_exact_text(r.prompt_text, normalize_whitespace=dedup_cfg.normalize_whitespace)
            ek = hashlib.sha256(etext.encode("utf-8")).hexdigest()
            exact_groups[ek].append(r)

        survivors: List[Stage3CandidateRecord] = []

        for ek in sorted(exact_groups.keys()):
            grp = sorted(exact_groups[ek], key=lambda x: x.item_id)
            if len(grp) == 1:
                survivors.append(grp[0])
                continue
            rep = grp[0]
            was_exact_multi_rep.add(rep.item_id)
            cid = f"ex-{ek[:12]}"
            clusters_out.append(
                DedupClusterRecord(
                    cluster_id=cid,
                    representative_item_id=rep.item_id,
                    member_item_ids=[x.item_id for x in grp],
                    method="sha256_exact_text",
                    cluster_kind="exact",
                    category=cat,
                    subtype=sub if sub else None,
                    similarity_threshold=None,
                    config_fingerprint=fp,
                    clustering_seed=seed,
                )
            )
            for x in grp:
                if x.item_id != rep.item_id:
                    final_labels[x.item_id] = "exact_duplicate"
            final_labels[rep.item_id] = "keep"  # placeholder until near pass (single-survivor case)
            survivors.append(rep)

        n = len(survivors)
        if n == 0:
            continue
        if n == 1:
            sid = survivors[0].item_id
            if sid in was_exact_multi_rep:
                final_labels[sid] = "cluster_representative"
            else:
                final_labels[sid] = "keep"
            continue

        texts = [
            normalize_for_near_duplicate(s.prompt_text, mode=near_mode)  # type: ignore[arg-type]
            for s in survivors
        ]
        dsu = _DSU(n)
        for i in range(n):
            for j in range(i + 1, n):
                if _similarity(texts[i], texts[j], metric) >= thr:
                    dsu.union(i, j)

        components: DefaultDict[int, List[int]] = defaultdict(list)
        for i in range(n):
            components[dsu.find(i)].append(i)

        for _, idxs in sorted(components.items(), key=lambda kv: min(survivors[i].item_id for i in kv[1])):
            idxs_sorted = sorted(idxs, key=lambda i: survivors[i].item_id)
            if len(idxs_sorted) == 1:
                i0 = idxs_sorted[0]
                sid = survivors[i0].item_id
                if sid in was_exact_multi_rep:
                    final_labels[sid] = "cluster_representative"
                else:
                    final_labels[sid] = "keep"
                continue

            rep_i = min(idxs_sorted, key=lambda i: survivors[i].item_id)
            rep_id = survivors[rep_i].item_id
            cid = f"nr-{hashlib.sha256(f'{cat}|{sub}|{rep_id}|{fp}'.encode()).hexdigest()[:12]}"
            clusters_out.append(
                DedupClusterRecord(
                    cluster_id=cid,
                    representative_item_id=rep_id,
                    member_item_ids=[survivors[i].item_id for i in idxs_sorted],
                    method=f"near_{metric}",
                    cluster_kind="near",
                    category=cat,
                    subtype=sub if sub else None,
                    similarity_threshold=thr,
                    config_fingerprint=fp,
                    clustering_seed=seed,
                )
            )
            for i in idxs_sorted:
                sid = survivors[i].item_id
                if sid == rep_id:
                    final_labels[sid] = "cluster_representative"
                else:
                    final_labels[sid] = "near_duplicate"

    return _finalize(rows, final_labels, flags_by_id, clusters_out)
