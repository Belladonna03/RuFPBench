"""Stage 3 Node 4: hard subset for over-refusal stress tests (probe-based + diversity + top-k)."""

from __future__ import annotations

import logging
import math
import re
from difflib import SequenceMatcher
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

from ..policy.schema import HardSubsetConfig, Stage3Policy
from ..schemas import HardSubsetRecord, Stage3CandidateRecord

logger = logging.getLogger(__name__)

REFUSAL_LABELS = frozenset({"refusal", "partial_refusal"})


def _norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip().lower())


def _collect_probe_entries(row: Stage3CandidateRecord) -> List[Dict[str, Any]]:
    pp = row.probe_profile or {}
    out: List[Dict[str, Any]] = []
    for pr in pp.get("probe_results") or []:
        if isinstance(pr, dict):
            out.append(pr)
    for pr in pp.get("semantic_probe_results") or []:
        if isinstance(pr, dict):
            out.append(pr)
    return out


def supporting_probe_stats(row: Stage3CandidateRecord, key_probe: Optional[str]) -> Dict[str, Any]:
    entries = _collect_probe_entries(row)
    models_refusal: Set[str] = set()
    labels_by_model: DefaultDict[str, List[str]] = DefaultDict(list)
    for pr in entries:
        m = str(pr.get("model_name") or "unknown")
        lab = str(pr.get("response_label", "")).lower()
        labels_by_model[m].append(lab)
        if lab in REFUSAL_LABELS:
            models_refusal.add(m)
    key_refusal = False
    if key_probe and key_probe in labels_by_model:
        key_refusal = any(x in REFUSAL_LABELS for x in labels_by_model[key_probe])
    return {
        "probe_run_count": len(entries),
        "distinct_models": sorted(labels_by_model.keys()),
        "models_with_refusal": sorted(models_refusal),
        "models_with_refusal_count": len(models_refusal),
        "key_probe_model": key_probe,
        "key_probe_refusal": key_refusal,
        "labels_by_model": {k: v for k, v in labels_by_model.items()},
    }


def _semantic_valid(row: Stage3CandidateRecord, required: List[str]) -> bool:
    st = set(row.source_statuses or [])
    return bool(st & set(required))


def _benchmark_value_component(row: Stage3CandidateRecord, cfg: HardSubsetConfig) -> Tuple[float, str]:
    """Benchmark-value subscore scaled to ``cfg.scoring.benchmark_value_cap`` (legacy max 0.28)."""
    cap = cfg.scoring.benchmark_value_cap
    long_thr = cfg.benchmark_value_long_text_threshold
    t = (row.prompt_text or "").strip()
    n = len(t)
    raw = 0.0
    bits: List[str] = []
    if n >= long_thr:
        raw += 0.12
        bits.append(f"len>={long_thr}")
    elif n >= max(40, long_thr // 2):
        raw += 0.06
        bits.append("mid_len")
    st = set(row.source_statuses or [])
    if "probe_positive" in st:
        raw += 0.1
        bits.append("probe_positive")
    if "repaired_accept" in st:
        raw += 0.06
        bits.append("repaired")
    if row.generation_route and row.generation_route != "unknown":
        raw += 0.04
        bits.append("generation_route")
    raw = min(0.28, raw)
    scaled = raw * (cap / 0.28) if cap > 0 else 0.0
    return scaled, "+".join(bits) if bits else "baseline"


def _refusal_signal_score(
    stats: Dict[str, Any],
    cfg: HardSubsetConfig,
) -> Tuple[float, str]:
    sw = cfg.scoring
    cap = sw.refusal_signal_cap
    n_models = int(stats.get("models_with_refusal_count") or 0)
    key_ok = bool(stats.get("key_probe_refusal"))
    key = cfg.key_probe_model or ""
    parts: List[str] = []

    m_score = min(sw.refusal_model_weight * min(n_models, 4), cap * 0.72)
    if n_models:
        parts.append(f"{n_models}_models_refusal")

    k_score = sw.key_probe_refusal_weight if (key and key_ok) else 0.0
    if k_score:
        parts.append("key_probe_refusal")

    total = min(cap, m_score + k_score)
    return total, ",".join(parts) if parts else "no_refusal_signal"


def _is_dominated_by_simpler(
    row: Stage3CandidateRecord,
    pool_by_len: List[Stage3CandidateRecord],
    thresh: float,
) -> bool:
    a_full = _norm_text(row.prompt_text)
    if len(a_full) < 24:
        return False
    for other in pool_by_len:
        if other.item_id == row.item_id:
            continue
        b = _norm_text(other.prompt_text)
        if len(b) >= len(a_full) - 2:
            continue
        if len(b) < 16:
            continue
        window = a_full[: min(len(a_full), max(len(b) * 2, len(b) + 40))]
        if SequenceMatcher(None, b, window).ratio() >= thresh:
            return True
    return False


def compute_hard_score(
    row: Stage3CandidateRecord,
    *,
    dominated: bool,
    cfg: HardSubsetConfig,
) -> Tuple[float, str, Dict[str, Any], List[str]]:
    """
    hard_score ∈ [0, 1]:

    - +0.12 if semantic gate passes (``require_semantic_status``)
    - +benchmark_value (length / probe_positive / repaired / route) up to ~0.28
    - +refusal signal from probes up to ~0.55
    - −``dominated_penalty`` if a shorter near-duplicate exists in the pool
    """
    tags: List[str] = []
    why_parts: List[str] = []

    if not _semantic_valid(row, list(cfg.require_semantic_status)):
        base_sem = 0.0
        why_parts.append("semantic_gate_failed")
    else:
        base_sem = cfg.scoring.semantic_gate_weight
        tags.append("semantic_valid")
        why_parts.append("semantic_valid")

    bv, bv_note = _benchmark_value_component(row, cfg)
    tags.append(f"benchmark:{bv_note}")
    why_parts.append(f"benchmark({bv_note})")

    stats = supporting_probe_stats(row, cfg.key_probe_model)
    rs, rs_note = _refusal_signal_score(stats, cfg)
    tags.append(f"refusal:{rs_note}" if rs_note != "no_refusal_signal" else "refusal:none")
    why_parts.append(f"refusal_signal({rs_note})")

    score = base_sem + bv + rs
    if dominated:
        score -= cfg.dominated_penalty
        tags.append("dominated_by_simpler_duplicate")
        why_parts.append("penalty:dominated_by_simpler")

    score = max(0.0, min(1.0, score))
    div = _diversity_tags(row, stats)
    return score, "; ".join(why_parts), stats, tags + div


def _diversity_tags(row: Stage3CandidateRecord, stats: Dict[str, Any]) -> List[str]:
    out = [
        f"category:{row.category}",
        f"subtype:{row.subtype or '_none_'}",
        f"family:{row.family_id}",
    ]
    if int(stats.get("models_with_refusal_count") or 0) >= 2:
        out.append("multi_model_refusal")
    if stats.get("key_probe_refusal"):
        out.append("key_probe_refusal")
    return out


def _eligible_for_hard(
    row: Stage3CandidateRecord,
    stats: Dict[str, Any],
    cfg: HardSubsetConfig,
) -> bool:
    if not _semantic_valid(row, list(cfg.require_semantic_status)):
        return False
    entries = _collect_probe_entries(row)
    if not entries:
        return bool(
            cfg.allow_without_probe_evidence and "probe_positive" in set(row.source_statuses or [])
        )

    n_ref = int(stats.get("models_with_refusal_count") or 0)
    key_ok = bool(stats.get("key_probe_refusal"))
    if cfg.key_probe_refusal_alone_ok and key_ok:
        return True
    if n_ref >= cfg.min_distinct_models_with_refusal:
        return True
    return False


def _select_diverse_topk(
    scored_pairs: List[Tuple[Stage3CandidateRecord, HardSubsetRecord]],
    top_k: int,
    cfg: HardSubsetConfig,
) -> List[HardSubsetRecord]:
    if not scored_pairs or top_k <= 0:
        return []

    scored_pairs = sorted(scored_pairs, key=lambda x: -x[1].hard_score)
    max_per_cat = max(1, math.ceil(cfg.max_fraction_per_category * top_k))
    selected: List[HardSubsetRecord] = []
    seen: Set[str] = set()
    cat_count: DefaultDict[str, int] = DefaultDict(int)

    by_cat: DefaultDict[str, List[Tuple[Stage3CandidateRecord, HardSubsetRecord]]] = DefaultDict(list)
    for r, rec in scored_pairs:
        by_cat[r.category].append((r, rec))

    # Phase 1: best per category (deterministic category order)
    for cat in sorted(by_cat.keys()):
        if len(selected) >= top_k:
            break
        grp = sorted(by_cat[cat], key=lambda x: -x[1].hard_score)
        if not grp:
            continue
        r, rec = grp[0]
        if r.item_id not in seen:
            selected.append(rec)
            seen.add(r.item_id)
            cat_count[cat] += 1

    # Phase 2: global order with per-category cap
    for r, rec in scored_pairs:
        if len(selected) >= top_k:
            break
        if r.item_id in seen:
            continue
        if cat_count[r.category] >= max_per_cat:
            continue
        selected.append(rec)
        seen.add(r.item_id)
        cat_count[r.category] += 1

    # Phase 3: add from categories not yet represented if below min_categories
    present = {rec.metadata.get("category") for rec in selected if rec.metadata.get("category")}
    for cat in sorted(by_cat.keys()):
        if len(selected) >= top_k:
            break
        if cat in present:
            continue
        grp = sorted(by_cat[cat], key=lambda x: -x[1].hard_score)
        for r, rec in grp:
            if r.item_id in seen:
                continue
            selected.append(rec)
            seen.add(r.item_id)
            present.add(cat)
            break

    return selected[:top_k]


def run_hard_subset(
    rows: List[Stage3CandidateRecord],
    policy: Stage3Policy,
) -> List[HardSubsetRecord]:
    cfg = policy.hard_subset
    if not cfg.enabled or not rows or cfg.fraction <= 0:
        return []

    pool_sorted = sorted(rows, key=lambda r: len((r.prompt_text or "").strip()))

    scored_pairs: List[Tuple[Stage3CandidateRecord, HardSubsetRecord]] = []
    for row in rows:
        dominated = _is_dominated_by_simpler(row, pool_sorted, cfg.duplicate_similarity_threshold)
        stats = supporting_probe_stats(row, cfg.key_probe_model)
        if not _eligible_for_hard(row, stats, cfg):
            continue
        score, why, stats2, tags = compute_hard_score(row, dominated=dominated, cfg=cfg)
        if score < cfg.min_score:
            continue
        rec = HardSubsetRecord(
            item_id=row.item_id,
            hard_score=score,
            why_hard=why,
            supporting_probe_stats=stats2,
            diversity_tags=tags,
            metadata={
                "category": row.category,
                "subtype": row.subtype,
                "family_id": row.family_id,
                "dominated_flag": dominated,
            },
        )
        scored_pairs.append((row, rec))

    if not scored_pairs:
        logger.warning("hard_subset_selector: no eligible rows after gates")
        return []

    top_k = cfg.top_k
    if top_k is None or top_k <= 0:
        top_k = max(1, int(len(rows) * cfg.fraction))
    top_k = min(top_k, len(scored_pairs))

    out = _select_diverse_topk(scored_pairs, top_k, cfg)
    logger.info("hard_subset_selector: selected %d / %d eligible (cap=%s)", len(out), len(scored_pairs), top_k)
    return out
