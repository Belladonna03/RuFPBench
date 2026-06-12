"""Stage 3 Node 3: soft balancing across strata + slice index (categories, probe, repair, vendor, NLG)."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict, deque
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from ..policy.schema import BalanceConfig, Stage3Policy
from ..schemas import BalancedRecord, Stage3CandidateRecord

logger = logging.getLogger(__name__)


def _len_bucket(nchars: int, edges: List[int]) -> str:
    edges = sorted(edges)
    prev = 0
    for e in edges:
        if nchars < e:
            return f"{prev}_{e}"
        prev = e
    return f"{prev}_inf"


def _route_key(generation_route: str) -> str:
    g = (generation_route or "").strip()
    if not g:
        return "_empty"
    first = g.split("|")[0].strip()
    return first[:120] if first else "_empty"


def _source_stage_label(r: Stage3CandidateRecord) -> str:
    ss = set(r.source_stages or [])
    if "stage25" in ss and "stage2" in ss:
        return "mixed"
    if "stage25" in ss:
        return "stage25"
    if "stage2" in ss:
        return "stage2"
    return "unknown"


def _stratum_key(r: Stage3CandidateRecord, edges: List[int]) -> str:
    sub = r.subtype or "_none_"
    lb = _len_bucket(len((r.prompt_text or "").strip()), edges)
    rk = _route_key(r.generation_route)
    st = _source_stage_label(r)
    return f"{r.category}\x1f{sub}\x1f{lb}\x1f{rk}\x1f{st}"


def _nlg_tags(text: str) -> List[str]:
    t = text or ""
    tags: List[str] = []
    if not t.strip():
        return tags
    cyr = len(re.findall(r"[\u0400-\u04FF]", t))
    ratio = cyr / max(len(t), 1)
    if ratio >= 0.35:
        tags.append("cyrillic_dense")
    words = re.findall(r"\w+", t, flags=re.UNICODE)
    if words:
        avg = sum(len(w) for w in words) / len(words)
        if avg >= 8.0:
            tags.append("long_tokens")
    punct = len(re.findall(r"[^\w\s]", t, flags=re.UNICODE))
    if punct / max(len(t), 1) >= 0.12:
        tags.append("punct_heavy")
    if len(re.split(r"[.!?…]+", t)) >= 3:
        tags.append("multi_sentence")
    return tags


def _vendor_model_names(r: Stage3CandidateRecord) -> List[str]:
    names: List[str] = []
    pp = r.probe_profile or {}
    for pr in pp.get("probe_results") or []:
        if isinstance(pr, dict) and pr.get("model_name"):
            names.append(str(pr["model_name"]))
    for pr in pp.get("semantic_probe_results") or []:
        if isinstance(pr, dict) and pr.get("model_name"):
            names.append(str(pr["model_name"]))
    return names


def _family_trim_global(
    rows: List[Stage3CandidateRecord],
    max_per_family: Optional[int],
) -> List[Stage3CandidateRecord]:
    """Cap rows per ``family_id`` across the full pool (before per-category family cap)."""
    if not max_per_family or max_per_family <= 0:
        return rows
    by_fam: DefaultDict[str, List[Stage3CandidateRecord]] = defaultdict(list)
    for r in rows:
        by_fam[r.family_id].append(r)
    kept: List[Stage3CandidateRecord] = []
    for fid, frs in by_fam.items():
        frs_sorted = sorted(frs, key=lambda x: x.item_id)
        kept.extend(frs_sorted[:max_per_family])
        if len(frs_sorted) > max_per_family:
            logger.debug(
                "global family cap: family=%s dropped %d rows",
                fid,
                len(frs_sorted) - max_per_family,
            )
    kept.sort(key=lambda x: x.item_id)
    return kept


def _family_trim(
    rows: List[Stage3CandidateRecord],
    max_per_family: Optional[int],
) -> List[Stage3CandidateRecord]:
    if not max_per_family or max_per_family <= 0:
        return rows
    by_cat: DefaultDict[str, List[Stage3CandidateRecord]] = defaultdict(list)
    for r in rows:
        by_cat[r.category].append(r)
    kept: List[Stage3CandidateRecord] = []
    for cat, grp in by_cat.items():
        by_fam: DefaultDict[str, List[Stage3CandidateRecord]] = defaultdict(list)
        for r in grp:
            by_fam[r.family_id].append(r)
        for fid, frs in by_fam.items():
            frs_sorted = sorted(frs, key=lambda x: x.item_id)
            if len(frs_sorted) <= max_per_family:
                kept.extend(frs_sorted)
            else:
                kept.extend(frs_sorted[:max_per_family])
                logger.debug(
                    "family cap: category=%s family=%s dropped %d rows",
                    cat,
                    fid,
                    len(frs_sorted) - max_per_family,
                )
    kept.sort(key=lambda x: x.item_id)
    return kept


def _stratum_category(stratum_key: str) -> str:
    return stratum_key.split("\x1f", 1)[0]


def _round_robin_strata(
    rows: List[Stage3CandidateRecord],
    cfg: BalanceConfig,
    edges: List[int],
) -> List[Stage3CandidateRecord]:
    """Soft balance: round-robin across strata; optional ``target_pool_size`` stops early (drops rest)."""
    if not rows:
        return rows
    buckets: DefaultDict[str, deque] = defaultdict(deque)
    for r in sorted(rows, key=lambda x: x.item_id):
        buckets[_stratum_key(r, edges)].append(r)

    weights = cfg.category_balance_weights or {}

    def _stratum_sort_key(k: str) -> Tuple[float, str]:
        cat = _stratum_category(k)
        w = float(weights.get(cat, 1.0))
        return (-w, k)

    order = sorted(buckets.keys(), key=_stratum_sort_key)
    selected: List[Stage3CandidateRecord] = []
    target = cfg.target_pool_size

    while any(buckets[k] for k in order):
        for k in order:
            if buckets[k]:
                selected.append(buckets[k].popleft())
                if target is not None and len(selected) >= target:
                    return selected
    return selected


def _apply_max_per_category(rows: List[Stage3CandidateRecord], cap: Optional[int]) -> List[Stage3CandidateRecord]:
    if cap is None:
        return rows
    by_cat: DefaultDict[str, List[Stage3CandidateRecord]] = defaultdict(list)
    for r in rows:
        by_cat[r.category].append(r)
    out: List[Stage3CandidateRecord] = []
    for cat in sorted(by_cat.keys()):
        grp = sorted(by_cat[cat], key=lambda x: x.item_id)
        out.extend(grp[:cap])
    return out


def _apply_min_per_category(rows: List[Stage3CandidateRecord], min_per: int, universe: List[Stage3CandidateRecord]) -> List[Stage3CandidateRecord]:
    """Ensure at least min_per per category when possible (add from universe)."""
    if min_per <= 0:
        return rows
    have = {r.item_id for r in rows}
    by_cat: DefaultDict[str, List[Stage3CandidateRecord]] = defaultdict(list)
    for r in rows:
        by_cat[r.category].append(r)
    u_by_cat: DefaultDict[str, List[Stage3CandidateRecord]] = defaultdict(list)
    for r in sorted(universe, key=lambda x: x.item_id):
        u_by_cat[r.category].append(r)

    extra: List[Stage3CandidateRecord] = []
    for cat in sorted(u_by_cat.keys()):
        need = min_per - len(by_cat.get(cat, []))
        if need <= 0:
            continue
        for r in u_by_cat[cat]:
            if r.item_id not in have and need > 0:
                extra.append(r)
                have.add(r.item_id)
                need -= 1
    if not extra:
        return rows
    merged = rows + extra
    merged.sort(key=lambda x: x.item_id)
    return merged


def build_slice_index(rows: List[Stage3CandidateRecord], policy: Stage3Policy) -> Dict[str, Any]:
    """Named slices over the balanced pool (ids only; traceability via pool rows)."""
    cfg = policy.balance
    slices: Dict[str, List[str]] = {}
    by_cat: DefaultDict[str, List[str]] = defaultdict(list)
    probe: List[str] = []
    repaired: List[str] = []
    vendor: List[str] = []
    nlg: DefaultDict[str, List[str]] = defaultdict(list)

    for r in rows:
        iid = r.item_id
        by_cat[r.category].append(iid)
        st = set(r.source_statuses or [])
        if "probe_positive" in st:
            probe.append(iid)
        if "repaired_accept" in st:
            repaired.append(iid)
        models = set(_vendor_model_names(r))
        if len(models) >= 2:
            vendor.append(iid)
        if cfg.nlg_slice_enabled:
            for tag in _nlg_tags(r.prompt_text):
                nlg[f"nlg:{tag}"].append(iid)

    for tdef in policy.target_slices or []:
        matched: List[str] = []
        for r in rows:
            if tdef.categories and r.category not in tdef.categories:
                continue
            st = set(r.source_statuses or [])
            if tdef.source_statuses_any and not (st & set(tdef.source_statuses_any)):
                continue
            matched.append(r.item_id)
        slices[tdef.name] = sorted(matched)

    for cat, ids in sorted(by_cat.items()):
        slices[f"category:{cat}"] = sorted(ids)

    slices["probe_positive"] = sorted(probe)
    slices["repaired"] = sorted(repaired)
    slices["vendor_sensitive"] = sorted(vendor)
    for k, ids in sorted(nlg.items()):
        slices[k] = sorted(ids)

    return {
        "slices": slices,
        "counts": {k: len(v) for k, v in slices.items()},
        "config_fingerprint": hashlib.sha256(
            json.dumps(policy.balance.model_dump(), sort_keys=True).encode("utf-8")
        ).hexdigest()[:12],
    }


def run_balancer_slice_builder(
    rows: List[Stage3CandidateRecord],
    policy: Stage3Policy,
) -> Tuple[List[Stage3CandidateRecord], List[BalancedRecord], Dict[str, Any]]:
    """
    Returns balanced candidates, per-row strata metadata, and slice index dict.
    """
    cfg = policy.balance
    edges = list(cfg.length_bucket_edges) if cfg.length_bucket_edges else [80, 200]

    if not rows:
        return [], [], {"slices": {}, "counts": {}}

    universe = list(rows)
    work = _family_trim_global(rows, cfg.max_items_per_family)
    work = _family_trim(work, cfg.max_rows_per_family_per_category)
    work = _apply_max_per_category(work, cfg.max_per_category)
    work = _round_robin_strata(work, cfg, edges)
    work = _apply_min_per_category(work, cfg.min_per_category, universe)

    balanced = sorted(work, key=lambda x: x.item_id)

    meta: List[BalancedRecord] = []
    for r in balanced:
        sk = _stratum_key(r, edges)
        nlg = _nlg_tags(r.prompt_text) if cfg.nlg_slice_enabled else []
        slice_membership: List[str] = []
        stset = set(r.source_statuses or [])
        if "probe_positive" in stset:
            slice_membership.append("probe_positive")
        if "repaired_accept" in stset:
            slice_membership.append("repaired")
        if len(set(_vendor_model_names(r))) >= 2:
            slice_membership.append("vendor_sensitive")
        for t in nlg:
            slice_membership.append(f"nlg:{t}")

        meta.append(
            BalancedRecord(
                item_id=r.item_id,
                category=r.category,
                stratum=sk,
                weight=1.0,
                metadata={
                    "length_bucket": _len_bucket(len((r.prompt_text or "").strip()), edges),
                    "subtype": r.subtype,
                    "generation_route_key": _route_key(r.generation_route),
                    "source_stage": _source_stage_label(r),
                    "balance_stratum": sk,
                    "original_prompt_id": r.original_prompt_id,
                    "stage1_prompt_id": r.stage1_prompt_id,
                    "family_id": r.family_id,
                    "nlg_tags": nlg,
                    "slice_membership": sorted(slice_membership),
                },
            )
        )

    slice_index = build_slice_index(balanced, policy)
    logger.info(
        "balancer_slice_builder: in=%d out=%d strata_distinct=%d",
        len(rows),
        len(balanced),
        len({m.stratum for m in meta}),
    )
    return balanced, meta, slice_index


def run_balance(
    rows: List[Stage3CandidateRecord],
    policy: Stage3Policy,
) -> Tuple[List[Stage3CandidateRecord], List[BalancedRecord]]:
    """Backward-compatible alias: balance only (no slice JSON side effect here)."""
    b, m, _ = run_balancer_slice_builder(rows, policy)
    return b, m
