"""Stage 3 Node 1: merge Stage 2 + Stage 2.5 artifacts into one deduplicated candidate pool."""

from __future__ import annotations

import hashlib
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from rufp_stage2.schemas import AcceptedSemanticPrompt, ProbePositivePrompt, ProbeResult
from rufp_stage25.schemas import RepairLineageRecord, RepairPromotionRecord, RepairedPrompt

from ..config import effective_artifacts_root
from ..io import load_jsonl, load_jsonl_dicts
from ..policy.schema import Stage3Policy
from ..schemas import Stage3CandidateRecord

logger = logging.getLogger(__name__)


def _root() -> str:
    return effective_artifacts_root()


def _join(base: str, *parts: str) -> str:
    return str(Path(_root()) / base / Path(*parts))


def _canonical_key(original_prompt_id: str, prompt_id: str) -> str:
    return (original_prompt_id or "").strip() or (prompt_id or "").strip()


def _item_id(canonical_key: str) -> str:
    h = hashlib.sha256(f"canon:{canonical_key}".encode("utf-8")).hexdigest()[:16]
    return f"s3-{h}"


def _cap(rows: List[Any], dry: bool, max_per: int) -> List[Any]:
    if not dry:
        return rows
    return rows[:max_per]


def _resolve_promoted_path(policy: Stage3Policy, stage25_dir: str) -> Optional[str]:
    """Prefer repaired_accept_set.jsonl when present, else repair_promoted_set.jsonl."""
    fn = policy.filenames
    for name in (getattr(fn, "repaired_accept", None) or "repaired_accept_set.jsonl", fn.repair_promoted):
        if not name:
            continue
        p = _join(stage25_dir, name)
        if Path(p).is_file():
            return p
    return None


def _resolve_family_map_path(policy: Stage3Policy) -> Optional[str]:
    fn = policy.filenames
    s1 = getattr(policy.inputs, "stage1_dir", None)
    if s1:
        p = _join(s1, fn.family_to_prompt_map)
        if Path(p).is_file():
            return p
    fallback = _join("artifacts/stage1", fn.family_to_prompt_map)
    if Path(fallback).is_file():
        return fallback
    return None


def _load_family_reverse_map(path: str) -> Dict[str, str]:
    """prompt_id -> family_id from Stage 1 family_to_prompt_map.jsonl."""
    out: Dict[str, str] = {}
    for row in load_jsonl_dicts(path):
        fid = str(row.get("family_id", ""))
        for pid in row.get("prompt_ids") or []:
            out[str(pid)] = fid
    return out


def _aggregate_probes_by_prompt(path: str) -> Dict[str, Dict[str, Any]]:
    """Build probe_profile payload keyed by prompt_id."""
    if not Path(path).is_file():
        return {}
    rows = load_jsonl(path, ProbeResult)
    by_pid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for pr in rows:
        by_pid[pr.prompt_id].append(pr.model_dump(mode="json"))
    out: Dict[str, Dict[str, Any]] = {}
    for pid, lst in by_pid.items():
        refusal_ct = sum(
            1
            for x in lst
            if str(x.get("response_label", "")).lower() in ("refusal", "partial_refusal")
        )
        out[pid] = {
            "probe_results": lst,
            "refusal_count": refusal_ct,
            "probe_count": len(lst),
        }
    return out


def _load_repair_lineage_index(path: str) -> Dict[str, RepairLineageRecord]:
    if not Path(path).is_file():
        return {}
    rows = load_jsonl(path, RepairLineageRecord)
    idx: Dict[str, RepairLineageRecord] = {}
    for r in rows:
        idx[r.original_prompt_id] = r
        idx[r.repaired_prompt_id] = r
    return idx


def _merge_probe_profiles(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    if not a:
        return dict(b)
    if not b:
        return dict(a)
    ra = a.get("probe_results") or []
    rb = b.get("probe_results") or []
    merged = list(ra) + list(rb)
    refusal_ct = sum(
        1
        for x in merged
        if str(x.get("response_label", "")).lower() in ("refusal", "partial_refusal")
    )
    return {
        "probe_results": merged,
        "refusal_count": refusal_ct,
        "probe_count": len(merged),
    }


@dataclass
class _Bucket:
    canonical_key: str
    source_stages: Set[str] = field(default_factory=set)
    source_statuses: Set[str] = field(default_factory=set)
    source_labels: Set[str] = field(default_factory=set)
    prompt_id: str = ""
    original_prompt_id: str = ""
    stage1_prompt_id: Optional[str] = None
    prompt_text: str = ""
    family_id: str = "unknown"
    category: str = "unknown"
    subtype: Optional[str] = None
    probe_profile: Dict[str, Any] = field(default_factory=dict)
    lineage_refs: List[str] = field(default_factory=list)
    generation_routes: Set[str] = field(default_factory=set)
    repair_metadata: Optional[Dict[str, Any]] = None
    lineage: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_lineage_ref(self, ref: str) -> None:
        if ref and ref not in self.lineage_refs:
            self.lineage_refs.append(ref)

    def merge_probe_profile(self, other: Dict[str, Any]) -> None:
        if not other:
            return
        self.probe_profile = _merge_probe_profiles(self.probe_profile, other)

    def to_record(self) -> Stage3CandidateRecord:
        stages = sorted(self.source_stages)
        statuses = sorted(self.source_statuses)
        gen_route = " | ".join(sorted(x for x in self.generation_routes if x))
        src = "+".join(sorted(self.source_labels)) if self.source_labels else "merged"
        return Stage3CandidateRecord(
            item_id=_item_id(self.canonical_key),
            source=src,
            prompt_id=self.prompt_id or self.original_prompt_id,
            original_prompt_id=self.original_prompt_id,
            stage1_prompt_id=self.stage1_prompt_id,
            prompt_text=self.prompt_text,
            family_id=self.family_id,
            category=self.category,
            subtype=self.subtype,
            source_stages=stages,
            source_statuses=statuses,
            probe_profile=dict(self.probe_profile),
            lineage_refs=list(self.lineage_refs),
            generation_route=gen_route,
            repair_metadata=self.repair_metadata,
            lineage=dict(self.lineage),
            metadata=dict(self.metadata),
            added_at=datetime.now(),
        )


def _get_bucket(
    buckets: Dict[str, _Bucket],
    canonical_key: str,
    *,
    original_prompt_id: str,
    prompt_id: str,
) -> _Bucket:
    if canonical_key not in buckets:
        buckets[canonical_key] = _Bucket(
            canonical_key=canonical_key,
            original_prompt_id=original_prompt_id or prompt_id,
            prompt_id=prompt_id or original_prompt_id,
        )
    return buckets[canonical_key]


def merge_candidate_pool_with_summary(
    policy: Stage3Policy,
    *,
    dry_run: bool,
) -> Tuple[List[Stage3CandidateRecord], Dict[str, Any]]:
    """
    Build deduplicated candidate rows keyed by Stage 1 / semantic ``original_prompt_id``.

    Returns pool rows and a summary dict (counts by source_stage / source_status).
    """
    fn = policy.filenames
    s2 = policy.inputs.stage2_dir
    s25 = policy.inputs.stage25_dir
    max_rows = policy.dry_run.max_rows_per_source if dry_run else 10**9

    buckets: Dict[str, _Bucket] = {}

    # --- refusal probes (Stage 2) — keyed by prompt_id ---
    probe_path = _join(s2, fn.refusal_probes)
    probes_by_prompt = _aggregate_probes_by_prompt(probe_path)
    if not Path(probe_path).is_file():
        logger.warning("Missing refusal probes: %s", probe_path)

    # --- optional family map (Stage 1) ---
    fam_path = _resolve_family_map_path(policy)
    fam_reverse: Dict[str, str] = {}
    if fam_path:
        fam_reverse = _load_family_reverse_map(fam_path)
        logger.info("Loaded family map: %s (%d prompt ids)", fam_path, len(fam_reverse))
    else:
        logger.info("No family_to_prompt_map found (optional); skipping")

    # --- repair lineage (optional) ---
    lineage_by_key: Dict[str, RepairLineageRecord] = {}
    lineage_path: Optional[str] = None
    if s25:
        lineage_path = _join(s25, fn.repair_lineage)
        if Path(lineage_path).is_file():
            lineage_by_key = _load_repair_lineage_index(lineage_path)
        else:
            logger.info("Optional repair_lineage missing: %s", lineage_path)

    def attach_probes(b: _Bucket, *prompt_ids: str) -> None:
        merged: Dict[str, Any] = {}
        for pid in prompt_ids:
            if not pid:
                continue
            if pid in probes_by_prompt:
                merged = _merge_probe_profiles(merged, probes_by_prompt[pid])
        if merged:
            b.merge_probe_profile(merged)

    def attach_lineage_record(b: _Bucket, key: str) -> None:
        rec = lineage_by_key.get(key)
        if not rec:
            return
        b.stage1_prompt_id = b.stage1_prompt_id or rec.stage1_prompt_id
        b.add_lineage_ref(f"repair_lineage:{rec.repair_id}")
        if rec.generation_route:
            b.generation_routes.add(rec.generation_route)
        b.lineage.setdefault("repair_lineage", rec.model_dump(mode="json"))

    # --- validated semantic ---
    vpath = _join(s2, fn.validated_semantic)
    if Path(vpath).is_file():
        raw = load_jsonl(vpath, AcceptedSemanticPrompt)
        for row in _cap(raw, dry_run, max_rows):
            inp = row.input
            ck = _canonical_key(inp.prompt_id, inp.prompt_id)
            b = _get_bucket(buckets, ck, original_prompt_id=inp.prompt_id, prompt_id=inp.prompt_id)
            b.source_stages.add("stage2")
            b.source_statuses.add("validated")
            b.source_labels.add("validated_semantic")
            b.prompt_id = inp.prompt_id
            b.original_prompt_id = inp.prompt_id
            b.prompt_text = inp.text
            b.family_id = inp.family_id
            b.category = inp.category
            b.subtype = inp.metadata.get("subtype")
            gr = inp.metadata.get("generation_route")
            if isinstance(gr, str) and gr:
                b.generation_routes.add(gr)
            b.add_lineage_ref(f"stage2:validated_semantic_set:{inp.prompt_id}")
            b.lineage.setdefault("stage2_paths", []).append(vpath)
            b.metadata.setdefault("stage2_validated_labels", row.model_dump(mode="json"))
            attach_probes(b, inp.prompt_id)
            if inp.prompt_id in fam_reverse and fam_reverse[inp.prompt_id] == inp.family_id:
                b.add_lineage_ref("stage1:family_to_prompt_map")
    else:
        logger.warning("Missing %s", vpath)

    # --- probe positive ---
    pp_path = _join(s2, fn.probe_positive)
    if Path(pp_path).is_file():
        raw = load_jsonl(pp_path, ProbePositivePrompt)
        for row in _cap(raw, dry_run, max_rows):
            sem = row.semantic_data
            inp = sem.input
            ck = _canonical_key(inp.prompt_id, inp.prompt_id)
            b = _get_bucket(buckets, ck, original_prompt_id=inp.prompt_id, prompt_id=inp.prompt_id)
            b.source_stages.add("stage2")
            b.source_statuses.add("probe_positive")
            b.source_labels.add("probe_positive")
            b.prompt_id = inp.prompt_id
            b.original_prompt_id = inp.prompt_id
            b.prompt_text = inp.text
            b.family_id = inp.family_id
            b.category = inp.category
            b.subtype = inp.metadata.get("subtype")
            gr = inp.metadata.get("generation_route")
            if isinstance(gr, str) and gr:
                b.generation_routes.add(gr)
            b.add_lineage_ref(f"stage2:probe_positive_set:{inp.prompt_id}")
            b.lineage.setdefault("stage2_paths", []).append(pp_path)
            b.metadata.setdefault("probe_positive_row", row.model_dump(mode="json"))
            attach_probes(b, inp.prompt_id)
            # embedded probes from Stage 2 positive set
            b.merge_probe_profile(
                {
                    "semantic_probe_results": [p.model_dump(mode="json") for p in row.probes],
                    "refusal_count_stage2": row.refusal_count,
                }
            )
            if inp.prompt_id in fam_reverse and fam_reverse[inp.prompt_id] == inp.family_id:
                b.add_lineage_ref("stage1:family_to_prompt_map")
    else:
        logger.warning("Missing %s", pp_path)

    # --- Stage 2.5 repaired / promoted ---
    if s25:
        prom_path = _resolve_promoted_path(policy, s25)
        rep_path = _join(s25, fn.repaired_prompts)
        if prom_path and Path(rep_path).is_file():
            promoted = load_jsonl(prom_path, RepairPromotionRecord)
            repaired_rows = load_jsonl(rep_path, RepairedPrompt)
            by_rep = {r.repaired_prompt_id: r for r in repaired_rows}
            for pr in _cap(promoted, dry_run, max_rows):
                rid = pr.repaired_prompt_id
                orig = pr.original_prompt_id
                rr = by_rep.get(rid)
                if not rr:
                    logger.warning("No repaired_prompt row for promoted id=%s", rid)
                    continue
                ck = _canonical_key(orig, orig)
                b = _get_bucket(buckets, ck, original_prompt_id=orig, prompt_id=rid)
                b.source_stages.add("stage25")
                b.source_statuses.add("repaired_accept")
                b.source_labels.add("repaired_accept")
                b.original_prompt_id = orig
                b.prompt_id = rid
                b.stage1_prompt_id = rr.stage1_prompt_id
                b.prompt_text = rr.repaired_text
                b.family_id = pr.family_id
                b.category = pr.category
                b.subtype = rr.subtype
                b.generation_routes.add(rr.generation_route)
                b.add_lineage_ref(f"stage25:repaired_prompt:{rid}")
                b.add_lineage_ref(f"stage25:original:{orig}")
                b.add_lineage_ref(f"stage1:stage1_prompt_id:{rr.stage1_prompt_id}")
                b.lineage.setdefault("stage25_paths", []).extend([prom_path, rep_path])
                promo_dump = pr.model_dump(mode="json")
                b.repair_metadata = {"promotion": promo_dump, "repaired_prompt": rr.model_dump(mode="json")}
                b.metadata.setdefault("stage25_promotion", promo_dump)
                attach_probes(b, orig, rid)
                attach_lineage_record(b, orig)
                attach_lineage_record(b, rid)
        else:
            logger.warning("Stage25 promoted/repaired missing or empty: promoted=%s repaired=%s", prom_path, rep_path)

    # Finalize: ensure probe_profile from refusal file wins merge order for overlapping keys
    out = [buckets[k].to_record() for k in sorted(buckets.keys())]
    summary = _build_summary(out)
    logger.info("candidate_merger: %d unique candidates (from %d buckets)", len(out), len(buckets))
    return out, summary


def _build_summary(rows: List[Stage3CandidateRecord]) -> Dict[str, Any]:
    """Count contributions: each row may add to multiple stage/status tags."""
    by_stage: Dict[str, int] = defaultdict(int)
    by_status: Dict[str, int] = defaultdict(int)
    pairs: Dict[str, int] = defaultdict(int)
    for r in rows:
        for s in r.source_stages:
            by_stage[s] += 1
        for t in r.source_statuses:
            by_status[t] += 1
        for s in r.source_stages:
            for t in r.source_statuses:
                pairs[f"{s}:{t}"] += 1
    return {
        "pool_size": len(rows),
        "count_by_source_stage": dict(sorted(by_stage.items())),
        "count_by_source_status": dict(sorted(by_status.items())),
        "count_by_stage_status_pair": dict(sorted(pairs.items())),
    }


def merge_candidate_pool(policy: Stage3Policy, *, dry_run: bool) -> List[Stage3CandidateRecord]:
    rows, _ = merge_candidate_pool_with_summary(policy, dry_run=dry_run)
    return rows
