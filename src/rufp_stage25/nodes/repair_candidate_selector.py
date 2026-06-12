import json
import os
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, List, Dict, Any, Optional

from ..schemas import RepairCandidate
from src.rufp_stage2.schemas import AcceptedSemanticPrompt, ReviewRecord, RejectRecord, ProbeResult, RefusalSignal

if TYPE_CHECKING:
    from ..policy.schema import Stage25Policy

logger = logging.getLogger(__name__)


def _passes_reason_policy(reason: str, policy: "Stage25Policy") -> bool:
    if reason in policy.repair_reasons.exclude:
        return False
    if policy.repair_reasons.allow:
        return reason in policy.repair_reasons.allow
    return True


def _apply_max_repairs(candidates: List[RepairCandidate], max_per_id: int) -> List[RepairCandidate]:
    """Globally sort by priority, then keep at most ``max_per_id`` rows per ``original_prompt_id``."""
    counts: Dict[str, int] = defaultdict(int)
    out: List[RepairCandidate] = []
    for c in sorted(candidates, key=lambda x: (-x.priority, x.original_prompt_id)):
        if counts[c.original_prompt_id] >= max_per_id:
            continue
        counts[c.original_prompt_id] += 1
        out.append(c)
    return out

class RepairCandidateSelectorNode:
    def __init__(self, stage2_run_id: str):
        self.stage2_run_id = stage2_run_id
        root = os.environ.get("RUFP_ARTIFACTS_ROOT", ".")
        self.base_dir = os.path.join(root, "artifacts", "stage2", stage2_run_id)

    def _load_jsonl(self, filename: str) -> List[Dict[str, Any]]:
        path = os.path.join(self.base_dir, filename)
        if not os.path.exists(path):
            logger.warning(f"File {path} not found.")
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def select_candidates(self, policy: Optional["Stage25Policy"] = None) -> List[RepairCandidate]:
        from ..policy.schema import Stage25Policy

        pol = policy or Stage25Policy()
        candidates: List[RepairCandidate] = []
        
        # 1. Load data from Stage 2
        validated = self._load_jsonl("validated_semantic_set.jsonl")
        reviews = self._load_jsonl("review_queue.jsonl")
        rejects = self._load_jsonl("reject_set.jsonl")
        probes = {}
        for p in self._load_jsonl("refusal_probe_results.jsonl"):
            pid = p["prompt_id"]
            if pid not in probes: probes[pid] = []
            probes[pid].append(p)

        # 2. Process Validated Semantic (Look for "no_probe_signal" or "weak_borderline")
        for item in validated:
            p = AcceptedSemanticPrompt(**item)
            pid = p.input.prompt_id
            
            # Check for no probe signal
            has_refusal = any(pr["response_label"] == RefusalSignal.REFUSAL for pr in probes.get(pid, []))
            
            repair_reason = None
            priority = 1
            
            if p.borderline.borderline_label == "weak":
                repair_reason = "natural_but_weak_borderline"
                priority = 3
            elif not has_refusal:
                repair_reason = "good_semantics_but_no_probe_signal"
                priority = 2

            if repair_reason:
                candidates.append(RepairCandidate(
                    original_prompt_id=pid,
                    family_id=p.input.family_id,
                    category=p.input.category,
                    subtype=p.input.metadata.get("subtype"),
                    generation_route=p.input.metadata.get("route", "unknown"),
                    source_bucket="validated_semantic_set",
                    repair_reason=repair_reason,
                    original_prompt_text=p.input.text,
                    stage2_labels_snapshot={
                        "safety": p.safety.model_dump(),
                        "naturalness": p.naturalness.model_dump(),
                        "borderline": p.borderline.model_dump()
                    },
                    priority=priority,
                    metadata={
                        "stage2_run_id": self.stage2_run_id,
                        "stage1_prompt_id": pid,
                    }
                ))

        # 3. Process Review Queue (Look for "too_bland", "translatedese", etc.)
        for item in reviews:
            r = ReviewRecord(**item)
            # Review reasons are semicolon-separated
            reasons = r.reason.split(";")
            
            repair_reason = None
            priority = 1
            
            if "unclear_safety" in reasons:
                # We don't repair unclear safety usually, but if it's combined with something else...
                # For now, skip pure unclear safety
                continue
            
            if "clunky_russian" in reasons:
                repair_reason = "translatedese"
                priority = 4
            elif "weak_borderline" in reasons:
                repair_reason = "too_bland"
                priority = 3
            elif "partial_refusal_with_weak_borderline" in reasons:
                repair_reason = "partial_refusal_but_repairable"
                priority = 5

            if repair_reason:
                # We need to find the original prompt data (category, etc.)
                # In a real system, we'd have a database or a full index. 
                # Here we assume we can extract it from the 'data' field of ReviewRecord
                data = r.data
                candidates.append(RepairCandidate(
                    original_prompt_id=r.prompt_id,
                    family_id=data.get("family_id", "unknown"), # Fallback if missing
                    category=data.get("category", "unknown"),
                    subtype=data.get("subtype"),
                    generation_route=data.get("route", "unknown"),
                    source_bucket="review_queue",
                    repair_reason=repair_reason,
                    original_prompt_text=data.get("text", "unknown"),
                    stage2_labels_snapshot=data,
                    priority=priority,
                    metadata={
                        "stage2_run_id": self.stage2_run_id,
                        "stage1_prompt_id": r.prompt_id,
                    }
                ))

        # 4. Process Reject Set (Only "too_obvious_safe")
        for item in rejects:
            rej = RejectRecord(**item)
            if rej.reason == "failed_hard_criteria":
                # For testing purposes, let's be more lenient with what we try to repair
                # In a real run, we'd follow the strict rules
                if rej.data.get("safety") == "safe":
                    candidates.append(RepairCandidate(
                        original_prompt_id=rej.prompt_id,
                        family_id=rej.data.get("family_id", "unknown"),
                        category=rej.data.get("category", "unknown"),
                        subtype=rej.data.get("subtype"),
                        generation_route=rej.data.get("route", "unknown"),
                        source_bucket="reject_set",
                        repair_reason="too_obvious_safe_or_broken_naturalness",
                        original_prompt_text=rej.data.get("text", "unknown"),
                        stage2_labels_snapshot=rej.data,
                        priority=2,
                        metadata={
                            "stage2_run_id": self.stage2_run_id,
                            "stage1_prompt_id": rej.prompt_id,
                        }
                    ))

        filtered: List[RepairCandidate] = []
        for c in candidates:
            if _passes_reason_policy(c.repair_reason, pol):
                filtered.append(c)
            else:
                logger.debug(
                    "Skipping candidate %s repair_reason=%s (policy filter)",
                    c.original_prompt_id,
                    c.repair_reason,
                )

        capped = _apply_max_repairs(filtered, pol.limits.max_repairs_per_prompt)
        if len(capped) < len(filtered):
            logger.info(
                "Applied max_repairs_per_prompt=%s: %d -> %d rows",
                pol.limits.max_repairs_per_prompt,
                len(filtered),
                len(capped),
            )
        return capped
