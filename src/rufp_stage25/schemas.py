from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

from rufp_stage2.schemas import (
    SafetyJudgment,
    NaturalnessJudgment,
    BorderlineJudgment,
    ProbeResult,
    SafetyLabel,
    NaturalnessLabel,
    BorderlineLabel,
)

class RepairSource(str, Enum):
    VALIDATED_SEMANTIC = "validated_semantic"
    REVIEW_QUEUE = "review_queue"
    PROBE_NEGATIVE = "probe_negative"

class RepairCandidate(BaseModel):
    original_prompt_id: str
    family_id: str
    category: str
    subtype: Optional[str] = None
    generation_route: str
    source_bucket: str # review_queue, reject_set, validated_semantic_set
    repair_reason: str
    original_prompt_text: str
    stage2_labels_snapshot: Dict[str, Any]
    priority: int = Field(default=1, ge=1, le=5)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RepairStrategy(str, Enum):
    STRENGTHEN_BORDERLINE_SURFACE = "strengthen_borderline_surface"
    REMOVE_OVERT_SAFETY_MARKER = "remove_overt_safety_marker"
    NATURALIZE_RUSSIAN = "naturalize_russian"
    REDUCE_STERILITY = "reduce_sterility"
    ADD_REALISTIC_CONTEXT = "add_realistic_context"
    SHARPEN_CATEGORY_SIGNAL = "sharpen_category_signal"
    REBALANCE_PROMPT_LENGTH = "rebalance_prompt_length"

class RepairPlan(BaseModel):
    original_prompt_id: str
    repair_strategy: RepairStrategy
    repair_goals: List[str] = Field(default_factory=list)
    must_preserve: List[str] = Field(default_factory=list)
    must_avoid: List[str] = Field(default_factory=list)
    expected_risk: str
    reasoning: str
    instructions: str
    planned_at: datetime = Field(default_factory=datetime.now)

class FinalRepairDecision(str, Enum):
    PENDING_REVALIDATION = "pending_revalidation"
    PROMOTED = "promoted"
    FAILED = "failed"
    NO_IMPROVEMENT = "no_improvement"
    REVIEW = "review"


class RepairedPrompt(BaseModel):
    """Repaired prompt row in repaired_prompts.jsonl (lineage fields + text)."""

    repair_id: str
    repaired_prompt_id: str
    original_prompt_id: str
    parent_stage2_run_id: str
    parent_stage25_run_id: str
    stage1_prompt_id: str
    family_id: str
    category: str
    subtype: Optional[str] = None
    generation_route: str
    repair_reason: str
    repair_strategy: RepairStrategy
    rewrite_changed: bool
    original_hash: str
    repaired_hash: str
    original_prompt_text: str
    repaired_text: str
    change_summary: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    repaired_at: datetime = Field(default_factory=datetime.now)


class RepairLineageRecord(BaseModel):
    """
    Machine-readable full audit row for repair_lineage.jsonl.
    Links Stage 1 → Stage 2 labels → repair plan → revalidation → decision.
    """

    repair_id: str
    original_prompt_id: str
    repaired_prompt_id: str
    stage1_prompt_id: str
    parent_stage2_run_id: str
    parent_stage25_run_id: str
    repair_reason: str
    repair_strategy: RepairStrategy
    repair_changed: bool
    original_hash: str
    repaired_hash: str
    stage2_labels_snapshot: Dict[str, Any] = Field(default_factory=dict)
    repair_plan_snapshot: Dict[str, Any] = Field(default_factory=dict)
    revalidation_snapshot: Dict[str, Any] = Field(default_factory=dict)
    final_repair_decision: str
    family_id: str
    category: str
    subtype: Optional[str] = None
    generation_route: str
    original_prompt_text: str
    repaired_text: str
    change_summary: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

class RepairRevalidationResult(BaseModel):
    """
    One row in repair_revalidation_results.jsonl: Stage 2 judges + probes on repaired text.
    Labels duplicate enums from judgments for quick filtering.
    """

    repaired_prompt_id: str
    original_prompt_id: str
    safety_label: SafetyLabel
    naturalness_label: NaturalnessLabel
    borderline_label: BorderlineLabel
    safety: SafetyJudgment
    naturalness: NaturalnessJudgment
    borderline: BorderlineJudgment
    probe_results: List[ProbeResult] = Field(default_factory=list)
    timing_ms_total: float
    timing_ms_breakdown: Dict[str, float] = Field(default_factory=dict)
    revalidated_at: datetime = Field(default_factory=datetime.now)

class RepairPromotionRecord(BaseModel):
    repaired_prompt_id: str
    original_prompt_id: str
    family_id: str
    category: str
    repair_strategy: RepairStrategy
    target_set: str = "repair_promoted_set"
    promotion_reason: str
    revalidation_summary: Dict[str, Any] = Field(default_factory=dict)
    before_labels: Dict[str, Any] = Field(default_factory=dict)
    after_labels: Dict[str, Any] = Field(default_factory=dict)
    label_improvement_score: float = 0.0
    probe_improved: bool = False
    quality_improved: bool = False
    promoted_at: datetime = Field(default_factory=datetime.now)


class RepairFailureRecord(BaseModel):
    repaired_prompt_id: str
    original_prompt_id: str
    family_id: str
    category: str
    repair_strategy: RepairStrategy
    failure_reason: str
    revalidation_summary: Dict[str, Any] = Field(default_factory=dict)
    before_labels: Dict[str, Any] = Field(default_factory=dict)
    after_labels: Dict[str, Any] = Field(default_factory=dict)
    data: Dict[str, Any] = Field(default_factory=dict)
    failed_at: datetime = Field(default_factory=datetime.now)


class RepairReviewRecord(BaseModel):
    repaired_prompt_id: str
    original_prompt_id: str
    family_id: str
    category: str
    repair_strategy: RepairStrategy
    review_reason: str
    revalidation_summary: Dict[str, Any] = Field(default_factory=dict)
    before_labels: Dict[str, Any] = Field(default_factory=dict)
    after_labels: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class Stage25DecisionSummary(BaseModel):
    """Written to stage25_summary.json by Repair Decision Aggregator (Node 5)."""

    run_id: str
    repair_candidates_sent: int
    repaired_prompts_count: int
    revalidation_count: int
    promoted: int
    failed: int
    review: int
    promotion_rate_by_category: Dict[str, float] = Field(default_factory=dict)
    promotion_rate_by_repair_strategy: Dict[str, float] = Field(default_factory=dict)
    average_label_improvement: float = 0.0
    label_improvements_sample: List[float] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)

class Stage25RunManifest(BaseModel):
    run_id: str
    input_run_id: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    config_snapshot: Dict[str, Any]
    counts: Dict[str, int] = Field(default_factory=dict)
    node_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    lineage_finalized: bool = False
    analysis_artifacts: Dict[str, str] = Field(default_factory=dict)
