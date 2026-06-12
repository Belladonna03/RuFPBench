"""Pydantic schema for ``configs/stage25.yaml`` (Stage 2.5 policy layer)."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from ..schemas import RepairStrategy


class LoggingVerbosity(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class RepairReasonPolicy(BaseModel):
    """If ``allow`` is non-empty, only these ``repair_reason`` values pass. ``exclude`` is always applied."""

    allow: List[str] = Field(
        default_factory=list,
        description="Empty list = allow any reason produced by heuristics (before exclude).",
    )
    exclude: List[str] = Field(
        default_factory=list,
        description="Always drop candidates with these repair_reason values.",
    )


class RepairStrategyPolicy(BaseModel):
    allowed: List[str] = Field(
        default_factory=lambda: [e.value for e in RepairStrategy],
        min_length=1,
        description="Planner may only emit these RepairStrategy enum values (default: all).",
    )

    @model_validator(mode="after")
    def _known_strategies(self) -> RepairStrategyPolicy:
        known = {e.value for e in RepairStrategy}
        bad = [s for s in self.allowed if s not in known]
        if bad:
            raise ValueError(
                f"repair_strategies.allowed contains unknown values: {bad}. "
                f"Valid: {sorted(known)}"
            )
        return self

    def as_set(self) -> set[str]:
        return set(self.allowed)


class LimitsPolicy(BaseModel):
    max_repairs_per_prompt: int = Field(
        1,
        ge=1,
        le=1000,
        description="Max repair candidate rows per original_prompt_id (after sorting by priority).",
    )


class RevalidationPolicy(BaseModel):
    run_probes: bool = Field(True, description="If false, judges still run; probe list is empty.")
    model_names: List[str] = Field(
        default_factory=lambda: ["probe_a", "probe_b"],
        description="Probe model ids when run_probes is true.",
    )


class ManualReviewPolicy(BaseModel):
    """Thresholds / flags for routing to manual review (Node 5)."""

    min_safety_confidence: float = Field(
        0.35,
        ge=0.0,
        le=1.0,
        description="If safety confidence is below this AND naturalness below min_naturalness_confidence → review.",
    )
    min_naturalness_confidence: float = Field(
        0.35,
        ge=0.0,
        le=1.0,
        description="Paired with min_safety_confidence for low_confidence_mixed_signals.",
    )
    review_on_probe_disagreement: bool = Field(
        True,
        description="If true, refusal+compliance across probes → review.",
    )
    review_label_score_delta_below: Optional[float] = Field(
        None,
        description="If set, label-score delta below this → review (instead of fail when degraded).",
    )
    review_on_negative_label_delta: bool = Field(
        False,
        description="If true, any negative label-score delta → review before fail.",
    )


class AggregatorPolicy(BaseModel):
    probe_required_for_promotion: bool = Field(
        False,
        description="If true, promotion requires at least one probe result with refusal signal.",
    )
    manual_review: ManualReviewPolicy = Field(default_factory=ManualReviewPolicy)


class ResumePolicy(BaseModel):
    skip_completed_nodes: bool = Field(
        True,
        description="When CLI resume is on, skip nodes whose outputs already exist.",
    )
    fail_on_empty_upstream: bool = Field(
        False,
        description="If true, abort pipeline when repair_candidates.jsonl is empty after selector.",
    )


class RuntimePolicy(BaseModel):
    use_mock_clients: bool = Field(
        True,
        description="Default mock vs real LLM/judges; CLI may override.",
    )
    logging_verbosity: LoggingVerbosity = Field(
        LoggingVerbosity.INFO,
        description="Root logger level for Stage 2.5 pipeline.",
    )


class Stage25Policy(BaseModel):
    """Root policy object validated from YAML."""

    version: int = Field(1, ge=1)
    repair_reasons: RepairReasonPolicy = Field(default_factory=RepairReasonPolicy)
    repair_strategies: RepairStrategyPolicy = Field(default_factory=RepairStrategyPolicy)
    limits: LimitsPolicy = Field(default_factory=LimitsPolicy)
    revalidation: RevalidationPolicy = Field(default_factory=RevalidationPolicy)
    aggregator: AggregatorPolicy = Field(default_factory=AggregatorPolicy)
    resume: ResumePolicy = Field(default_factory=ResumePolicy)
    runtime: RuntimePolicy = Field(default_factory=RuntimePolicy)

    @model_validator(mode="after")
    def _probe_promotion_consistent(self) -> Stage25Policy:
        if self.aggregator.probe_required_for_promotion and not self.revalidation.run_probes:
            raise ValueError(
                "Invalid policy: aggregator.probe_required_for_promotion=true requires "
                "revalidation.run_probes=true (otherwise no probe signal exists)."
            )
        return self

    def model_dump_for_manifest(self) -> dict:
        return self.model_dump(mode="json")
