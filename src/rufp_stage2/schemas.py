from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class SafetyLabel(str, Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    UNCERTAIN = "uncertain"

class NaturalnessLabel(str, Enum):
    NATURAL = "natural"
    CLUNKY = "clunky"
    BROKEN = "broken"

class BorderlineLabel(str, Enum):
    STRONG = "strong"
    WEAK = "weak"
    NOT_BORDERLINE = "not_borderline"

class RefusalSignal(str, Enum):
    REFUSAL = "refusal"
    COMPLIANCE = "compliance"
    PARTIAL_REFUSAL = "partial_refusal"
    UNCERTAIN = "uncertain"

class Stage2InputPrompt(BaseModel):
    prompt_id: str
    family_id: str
    category: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SafetyJudgment(BaseModel):
    prompt_id: str
    safety_label: SafetyLabel
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    failure_modes: List[str] = Field(default_factory=list)
    model_name: str
    timestamp: datetime = Field(default_factory=datetime.now)

class NaturalnessJudgment(BaseModel):
    prompt_id: str
    naturalness_label: NaturalnessLabel
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    issue_tags: List[str] = Field(default_factory=list)
    model_name: str
    timestamp: datetime = Field(default_factory=datetime.now)

class BorderlineJudgment(BaseModel):
    prompt_id: str
    borderline_label: BorderlineLabel
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    issue_tags: List[str] = Field(default_factory=list)
    model_name: str
    timestamp: datetime = Field(default_factory=datetime.now)

class ProbeResult(BaseModel):
    prompt_id: str
    model_name: str
    raw_response_text: str
    response_label: RefusalSignal
    latency_ms: float
    provider_status: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

class AggregationDecision(BaseModel):
    prompt_id: str
    final_set: str # e.g., "validated_semantic", "probe_positive", "review", "reject"
    reasoning: str

class AcceptedSemanticPrompt(BaseModel):
    input: Stage2InputPrompt
    safety: SafetyJudgment
    naturalness: NaturalnessJudgment
    borderline: BorderlineJudgment
    accepted_at: datetime = Field(default_factory=datetime.now)

class ProbePositivePrompt(BaseModel):
    semantic_data: AcceptedSemanticPrompt
    probes: List[ProbeResult]
    refusal_count: int

class ReviewRecord(BaseModel):
    prompt_id: str
    reason: str
    data: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.now)

class RejectRecord(BaseModel):
    prompt_id: str
    reason: str
    data: Dict[str, Any]
    rejected_at: datetime = Field(default_factory=datetime.now)

class Stage2RunManifest(BaseModel):
    run_id: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    input_path: str
    config_snapshot: Dict[str, Any]
    counts: Dict[str, int] = Field(default_factory=dict)
