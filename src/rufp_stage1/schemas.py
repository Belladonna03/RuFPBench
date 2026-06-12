from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class GenerationRoute(str, Enum):
    DIRECT_EXPANSION = "direct_expansion"
    ANALYSIS_FRAMING = "analysis_framing"
    HISTORICAL_EDUCATIONAL_FRAMING = "historical_educational_framing"
    FICTIONAL_GAME_FRAMING = "fictional_game_framing"
    ADVERSARIAL_BENIGN_WRAPPING = "adversarial_benign_wrapping"
    SURFACE_VARIATION = "surface_variation"

class PromptLength(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

class BaseItemRecord(BaseModel):
    base_item_id: str
    category: str
    subtype: Optional[str] = None
    family_id: str
    family_name: str
    canonical_form: str
    base_item_text: str
    source_url: Optional[str] = None
    source_title: Optional[str] = None
    notes: Optional[str] = None

class FamilyBatch(BaseModel):
    batch_id: str
    family_id: str
    family_name: str
    items: List[BaseItemRecord]

class FamilyRouteDecision(BaseModel):
    family_id: str
    primary_route: GenerationRoute
    secondary_route: Optional[GenerationRoute] = None
    safe_core_meaning: str
    must_keep: List[str] = Field(default_factory=list)
    must_avoid: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)

class Candidate(BaseModel):
    candidate_id: str
    family_id: str
    category: str
    subtype: Optional[str] = None
    length_bucket: PromptLength
    route: GenerationRoute
    prompt_text: str
    rationale: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class NaturalizedCandidate(BaseModel):
    candidate_id: str
    original_text: str
    rewritten_text: str
    change_type: str
    rewrite_changed: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RefinedCandidate(BaseModel):
    candidate_id: str
    refined_text: str
    borderline_strategy: str
    refinement_note: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class FilterDecision(BaseModel):
    is_accepted: bool
    reason: Optional[str] = None

class AcceptedPrompt(BaseModel):
    prompt_id: str
    candidate_id: str
    family_id: str
    category: str
    subtype: Optional[str] = None
    length_bucket: PromptLength
    route: GenerationRoute
    text: str
    rationale: str
    refinement_note: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    accepted_at: datetime = Field(default_factory=datetime.now)

class RejectedPrompt(BaseModel):
    candidate_id: str
    family_id: str
    text: str
    reason: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    rejected_at: datetime = Field(default_factory=datetime.now)

class RunManifest(BaseModel):
    run_id: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    total_items: int
    total_candidates: int
    total_accepted: int
    total_rejected: int
    config_snapshot: Dict[str, Any]
