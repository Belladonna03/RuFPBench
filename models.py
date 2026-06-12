from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

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

class BaseItem(BaseModel):
    id: str
    text: str
    family: str
    category: str
    subtype: Optional[str] = None

class FamilyConfig(BaseModel):
    family: str
    route: GenerationRoute
    description: Optional[str] = None

class PromptCandidate(BaseModel):
    id: str
    base_item_id: str
    text: str
    length: PromptLength
    route: GenerationRoute
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Stage1Result(BaseModel):
    candidates: List[PromptCandidate]
    rejects: List[Dict[str, Any]] = Field(default_factory=list)
