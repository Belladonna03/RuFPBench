"""Pydantic models for extraction outputs and document pseudo-graphs."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class RawEntity(BaseModel):
    """Single GLiNER span."""

    text: str
    label: str
    score: float = Field(ge=0.0)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    extraction_source: Literal["gliner"] = "gliner"

    @model_validator(mode="after")
    def span_order(self) -> RawEntity:
        if self.end < self.start:
            raise ValueError("end must be >= start")
        return self


class RawKeyphrase(BaseModel):
    """Single KeyBERT phrase."""

    text: str
    score: float = Field(ge=0.0)
    label: Literal["keyphrase"] = "keyphrase"
    extraction_source: Literal["keybert"] = "keybert"
    start: Optional[int] = None
    end: Optional[int] = None

    @model_validator(mode="after")
    def span_order(self) -> "RawKeyphrase":
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("end must be >= start when both offsets are set")
        return self


class RawMention(BaseModel):
    """One surface mention contributing to a merged node (traceability)."""

    text: str
    label: str
    extraction_source: str
    score: float = Field(ge=0.0)
    start: Optional[int] = None
    end: Optional[int] = None

    @model_validator(mode="after")
    def span_order(self) -> "RawMention":
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("end must be >= start when both offsets are set")
        return self


class MergedNode(BaseModel):
    """Deduplicated node with preserved raw mentions."""

    node_id: str
    canonical_text: str
    normalized_text: str
    raw_mentions: list[RawMention] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    max_score: float = Field(default=0.0, ge=0.0)
    mention_count: int = Field(default=0, ge=0)


EdgeType = Literal["same_prompt", "same_sentence", "nearby_window"]


class EdgeEvidence(BaseModel):
    """Why this edge exists: sentence alignment, char gap, short label."""

    sentence_id: Optional[int] = None
    sentence_ids: list[int] = Field(
        default_factory=list,
        description="All shared sentence indices for same_sentence (sorted, unique).",
    )
    char_distance: Optional[int] = None
    note: Optional[str] = None

    @model_validator(mode="after")
    def sync_sentence_id_from_list(self) -> "EdgeEvidence":
        if self.sentence_ids and self.sentence_id is None:
            return self.model_copy(update={"sentence_id": min(self.sentence_ids)})
        return self


class Edge(BaseModel):
    """Co-occurrence or proximity link between merged nodes."""

    source_node_id: str
    target_node_id: str
    edge_type: EdgeType
    weight: float = 1.0
    evidence: EdgeEvidence = Field(default_factory=EdgeEvidence)

    model_config = {"extra": "forbid"}


class DocumentMetadata(BaseModel):
    """Per-document stats and optional source fields."""

    source: Optional[str] = None
    category: Optional[str] = None
    num_entities: int = 0
    num_keyphrases: int = 0
    num_nodes: int = 0
    num_edges: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("num_entities", "num_keyphrases", "num_nodes", "num_edges")
    @classmethod
    def counts_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("count fields must be >= 0")
        return v


class DocumentGraph(BaseModel):
    """One input row → full pseudo-graph record for JSONL output."""

    id: str
    original_text: str
    normalized_text: str
    entities: list[RawEntity] = Field(default_factory=list)
    keyphrases: list[RawKeyphrase] = Field(default_factory=list)
    merged_nodes: list[MergedNode] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
