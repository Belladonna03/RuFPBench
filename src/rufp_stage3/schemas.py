"""Stage 3 Pydantic v2 models — JSONL-serializable rows."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class Stage3CandidateRecord(BaseModel):
    """Unified candidate after merging Stage 2 + Stage 2.5 sources (candidate merger node)."""

    model_config = ConfigDict(populate_by_name=True)

    item_id: str
    source: str
    prompt_id: str
    original_prompt_id: str
    stage1_prompt_id: Optional[str] = None
    prompt_text: str = Field(..., validation_alias=AliasChoices("prompt_text", "text"))
    family_id: str
    category: str
    subtype: Optional[str] = None

    source_stages: List[str] = Field(default_factory=list)
    source_statuses: List[str] = Field(default_factory=list)

    probe_profile: Dict[str, Any] = Field(default_factory=dict)
    lineage_refs: List[str] = Field(default_factory=list)
    generation_route: str = ""
    repair_metadata: Optional[Dict[str, Any]] = None

    lineage: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    added_at: datetime = Field(default_factory=datetime.now)

    @property
    def text(self) -> str:
        """Backward compatibility for code paths that read ``row.text``."""
        return self.prompt_text

    @model_validator(mode="after")
    def _retrofill_stage_lists(self) -> Stage3CandidateRecord:
        """Allow loading legacy pool rows that predate source_stages / source_statuses."""
        if not self.source_stages:
            if self.repair_metadata is not None or "stage25" in self.source:
                self.source_stages = ["stage25"]
            else:
                self.source_stages = ["stage2"]
        if not self.source_statuses:
            src = self.source.lower()
            if "repaired" in src or "promoted" in src or self.repair_metadata is not None:
                self.source_statuses = ["repaired_accept"]
            elif "probe" in src:
                self.source_statuses = ["probe_positive"]
            else:
                self.source_statuses = ["validated"]
        return self


class QCLabelRecord(BaseModel):
    """Per-candidate QC outcome (QC + dedup + clustering node)."""

    model_config = ConfigDict(populate_by_name=True)

    item_id: str
    qc_label: str = Field(
        "keep",
        description="keep | exact_duplicate | near_duplicate | noisy | low_value | cluster_representative",
    )
    qc_pass: bool = True
    qc_flags: List[str] = Field(default_factory=list)
    notes: str = ""
    original_prompt_id: Optional[str] = None
    cluster_id: Optional[str] = None
    labeled_at: datetime = Field(default_factory=datetime.now)

    @model_validator(mode="before")
    @classmethod
    def _legacy_qc_label(cls, data: Any) -> Any:
        if isinstance(data, dict) and "qc_label" not in data:
            d = dict(data)
            d["qc_label"] = "keep" if d.get("qc_pass") else "noisy"
            return d
        return data

    @model_validator(mode="after")
    def _sync_qc_pass(self) -> QCLabelRecord:
        survivors = {"keep", "cluster_representative"}
        if self.qc_label in survivors:
            self.qc_pass = True
        elif self.qc_label in ("exact_duplicate", "near_duplicate", "noisy", "low_value"):
            self.qc_pass = False
        return self


class DedupClusterRecord(BaseModel):
    """Exact / near-duplicate cluster membership (dedup + clustering node)."""

    cluster_id: str
    representative_item_id: str
    member_item_ids: List[str]
    method: str = "text_hash"
    cluster_kind: str = Field("exact", description="exact | near")
    category: str = ""
    subtype: Optional[str] = None
    similarity_threshold: Optional[float] = None
    config_fingerprint: str = ""
    clustering_seed: int = 0
    created_at: datetime = Field(default_factory=datetime.now)


class BalancedRecord(BaseModel):
    """Stratum / weight after balancing (balance node)."""

    item_id: str
    category: str
    stratum: str = "default"
    weight: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HardSubsetRecord(BaseModel):
    """Hard subset row (Node 4): score, rationale, probe stats, diversity tags."""

    model_config = ConfigDict(populate_by_name=True)

    item_id: str
    hard_score: float = Field(validation_alias=AliasChoices("hard_score", "hardness_score"))
    why_hard: str = ""
    supporting_probe_stats: Dict[str, Any] = Field(default_factory=dict)
    diversity_tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _legacy_hard_subset(cls, data: Any) -> Any:
        if isinstance(data, dict):
            d = dict(data)
            if "hard_score" not in d and "hardness_score" in d:
                d["hard_score"] = d["hardness_score"]
            if "why_hard" not in d and d.get("rationale"):
                d["why_hard"] = d["rationale"]
            return d
        return data

    @property
    def hardness_score(self) -> float:
        return self.hard_score

    @property
    def rationale(self) -> str:
        return self.why_hard


class SplitAssignmentRecord(BaseModel):
    """dev / test / review_holdout assignment (split builder + leakage constraints)."""

    model_config = ConfigDict(populate_by_name=True)

    item_id: str
    split: str
    fold_seed: int = 0
    family_id: Optional[str] = None
    cluster_id: Optional[str] = None
    constraint_group_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FinalExportRecord(BaseModel):
    """One exported benchmark row / file pointer (exporters)."""

    item_id: str
    split: str
    export_relpath: str
    sha256_text: Optional[str] = None
    format: str = "jsonl"
    exported_at: datetime = Field(default_factory=datetime.now)


class Stage3Summary(BaseModel):
    """Aggregated counts for ``stage3_summary.json``."""

    run_id: str
    pool_size: int
    qc_passed: int
    dedup_clusters: int
    balanced_size: int
    hard_subset_size: int
    dev_size: int
    test_size: int
    review_holdout_size: int
    exports_dir: str
    generated_at: datetime = Field(default_factory=datetime.now)


class Stage3RunManifest(BaseModel):
    """Run metadata written once per Stage 3 pipeline execution."""

    run_id: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    config_snapshot: Dict[str, Any] = Field(default_factory=dict)
    input_paths: Dict[str, str] = Field(default_factory=dict)
    output_paths: Dict[str, str] = Field(default_factory=dict)
    node_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    dry_run: bool = False


class AuditPackRecord(BaseModel):
    """One row in ``audit_pack_{random,hard,repaired}.jsonl`` for human spot-check / expert review."""

    model_config = ConfigDict(populate_by_name=True)

    item_id: str
    prompt_text: str
    category: str
    subtype: Optional[str] = None
    source_stage: str = Field(
        "",
        description="Human-readable join of source_stages (e.g. stage2|stage25)",
    )
    source_stages: List[str] = Field(default_factory=list)
    probe_profile: Dict[str, Any] = Field(default_factory=dict)
    repair_flag: bool = False
    qc_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="QC label, flags, cluster_id from Stage 3 Node 2",
    )


class FinalLineageRecord(BaseModel):
    """One row of ``final_lineage.jsonl`` — machine-readable cross-stage trace for release."""

    model_config = ConfigDict(populate_by_name=True)

    schema_version: Literal[1] = 1
    stage3_run_id: str
    item_id: str
    stage1_prompt: Dict[str, Any] = Field(
        default_factory=dict,
        description="Stage 1 identity: prompt ids, family_id",
    )
    stage2: Dict[str, Any] = Field(
        default_factory=dict,
        description="Condensed Stage 2 labels / probe row refs (from merger metadata)",
    )
    stage25_repair: Optional[Dict[str, Any]] = Field(
        None,
        description="Present when Stage 2.5 repair path contributed; else null",
    )
    stage3_qc: Dict[str, Any] = Field(default_factory=dict, description="QC + dedup outcome for this item_id")
    final_split_assignment: Dict[str, Any] = Field(
        default_factory=dict,
        description="dev | test | review_holdout + leakage fields",
    )
    in_hard_subset: bool = False
    lineage_refs: List[str] = Field(default_factory=list)
