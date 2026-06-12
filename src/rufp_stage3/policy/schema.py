"""Pydantic config for Stage 3 (``configs/stage3.yaml``) — shaping knobs without code edits."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class InputPaths(BaseModel):
    """Relative to repo root or absolute; Stage 2 / Stage 2.5 artifact directories."""

    stage2_dir: str = Field(..., description="e.g. artifacts/stage2/<run_id>")
    stage25_dir: Optional[str] = Field(None, description="Optional Stage 2.5 run dir for promoted repairs")
    stage1_dir: Optional[str] = Field(
        None,
        description="Optional Stage 1 dir for family_to_prompt_map.jsonl (traceability)",
    )


class FilenameMap(BaseModel):
    """Filenames inside stage2_dir / stage25_dir."""

    validated_semantic: str = "validated_semantic_set.jsonl"
    probe_positive: str = "probe_positive_set.jsonl"
    review_queue: str = "review_queue.jsonl"
    refusal_probes: str = "refusal_probe_results.jsonl"
    family_to_prompt_map: str = "family_to_prompt_map.jsonl"
    stage2_summary: str = "stage2_summary.json"
    repaired_accept: str = "repaired_accept_set.jsonl"
    repair_promoted: str = "repair_promoted_set.jsonl"
    repair_failed: str = "repair_failed_set.jsonl"
    repaired_prompts: str = "repaired_prompts.jsonl"
    repair_lineage: str = "repair_lineage.jsonl"
    stage25_summary: str = "stage25_summary.json"


class DedupConfig(BaseModel):
    method: str = Field("sha256_text", description="sha256_text | none | skip")
    normalize_whitespace: bool = True


class QcDedupClusterConfig(BaseModel):
    """Flattened QC + near-dedup + clustering (Node 2). Populated from YAML ``qc_dedup`` or nested ``qc_thresholds`` / ``near_duplicate`` / ``clustering``."""

    min_text_len: int = Field(8, ge=0, description="Below this length → low_value (after strip)")
    max_repeat_char_run: int = Field(40, ge=1, description="Long unbroken repeats → noisy")
    near_duplicate_threshold: float = Field(0.88, ge=0.0, le=1.0)
    near_duplicate_metric: Literal["difflib_ratio", "jaccard_words"] = Field(
        "difflib_ratio",
        description="Similarity for near-duplicate edges",
    )
    near_normalization: Literal["whitespace_lower", "alnum_lower"] = Field(
        "whitespace_lower",
        description="Normalization before near-duplicate comparison",
    )
    clustering_seed: int = Field(42, description="Logged for reproducibility")
    injection_patterns: List[str] = Field(
        default_factory=lambda: [r"(?i)\bignore previous\b"],
        description="Regexes; match → noisy / injection risk",
    )
    dedup_strata: List[Literal["category", "subtype"]] = Field(
        default_factory=lambda: ["category", "subtype"],
        description="Independent buckets for exact+near dedup (order: category then subtype)",
    )


class BalanceConfig(BaseModel):
    max_per_category: Optional[int] = Field(None, description="Cap rows per category; None = no cap")
    min_per_category: int = Field(0, ge=0)
    max_rows_per_family_per_category: Optional[int] = Field(
        None,
        ge=1,
        description="Max rows per (category, family_id); None = no cap",
    )
    max_items_per_family: Optional[int] = Field(
        None,
        ge=1,
        description="Global cap: max rows per family_id after category grouping (before per-category family cap)",
    )
    category_balance_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Optional positive weights; higher → earlier pulls in round-robin across strata (by category)",
    )
    target_pool_size: Optional[int] = Field(
        None,
        ge=1,
        description="Optional global cap; soft round-robin across strata stops at N",
    )
    length_bucket_edges: List[int] = Field(
        default_factory=lambda: [80, 200],
        description="Char-length bucket boundaries for balancing strata",
    )
    nlg_slice_enabled: bool = Field(True, description="Emit NLG heuristic tags + slices")


class TargetSliceDefinition(BaseModel):
    """Named slice over balanced pool (ids only), in addition to built-in category/probe/repaired slices."""

    name: str = Field(..., min_length=1)
    categories: List[str] = Field(default_factory=list, description="If non-empty, include rows in these categories")
    source_statuses_any: List[str] = Field(
        default_factory=list,
        description="If non-empty, row must have any of these source_statuses",
    )

    @model_validator(mode="after")
    def _non_empty_filter(self) -> TargetSliceDefinition:
        if not self.categories and not self.source_statuses_any:
            raise ValueError(
                "target_slices: each entry needs non-empty 'categories' and/or 'source_statuses_any'"
            )
        return self


class HardSubsetScoringConfig(BaseModel):
    """Weights / caps for hard-subset scoring (Node 4)."""

    semantic_gate_weight: float = Field(0.12, ge=0.0, le=1.0)
    benchmark_value_cap: float = Field(0.28, ge=0.0, le=1.0)
    refusal_signal_cap: float = Field(0.55, ge=0.0, le=1.0)
    refusal_model_weight: float = Field(0.12, ge=0.0, le=1.0, description="Per-model refusal contribution (capped by refusal_signal_cap)")
    key_probe_refusal_weight: float = Field(0.22, ge=0.0, le=1.0)


class HardSubsetConfig(BaseModel):
    """Hard subset (Node 4): probe-refusal signals, diversity, top-k."""

    enabled: bool = True
    fraction: float = Field(0.15, ge=0.0, le=1.0)
    min_score: float = Field(0.0, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, description="Cap subset size; overrides fraction when set (>=1 when set)")
    min_categories_in_subset: int = Field(2, ge=1)
    max_fraction_per_category: float = Field(0.45, ge=0.0, le=1.0)
    key_probe_model: Optional[str] = None
    min_distinct_models_with_refusal: int = Field(2, ge=1)
    key_probe_refusal_alone_ok: bool = True
    duplicate_similarity_threshold: float = Field(0.92, ge=0.0, le=1.0)
    require_semantic_status: List[str] = Field(default_factory=lambda: ["validated", "probe_positive"])
    benchmark_value_long_text_threshold: int = Field(120, ge=1)
    dominated_penalty: float = Field(0.38, ge=0.0, le=1.0)
    allow_without_probe_evidence: bool = False
    scoring: HardSubsetScoringConfig = Field(default_factory=HardSubsetScoringConfig)

    @field_validator("top_k")
    @classmethod
    def _top_k_positive(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError("hard_subset.top_k must be >= 1 when set")
        return v


class SplitConfig(BaseModel):
    dev_fraction: float = Field(0.7, ge=0.0, le=1.0)
    test_fraction: float = Field(0.2, ge=0.0, le=1.0)
    random_seed: int = 42

    @model_validator(mode="after")
    def _split_sum(self) -> SplitConfig:
        if self.dev_fraction + self.test_fraction > 1.0 + 1e-9:
            raise ValueError(
                f"split.dev_fraction + split.test_fraction must be <= 1.0, got {self.dev_fraction + self.test_fraction}"
            )
        return self


class SplitExportConfig(BaseModel):
    """Node 5: leakage, repaired mixing, shuffle."""

    enforce_single_split_per_family: bool = Field(
        True,
        description="All rows sharing family_id stay in one split",
    )
    enforce_single_split_per_dedup_cluster: bool = Field(
        True,
        description="Dedup cluster members stay in one split",
    )
    family_leakage_policy: Literal["strict", "off"] = Field(
        "strict",
        description="strict: apply enforce_single_split_per_family; off: allow same family in multiple splits",
    )
    cluster_leakage_policy: Literal["strict", "off"] = Field(
        "strict",
        description="strict: apply enforce_single_split_per_dedup_cluster; off: allow cluster to span splits",
    )
    balance_repaired_across_splits: bool = Field(
        True,
        description="Interleave repaired-heavy groups when ordering",
    )
    repaired_original_mixing: Literal["interleave", "repaired_first", "non_repaired_first"] = Field(
        "interleave",
        description="Group ordering before shuffle: interleave | block repaired first | block non-repaired first",
    )
    shuffle_groups_with_seed: bool = Field(
        True,
        description="Shuffle constraint groups after ordering (deterministic via split.random_seed)",
    )


class ExportFormatsConfig(BaseModel):
    """What Node 5 writes under stage3_exports/."""

    write_jsonl: bool = Field(True, description="benchmark_<split>.jsonl")
    write_csv: bool = Field(True, description="benchmark_<split>.csv")
    write_hard_subset_exports: bool = Field(True, description="hard_subset_benchmark.*")
    write_final_export_jsonl: bool = Field(True, description="final_export.jsonl row pointers")

    @model_validator(mode="after")
    def _at_least_one_tabular(self) -> ExportFormatsConfig:
        if not self.write_jsonl and not self.write_csv:
            raise ValueError("export_formats: enable write_jsonl and/or write_csv")
        return self


class DryRunConfig(BaseModel):
    max_rows_per_source: int = Field(500, ge=1)


class ResumeConfig(BaseModel):
    skip_completed_nodes: bool = True


class RuntimeConfig(BaseModel):
    logging_verbosity: Literal["debug", "info", "warning", "error"] = "info"


class AuditPackConfig(BaseModel):
    """Optional human audit samples (exported via ``export-stage3-audit-packs`` only; does not run in the main pipeline)."""

    random_count: int = Field(50, ge=0, description="Rows for audit_pack_random.jsonl; 0 = skip")
    hard_count: int = Field(30, ge=0, description="Rows for audit_pack_hard.jsonl (from hard subset)")
    repaired_count: int = Field(25, ge=0, description="Rows for audit_pack_repaired.jsonl")
    random_seed: int = Field(42, description="Deterministic RNG for random/repaired sampling")
    sample_random_from_split: Literal["all", "dev", "test", "review_holdout"] = Field(
        "all",
        description="Limit random pool to rows assigned to this split (needs split_assignment.jsonl)",
    )
    hard_order: Literal["score_desc", "random"] = Field(
        "score_desc",
        description="How to pick hard-pack rows when hard_count < eligible count",
    )


class Stage3Policy(BaseModel):
    """Full Stage 3 policy. Use nested YAML sections (``qc_thresholds``, ``near_duplicate``, …) or flat ``qc_dedup`` — they merge."""

    model_config = {"extra": "forbid"}

    version: int = Field(1, ge=1)
    inputs: InputPaths
    filenames: FilenameMap = Field(default_factory=FilenameMap)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    qc_dedup: QcDedupClusterConfig = Field(default_factory=QcDedupClusterConfig)
    balance: BalanceConfig = Field(default_factory=BalanceConfig)
    target_slices: List[TargetSliceDefinition] = Field(
        default_factory=list,
        description="Extra named slices in slice_index (categories / statuses filters)",
    )
    hard_subset: HardSubsetConfig = Field(default_factory=HardSubsetConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    split_export: SplitExportConfig = Field(default_factory=SplitExportConfig)
    export_formats: ExportFormatsConfig = Field(default_factory=ExportFormatsConfig)
    dry_run: DryRunConfig = Field(default_factory=DryRunConfig)
    resume: ResumeConfig = Field(default_factory=ResumeConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    audit_packs: AuditPackConfig = Field(
        default_factory=AuditPackConfig,
        description="Defaults for audit JSONL packs (CLI export only)",
    )

    @model_validator(mode="before")
    @classmethod
    def _merge_nested_yaml_into_flat(cls, data: Any) -> Any:
        """Allow structured YAML without breaking legacy flat keys."""
        if not isinstance(data, dict):
            return data

        qt = data.pop("qc_thresholds", None)
        nd = data.pop("near_duplicate", None)
        cl = data.pop("clustering", None)
        if qt or nd or cl:
            qd: Dict[str, Any] = dict(data.get("qc_dedup") or {})
            if isinstance(qt, dict):
                if "min_text_len" in qt:
                    qd["min_text_len"] = qt["min_text_len"]
                if "max_repeat_char_run" in qt:
                    qd["max_repeat_char_run"] = qt["max_repeat_char_run"]
                if "injection_patterns" in qt:
                    qd["injection_patterns"] = qt["injection_patterns"]
            if isinstance(nd, dict):
                if "threshold" in nd:
                    qd["near_duplicate_threshold"] = nd["threshold"]
                if "metric" in nd:
                    qd["near_duplicate_metric"] = nd["metric"]
                if "normalization" in nd:
                    qd["near_normalization"] = nd["normalization"]
            if isinstance(cl, dict):
                if "seed" in cl:
                    qd["clustering_seed"] = cl["seed"]
                if "dedup_strata" in cl:
                    qd["dedup_strata"] = cl["dedup_strata"]
            data["qc_dedup"] = qd

        bal = data.pop("balancing", None)
        if isinstance(bal, dict):
            b = dict(data.get("balance") or {})
            for k in (
                "max_per_category",
                "min_per_category",
                "max_rows_per_family_per_category",
                "max_items_per_family",
                "category_balance_weights",
                "target_pool_size",
                "length_bucket_edges",
                "nlg_slice_enabled",
            ):
                if k in bal:
                    b[k] = bal[k]
            if "category_targets" in bal and isinstance(bal["category_targets"], dict):
                b.setdefault("category_balance_weights", bal["category_targets"])
            data["balance"] = b

        return data
