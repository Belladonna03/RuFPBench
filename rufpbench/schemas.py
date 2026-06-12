from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

PromptSafety = Literal["benign", "unsafe", "ambiguous", "invalid"]
RefusalLabel = Literal[
    "full_refusal",
    "partial_refusal",
    "compliance",
    "extra_safety_guidance",
    "format_deviation",
    "clarification_request",
    "empty",
    "error",
    "unclear",
]
IntendedDistribution = Literal[
    "borderline_benign",
    "adversarial_benign",
    "benign_control",
    "unsafe_control",
]
FinalBucket = Literal[
    "safe_answered",
    "safe_refused_borderline",
    "benign_prompt_unsafe_response",
    "unsafe_safe_handled",
    "unsafe_refused",
    "unsafe_answered",
    "quarantine",
]
RouterAction = Literal[
    "accept",
    "drop",
    "repair_to_benign",
    "harden_borderline",
    "harden_for_more_refusals",
    "clarify_benign_intent",
    "make_unsafe_control_less_trivial",
    "constrain_safe_answer",
    "relabel_accept",
    "quarantine",
]


@dataclass
class SeedIntent:
    seed_id: str
    category: str
    risk_axis: str
    unsafe_intent_summary: str
    benign_use_case: str
    sensitive_surface: str
    ru_phenomena: list[str]
    source: str = "synthetic_native_ru"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TacticSpec:
    tactic_id: str
    family: str
    ru_name: str
    use_for: list[str]
    safe_rule: str
    forbidden_rule: str
    mutation_goal: str
    source: str = "wildteaming_ru_adapted"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationJob:
    job_id: str
    round_id: int
    target_distribution: IntendedDistribution
    category: str
    risk_axis: str
    benign_use_case: str | None
    unsafe_intent_summary: str | None
    sensitive_surface: str
    ru_phenomena: list[str]
    tactics: list[str]
    tactic_specs: list[dict[str, Any]] = field(default_factory=list)
    output_count: int = 5
    sibling_group_id: str = ""
    recipe_id: str = ""
    source: str = "sampler"
    parent_candidate_id: str | None = None
    parent_prompt: str | None = None
    mutation_type: str | None = None
    mutation_depth: int = 0
    failure_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidatePrompt:
    prompt_id: str
    prompt: str
    intended_label: IntendedDistribution
    category: str
    risk_axis: str
    ru_phenomena: list[str]
    seed_id: str | None = None
    source: str = "synthetic"
    generation_round: int = 0
    generator_model: str = ""
    job_id: str | None = None
    sibling_group_id: str | None = None
    recipe_id: str | None = None
    tactics: list[str] = field(default_factory=list)
    parent_prompt_id: str | None = None
    mutation_type: str | None = None
    mutation_depth: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeVote:
    model: str
    label: PromptSafety
    confidence: float
    category: str = ""
    rationale: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromptSafetyResult:
    prompt_id: str
    final_label: PromptSafety
    benign_votes: int
    unsafe_votes: int
    ambiguous_votes: int
    invalid_votes: int
    votes: list[JudgeVote]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["votes"] = [v.to_dict() for v in self.votes]
        return d


@dataclass
class TargetResponse:
    prompt_id: str
    model: str
    response: str
    refusal_label: RefusalLabel
    refusal_confidence: float
    response_safety_label: PromptSafety = "ambiguous"
    latency_ms: float | None = None
    error: str = ""
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, redact_response: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if redact_response:
            d["response"] = "[REDACTED_BY_RUFPBENCH]"
        return d


@dataclass
class FinalRecord:
    prompt_id: str
    prompt: str
    intended_label: str
    prompt_safety_label: PromptSafety
    final_bucket: FinalBucket
    category: str
    risk_axis: str
    ru_phenomena: list[str]
    seed_id: str | None
    refused_by: list[str]
    answered_by: list[str]
    unsafe_answered_by: list[str]
    hard_score: float
    prompt_safety_votes: list[dict[str, Any]]
    target_results: list[dict[str, Any]]
    tactics: list[str] = field(default_factory=list)
    job_id: str | None = None
    sibling_group_id: str | None = None
    recipe_id: str | None = None
    parent_prompt_id: str | None = None
    mutation_type: str | None = None
    mutation_depth: int = 0
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RouterDecision:
    candidate_id: str
    prompt_id: str
    action: RouterAction
    reason: str
    observed_bucket: str
    intended_distribution: str
    next_job: GenerationJob | None = None
    accepted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.next_job is not None:
            d["next_job"] = self.next_job.to_dict()
        return d


@dataclass
class RecipeStats:
    recipe_id: str
    attempts: int = 0
    safe_refused_borderline: int = 0
    safe_answered: int = 0
    unsafe_safe_handled: int = 0
    unsafe_refused: int = 0
    unsafe_answered: int = 0
    benign_prompt_unsafe_response: int = 0
    quarantine: int = 0
    dropped: int = 0

    @property
    def yield_rate(self) -> float:
        return self.safe_refused_borderline / max(1, self.attempts)

    @property
    def unsafe_rate(self) -> float:
        return (self.unsafe_refused + self.unsafe_answered) / max(1, self.attempts)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["yield_rate"] = round(self.yield_rate, 4)
        d["unsafe_rate"] = round(self.unsafe_rate, 4)
        return d
