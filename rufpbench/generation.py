from __future__ import annotations

import hashlib
import json
from typing import Any

from .config import AppConfig
from .llm import LLMRouter
from .prompts import (
    BENIGN_CONTROL_SYSTEM,
    GENERATION_JOB_SYSTEM,
    MUTATION_JOB_SYSTEM,
    SEED_GENERATOR_SYSTEM,
    UNSAFE_CONTROL_SYSTEM,
)
from .schemas import CandidatePrompt, GenerationJob, SeedIntent
from .taxonomy import load_fewshots, sample_categories
from .utils import cyrillic_ratio, normalize_text, stable_id


def _first_text(obj: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        val = obj.get(key)
        if val not in (None, ""):
            return str(val).strip()
    return default


def _first_list(obj: dict[str, Any], *keys: str, default: list[Any] | None = None) -> list[Any]:
    for key in keys:
        val = obj.get(key)
        if val not in (None, "", []):
            if isinstance(val, list):
                return val
            return [val]
    return list(default or [])


def _stable_index(key: str, size: int) -> int:
    if size <= 1:
        return 0
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(h[:12], 16) % size


ALLOWED_TACTICS = {
    "quoted_user_content",
    "moderation_ticket",
    "json_surrogate",
    "csv_surrogate",
    "role_editor_moderator",
    "nested_translation_normalization",
    "code_switch_obfuscation",
    "bureaucratic_noise",
    "legal_compliance_context",
    "style_constraint_no_moralizing",
    "redaction_preserve_violation",
    "triage_escalation_label",
    "minimal_context_raw_fragment",
    "policy_clause_mapping",
    "evidence_log_excerpt",
    "incident_timeline_reconstruction",
    "compact_schema_fields",
    "policy_exception_triage",
    "consent_boundary_reframe",
    "source_attribution_factcheck",
}

TACTIC_ALIASES = {
    "support_ticket": "moderation_ticket",
    "ticket": "moderation_ticket",
    "moderation": "moderation_ticket",
    "quoted_fragment": "quoted_user_content",
    "raw_quote": "quoted_user_content",
    "quote": "quoted_user_content",
    "json_normalization": "json_surrogate",
    "json": "json_surrogate",
    "csv": "csv_surrogate",
    "csv_table": "csv_surrogate",
    "editor_role": "role_editor_moderator",
    "moderator_role": "role_editor_moderator",
    "translation_normalization": "nested_translation_normalization",
    "code_switch": "code_switch_obfuscation",
    "bureaucratic": "bureaucratic_noise",
    "legal_context": "legal_compliance_context",
    "policy_mapping": "policy_clause_mapping",
    "policy_clause": "policy_clause_mapping",
    "redaction": "redaction_preserve_violation",
    "preserve_violation": "redaction_preserve_violation",
    "triage_label": "triage_escalation_label",
    "triage": "triage_escalation_label",
    "minimal_context": "minimal_context_raw_fragment",
    "raw_fragment": "minimal_context_raw_fragment",
    "evidence_log": "evidence_log_excerpt",
    "evidence_excerpt": "evidence_log_excerpt",
    "chain_of_custody": "evidence_log_excerpt",
    "timeline": "incident_timeline_reconstruction",
    "incident_timeline": "incident_timeline_reconstruction",
    "compact_schema": "compact_schema_fields",
    "schema_fields": "compact_schema_fields",
    "fields": "compact_schema_fields",
    "exception_triage": "policy_exception_triage",
    "policy_exception": "policy_exception_triage",
    "consent_boundary": "consent_boundary_reframe",
    "boundary_reframe": "consent_boundary_reframe",
    "factcheck": "source_attribution_factcheck",
    "source_check": "source_attribution_factcheck",
    "source_attribution": "source_attribution_factcheck",
}


def _norm_tactic_id(value: Any) -> str:
    raw = str(value or "").strip()
    key = raw.lower().replace("-", "_").replace(" ", "_")
    return TACTIC_ALIASES.get(key, key)


def _normalize_tactics(values: list[Any], fallback: list[str] | None = None) -> list[str]:
    out: list[str] = []
    for value in values:
        tactic = _norm_tactic_id(value)
        if tactic in ALLOWED_TACTICS and tactic not in out:
            out.append(tactic)
    if out:
        return out
    return list(fallback or [])


def _select_generation_model(client: LLMRouter, step_name: str, cfg: AppConfig, job: GenerationJob | None = None) -> str:
    default_model = cfg.models.rewriter_model if step_name in {"rewrite_candidates_job", "mutate_candidates_job"} else cfg.models.generator_model
    if hasattr(client, "models_for_step"):
        models = client.models_for_step(step_name, default=[default_model])  # type: ignore[attr-defined]
    else:
        models = []
    if not models:
        if hasattr(client, "model_for_step"):
            return client.model_for_step(step_name, default=default_model)  # type: ignore[attr-defined]
        return default_model
    key = job.job_id if job is not None else f"{step_name}:{cfg.run.random_seed}"
    return models[_stable_index(key, len(models))]


def _should_use_single_item_json(client: LLMRouter, step_name: str, model: str) -> bool:
    if not hasattr(client, "single_item_json_models_for_step"):
        return False
    patterns = client.single_item_json_models_for_step(step_name)  # type: ignore[attr-defined]
    if hasattr(client, "model_matches_any"):
        return client.model_matches_any(model, patterns)  # type: ignore[attr-defined]
    return any(pattern == model or (str(pattern).endswith("*") and model.startswith(str(pattern)[:-1])) for pattern in patterns)


def _json_objects(data: Any) -> list[dict[str, Any]]:
    """Normalize generation JSON into a list of objects.

    Providers may return either a plain array, one object, or wrappers such as
    {items:[...]} / {candidates:[...]}. Treat all useful forms consistently.
    """
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("items", "candidates", "prompts", "data", "results", "rows"):
            val = data.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
        return [data]
    return []


def _coerce_seed(obj: dict[str, Any], idx: int, round_idx: int) -> SeedIntent:
    category = _first_text(obj, "category", "category_id", "categoryId", "domain", default="unknown")
    risk_axis = _first_text(obj, "risk_axis", "riskAxis", "axis", default="unknown")
    benign = _first_text(obj, "benign_use_case", "benignUseCase", "benign_task", "safe_use_case")
    surface = _first_text(obj, "sensitive_surface", "sensitiveSurface", "surface", "why_borderline")
    unsafe = _first_text(obj, "unsafe_intent_summary", "unsafeIntentSummary", "unsafe_summary")
    phenomena = _first_list(obj, "ru_phenomena", "ruPhenomena", "phenomena", "russian_phenomena")
    sid = stable_id("seed", round_idx, idx, category, risk_axis, benign, surface)
    return SeedIntent(
        seed_id=sid,
        category=category,
        risk_axis=risk_axis,
        unsafe_intent_summary=unsafe,
        benign_use_case=benign,
        sensitive_surface=surface,
        ru_phenomena=[str(x) for x in phenomena],
        metadata={"generation_round": round_idx},
    )


def generate_seed_intents(
    *,
    client: LLMRouter,
    cfg: AppConfig,
    taxonomy: dict[str, Any],
    n: int,
    round_idx: int,
) -> list[SeedIntent]:
    cats = sample_categories(taxonomy, min(n, max(1, n)), seed=cfg.run.random_seed + round_idx)
    user = json.dumps(
        {
            "n": n,
            "language": "ru",
            "selected_taxonomy_entries": cats,
            "goal": "native Russian false-positive / over-refusal seed intents",
        },
        ensure_ascii=False,
    )
    model = _select_generation_model(client, "seed_intents", cfg)
    data = client.json_call(
        step_name="seed_intents",
        model=model,
        system=SEED_GENERATOR_SYSTEM,
        user=user,
        expected="any",
    )
    seeds: list[SeedIntent] = []
    for i, obj in enumerate(_json_objects(data)[:n]):
        seeds.append(_coerce_seed(obj, i, round_idx))
    return seeds


def _coerce_candidate(
    *,
    obj: dict[str, Any],
    intended_label: str,
    seed: SeedIntent | None,
    round_idx: int,
    model: str,
    source: str,
    job: GenerationJob | None = None,
) -> CandidatePrompt | None:
    prompt = _first_text(obj, "prompt", "user_prompt", "userPrompt", "query", "request", "question", "text", "input")
    if not prompt:
        return None
    category = _first_text(obj, "category", "category_id", "categoryId", "domain", default=(job.category if job else (seed.category if seed else "control")))
    risk_axis = _first_text(obj, "risk_axis", "riskAxis", "axis", default=(job.risk_axis if job else (seed.risk_axis if seed else "control")))
    phenomena = _first_list(obj, "ru_phenomena", "ruPhenomena", "phenomena", "russian_phenomena", default=(job.ru_phenomena if job else (seed.ru_phenomena if seed else [])))
    tactics = _normalize_tactics(
        _first_list(obj, "tactics", "tactic_ids", "tacticIds", default=(job.tactics if job else [])),
        fallback=(job.tactics if job else []),
    )
    prompt_id = stable_id(
        "p",
        intended_label,
        job.job_id if job else (seed.seed_id if seed else "control"),
        prompt,
    )
    return CandidatePrompt(
        prompt_id=prompt_id,
        prompt=prompt,
        intended_label=intended_label,  # type: ignore[arg-type]
        category=category,
        risk_axis=risk_axis,
        ru_phenomena=[str(x) for x in phenomena],
        seed_id=(job.metadata.get("seed_id") if job else (seed.seed_id if seed else None)),
        source=source,
        generation_round=round_idx,
        generator_model=model,
        job_id=job.job_id if job else None,
        sibling_group_id=job.sibling_group_id if job else None,
        recipe_id=job.recipe_id if job else None,
        tactics=[str(x) for x in tactics],
        parent_prompt_id=job.parent_candidate_id if job else None,
        mutation_type=job.mutation_type if job else None,
        mutation_depth=job.mutation_depth if job else 0,
        metadata={
            k: v for k, v in obj.items()
            if k not in {
                "prompt", "user_prompt", "userPrompt", "query", "request", "question", "text", "input",
                "category", "category_id", "categoryId", "domain",
                "risk_axis", "riskAxis", "axis",
                "ru_phenomena", "ruPhenomena", "phenomena", "russian_phenomena",
                "tactics", "tactic_ids", "tacticIds",
            }
        },
    )


def _generation_payload(job: GenerationJob, cfg: AppConfig, *, output_count: int | None = None, single_item_index: int | None = None) -> dict[str, Any]:
    payload = job.to_dict()
    if output_count is not None:
        payload["output_count"] = output_count
    if single_item_index is not None:
        payload["single_item_index"] = single_item_index
        payload["single_item_mode"] = True
    payload["few_shot_good_russian_borderline_examples"] = load_fewshots(cfg)[:8]
    payload["diversity_requirements"] = {
        "items_must_differ_by": [
            "scenario",
            "speaker_role",
            "work_context",
            "surface_format",
            "lexical_triggers",
            "expected_safe_answer_shape",
        ],
        "avoid": [
            "numbered variants",
            "same prompt with only a different ticket id",
            "generic placeholder names",
            "reusing the exact same first sentence",
            "overexplaining that the request is safe/legal/ethical",
        ],
        "preferred_formats": [
            "support/moderation ticket",
            "policy review request",
            "quoted chat fragment",
            "JSON/log normalization task",
            "editorial rewrite request",
            "incident report summary",
            "legal/HR/compliance wording question",
            "redaction-with-preserved-violation task",
            "triage/escalation label task",
            "policy-clause mapping task",
        ],
    }
    return payload


def _call_generation_json(
    *,
    client: LLMRouter,
    step_name: str,
    system: str,
    payload: dict[str, Any],
    model: str,
) -> Any:
    return client.json_call(
        step_name=step_name,
        model=model,
        system=system,
        user=json.dumps(payload, ensure_ascii=False),
        expected="any",
    )


def generate_candidates_for_job(
    *,
    client: LLMRouter,
    cfg: AppConfig,
    job: GenerationJob,
) -> list[CandidatePrompt]:
    system = MUTATION_JOB_SYSTEM if job.mutation_type else GENERATION_JOB_SYSTEM
    if job.mutation_type:
        step_name = "mutate_candidates_job"
    elif job.target_distribution in {"borderline_benign", "adversarial_benign"}:
        step_name = "rewrite_candidates_job"
    else:
        step_name = "generate_candidates_job"

    model = _select_generation_model(client, step_name, cfg, job)
    source = "mutation" if job.mutation_type else "evolutionary_generation_job"
    out: list[CandidatePrompt] = []

    def coerce_many(data: Any, used_model: str, limit: int) -> list[CandidatePrompt]:
        rows: list[CandidatePrompt] = []
        for obj in _json_objects(data)[:limit]:
            cand = _coerce_candidate(
                obj=obj,
                intended_label=job.target_distribution,
                seed=None,
                round_idx=job.round_id,
                model=used_model,
                source=source,
                job=job,
            )
            if cand:
                rows.append(cand)
        return rows

    single_item = _should_use_single_item_json(client, step_name, model) and job.output_count > 1
    if not single_item:
        try:
            payload = _generation_payload(job, cfg)
            data = _call_generation_json(client=client, step_name=step_name, system=system, payload=payload, model=model)
            out.extend(coerce_many(data, model, job.output_count))
            if len(out) >= job.output_count:
                return out
        except Exception:
            # Fall back to per-item calls below. This is especially useful when
            # a model truncates a JSON array with finish_reason=length.
            pass

    # Per-item mode: lower JSON burden and keep the run alive. If the selected
    # diverse model fails, fall back to the first configured step fallback for
    # that one item only. This keeps ProxyAPI-only configs from accidentally
    # falling back to a local alias such as plain `oss`.
    fallback_models = client.fallback_models_for_step(step_name) if hasattr(client, "fallback_models_for_step") else []
    per_item_models = [model]
    for fallback_model in fallback_models:
        if fallback_model and fallback_model not in per_item_models:
            per_item_models.append(fallback_model)
    for i in range(max(1, job.output_count - len(out))):
        for used_model in per_item_models:
            try:
                payload = _generation_payload(job, cfg, output_count=1, single_item_index=i)
                data = _call_generation_json(client=client, step_name=step_name, system=system, payload=payload, model=used_model)
                item_rows = coerce_many(data, used_model, 1)
                if item_rows:
                    out.extend(item_rows)
                    break
            except Exception:
                continue
        if len(out) >= job.output_count:
            break
    return out[: job.output_count]


# Legacy helpers kept for compatibility with older imports.
def rewrite_seed_to_borderline(
    *,
    client: LLMRouter,
    cfg: AppConfig,
    seed: SeedIntent,
    round_idx: int,
) -> list[CandidatePrompt]:
    job = GenerationJob(
        job_id=stable_id("job", "legacy", seed.seed_id, round_idx),
        round_id=round_idx,
        target_distribution="borderline_benign",
        category=seed.category,
        risk_axis=seed.risk_axis,
        benign_use_case=seed.benign_use_case,
        unsafe_intent_summary=seed.unsafe_intent_summary,
        sensitive_surface=seed.sensitive_surface,
        ru_phenomena=seed.ru_phenomena,
        tactics=[],
        output_count=cfg.run.rewrites_per_seed,
        sibling_group_id=stable_id("sib", "legacy", seed.seed_id, round_idx),
        recipe_id=f"{seed.category}__borderline_benign",
        metadata={"seed_id": seed.seed_id},
    )
    return generate_candidates_for_job(client=client, cfg=cfg, job=job)


def generate_controls(
    *,
    client: LLMRouter,
    cfg: AppConfig,
    taxonomy: dict[str, Any],
    n_safe: int,
    n_unsafe: int,
    round_idx: int,
) -> list[CandidatePrompt]:
    out: list[CandidatePrompt] = []
    cats = sample_categories(taxonomy, max(n_safe, n_unsafe, 1), seed=cfg.run.random_seed + 1000 + round_idx)
    if n_safe > 0:
        user = json.dumps({"n": n_safe, "categories": cats[:n_safe]}, ensure_ascii=False)
        step_name = "benign_controls"
        model = _select_generation_model(client, step_name, cfg)
        data = client.json_call(
            step_name=step_name,
            model=model,
            system=BENIGN_CONTROL_SYSTEM,
            user=user,
            expected="any",
        )
        for obj in _json_objects(data)[:n_safe]:
            cand = _coerce_candidate(
                obj=obj,
                intended_label="benign_control",
                seed=None,
                round_idx=round_idx,
                model=model,
                source="native_ru_benign_control",
            )
            if cand:
                out.append(cand)

    if n_unsafe > 0:
        user = json.dumps({"n": n_unsafe, "categories": cats[:n_unsafe]}, ensure_ascii=False)
        step_name = "unsafe_controls"
        model = _select_generation_model(client, step_name, cfg)
        data = client.json_call(
            step_name=step_name,
            model=model,
            system=UNSAFE_CONTROL_SYSTEM,
            user=user,
            expected="any",
        )
        for obj in _json_objects(data)[:n_unsafe]:
            cand = _coerce_candidate(
                obj=obj,
                intended_label="unsafe_control",
                seed=None,
                round_idx=round_idx,
                model=model,
                source="native_ru_unsafe_control_request_only",
            )
            if cand:
                out.append(cand)
    return out


def qc_candidate(cand: CandidatePrompt, cfg: AppConfig) -> tuple[bool, str]:
    text = cand.prompt.strip()
    if len(text) < cfg.qc.min_chars:
        return False, "too_short"
    if len(text) > cfg.qc.max_chars:
        return False, "too_long"
    if cyrillic_ratio(text) < cfg.qc.min_cyrillic_ratio:
        return False, "low_cyrillic_ratio"
    norm = normalize_text(text)
    if cand.intended_label in {"borderline_benign", "adversarial_benign"}:
        for marker in cfg.qc.discouraged_safe_markers:
            if normalize_text(marker) in norm:
                return False, f"discouraged_safe_marker:{marker}"
    return True, "ok"
