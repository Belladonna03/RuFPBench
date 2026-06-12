from __future__ import annotations

import random
from typing import Any

from .config import AppConfig
from .mutations import make_mutation_job
from .schemas import CandidatePrompt, FinalRecord, RouterDecision, TacticSpec


def route_candidate(
    *,
    cfg: AppConfig,
    candidate: CandidatePrompt,
    record: FinalRecord,
    next_round_id: int,
    tactics: dict[str, TacticSpec],
    compatibility: dict[str, Any],
    rng: random.Random,
) -> RouterDecision:
    intended = candidate.intended_label
    bucket = record.final_bucket
    safety = record.prompt_safety_label

    action = "accept"
    reason = "observed_bucket_is_usable"
    accepted = True

    if bucket == "safe_refused_borderline":
        if len(record.refused_by) < cfg.validation.hard_min_refusal_models and candidate.mutation_depth < cfg.run.max_mutation_depth:
            action = "harden_for_more_refusals"
            reason = "valid_fp_try_to_make_harder"
            accepted = True
        else:
            action = "accept"
            reason = "valid_fp_or_hard_case"
    elif bucket == "safe_answered":
        if intended in {"borderline_benign", "adversarial_benign"}:
            action = "harden_borderline"
            reason = "benign_but_no_refusal"
            accepted = True  # still useful as safe_answered control
        elif intended == "benign_control":
            action = "accept"
            reason = "valid_safe_answered_control"
            accepted = True
        else:
            action = "relabel_accept"
            reason = "unsafe_control_observed_as_benign_answered"
            accepted = True
    elif bucket == "benign_prompt_unsafe_response":
        action = "constrain_safe_answer"
        reason = "benign_prompt_triggered_unsafe_target_response"
        accepted = True
    elif bucket in {"unsafe_refused", "unsafe_safe_handled"}:
        if intended in {"borderline_benign", "adversarial_benign", "benign_control"}:
            action = "repair_to_benign"
            reason = "intended_benign_but_prompt_judged_unsafe"
            accepted = True  # keep observed control signal, but repair for FP mining
        else:
            action = "accept"
            reason = "valid_unsafe_safe_handled_control" if bucket == "unsafe_safe_handled" else "valid_unsafe_refused_control"
            accepted = True
    elif bucket == "unsafe_answered":
        if intended in {"borderline_benign", "adversarial_benign", "benign_control"}:
            action = "repair_to_benign"
            reason = "intended_benign_but_unsafe_answered"
            accepted = True
        else:
            action = "accept"
            reason = "valid_unsafe_answered_fn_control"
            accepted = True
    else:  # quarantine
        if intended in {"borderline_benign", "adversarial_benign", "benign_control"} and safety in {"ambiguous", "invalid"}:
            action = "clarify_benign_intent"
            reason = "ambiguous_or_invalid_prompt_needs_benign_clarification"
            accepted = bool(cfg.validation.keep_ambiguous_quarantine)
        elif intended == "unsafe_control":
            action = "make_unsafe_control_less_trivial"
            reason = "unsafe_control_not_clearly_classified_or_became_benign"
            accepted = False
        else:
            action = "quarantine"
            reason = "not_actionable_quarantine"
            accepted = bool(cfg.validation.keep_ambiguous_quarantine)

    next_job = None
    if action in {
        "repair_to_benign",
        "harden_borderline",
        "harden_for_more_refusals",
        "clarify_benign_intent",
        "make_unsafe_control_less_trivial",
        "constrain_safe_answer",
    }:
        next_job = make_mutation_job(
            cfg=cfg,
            candidate=candidate,
            record=record,
            action=action,
            reason=reason,
            round_id=next_round_id,
            tactics=tactics,
            compatibility=compatibility,
            rng=rng,
        )

    return RouterDecision(
        candidate_id=candidate.prompt_id,
        prompt_id=candidate.prompt_id,
        action=action,  # type: ignore[arg-type]
        reason=reason,
        observed_bucket=bucket,
        intended_distribution=intended,
        next_job=next_job,
        accepted=accepted,
        metadata={"prompt_safety_label": safety, "hard_score": record.hard_score},
    )
