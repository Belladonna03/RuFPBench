"""
Independent LLM validation of verified pairs (e.g. gpt-oss) before run_target.
Sees unsafe_contrast only for pair quality scoring; does not send it to the target chat model.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .prompts import VALIDATE_PAIRS_SYSTEM_PROMPT, VALIDATE_PAIRS_USER_TEMPLATE
from .utils import (
    REQUIRED_PAIR_FIELDS,
    add_file_handler,
    append_jsonl_line,
    create_openai_client,
    extract_json_object,
    get_logger,
    mask_secret,
    normalize_for_dedup,
    read_env,
    read_jsonl,
    sleep_before_retry,
    warn_if_same_model_id,
    write_jsonl,
)

ENV_GENERATOR_MODEL = "RU_FP_GENERATOR_MODEL"
ENV_VALIDATOR_BASE_URL = "RU_FP_VALIDATOR_BASE_URL"
ENV_VALIDATOR_MODEL = "RU_FP_VALIDATOR_MODEL"
ENV_VALIDATOR_API_KEY = "RU_FP_VALIDATOR_API_KEY"

_PRIOR_MAX = 24
_PRIOR_SNIP_LEN = 220
_API_LOCK = threading.Lock()
_STAGE_LOG_NAME = "validator_runs.jsonl"  # under --log-dir (separate from run_dir/validator_runs.jsonl artifact)


def _format_prior_snippets(snippets: List[str]) -> str:
    if not snippets:
        return "нет (это первая строка во входе или список пуст)."
    lines: List[str] = []
    for i, s in enumerate(snippets[-_PRIOR_MAX:], start=1):
        t = s.strip().replace("\n", " ")
        if len(t) > _PRIOR_SNIP_LEN:
            t = t[:_PRIOR_SNIP_LEN] + "…"
        lines.append(f"{i}. {t}")
    return "\n".join(lines)


def _row_to_out(row: Dict) -> Dict[str, str]:
    return {k: str(row.get(k) or "") for k in REQUIRED_PAIR_FIELDS}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes", "да"):
            return True
        if v in ("false", "0", "no", "нет"):
            return False
    return bool(value)


def _response_usage_dict(usage: Any) -> Dict[str, Any]:
    """Token usage for logs only (no API keys or secrets)."""
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return {k: usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens") if k in usage}
    out: Dict[str, Any] = {}
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        v = getattr(usage, k, None)
        if v is not None:
            out[k] = v
    return out


def append_validator_stage_log(
    log_dir: str,
    row: Dict[str, Any],
    model: str,
    v2: Dict[str, Any],
    reason: str,
    meta: Dict[str, Any],
) -> None:
    """
    One JSONL line per validated row: timestamp, model id, verdict fields, latency, usage.
    Path: {log_dir}/validator_runs.jsonl (not the run artifact validator_runs.jsonl in run-dir).
    """
    p = str(Path(log_dir).resolve() / _STAGE_LOG_NAME)
    ts = datetime.now(timezone.utc)
    line: Dict[str, Any] = {
        "timestamp": ts.isoformat().replace("+00:00", "Z"),
        "stage": "validate_pairs",
        "id": str(row.get("id", "")),
        "validator_model": model,
        "topic": str(row.get("topic", "")),
        "safe_verdict": str(v2.get("safe_verdict", "unclear")),
        "unsafe_verdict": str(v2.get("unsafe_verdict", "unclear")),
        "pair_minimal": bool(_coerce_bool(v2.get("pair_minimal", False))),
        "borderline_quality": str(v2.get("borderline_quality", "")),
        "fp_risk": str(v2.get("fp_risk", "")),
        "keep": _coerce_bool(v2.get("keep", False)),
        "reason": (reason or "")[:2000],
        "latency_sec": round(float(meta.get("latency_sec", 0.0) or 0.0), 2),
        "usage": meta.get("usage") or {},
    }
    append_jsonl_line(p, line)


def _parse_validation(obj: Dict[str, Any]) -> Tuple[bool, str, str, Dict[str, Any]]:
    """
    v2: keep, safe_verdict, unsafe_verdict, ...
    legacy: accept, rejection_code, notes (old validator output).
    """
    is_legacy = "keep" not in obj and "safe_verdict" not in obj and "accept" in obj
    if is_legacy:
        accept = _coerce_bool(obj.get("accept", False))
        code = str(obj.get("rejection_code") or ("ok" if accept else "other")).strip()
        notes = str(obj.get("notes") or "").strip()
        if len(notes) > 500:
            notes = notes[:500] + "…"
        return accept, code, notes, {"schema": "legacy"}

    reason = str(obj.get("reason", "")).strip()
    if len(reason) > 500:
        reason = reason[:500] + "…"
    safe_v = str(obj.get("safe_verdict", "unclear")).lower().strip()
    if safe_v not in ("allowed", "borderline_allowed", "disallowed", "unclear"):
        safe_v = "unclear"
    unsafe_v = str(obj.get("unsafe_verdict", "unclear")).lower().strip()
    if unsafe_v not in ("disallowed", "allowed", "unclear"):
        unsafe_v = "unclear"
    bq = str(obj.get("borderline_quality", "low")).lower().strip()
    if bq not in ("low", "medium", "high"):
        bq = "low"
    fpr = str(obj.get("fp_risk", "low")).lower().strip()
    if fpr not in ("low", "medium", "high"):
        fpr = "low"
    pair_minimal = _coerce_bool(obj.get("pair_minimal", False))
    if "keep" in obj and obj.get("keep") is not None:
        keep = _coerce_bool(obj.get("keep"))
    else:
        keep = bool(
            safe_v in ("allowed", "borderline_allowed")
            and unsafe_v == "disallowed"
            and pair_minimal
        )
    extra = {
        "schema": "v2",
        "safe_verdict": safe_v,
        "unsafe_verdict": unsafe_v,
        "pair_minimal": pair_minimal,
        "borderline_quality": bq,
        "fp_risk": fpr,
        "keep": keep,
    }
    code = "kept" if keep else f"reject:{safe_v}:{unsafe_v}"[:100]
    return keep, code, reason, extra


def _output_passes_filter(
    v2: Dict[str, Any],
    legacy_model_accept: Optional[bool] = None,
) -> bool:
    """
    Pair goes to pairs.validated.jsonl only if all conditions hold.
    *unclear* and other non-matching verdicts => False (still written to validator_runs).
    """
    if v2.get("json_parse_error") or v2.get("schema") == "error":
        return False
    if v2.get("schema") == "legacy":
        if legacy_model_accept is None:
            return False
        return bool(legacy_model_accept)
    if "safe_verdict" not in v2 and legacy_model_accept is not None:
        return bool(legacy_model_accept)
    if "safe_verdict" not in v2:
        return False
    if not _coerce_bool(v2.get("keep")):
        return False
    # Включаем в pairs.validated все пары, где валидатор поставил keep=True (временно без порога
    # borderline_quality/fp_risk, чтобы не терять примеры на первом прогоне).
    return True


def _verdict_included(verdict: Dict[str, Any]) -> bool:
    """Recompute inclusion for resume/rebuild: prefer filter_pass; re-run rules on v2; legacy accept."""
    if "filter_pass" in verdict:
        return _coerce_bool(verdict.get("filter_pass"))
    v2 = {
        k: verdict[k]
        for k in (
            "schema",
            "keep",
            "safe_verdict",
            "unsafe_verdict",
            "pair_minimal",
            "borderline_quality",
            "fp_risk",
        )
        if k in verdict
    }
    if "safe_verdict" in v2 and verdict.get("schema") != "legacy":
        if "schema" not in v2:
            v2["schema"] = "v2"
        return _output_passes_filter(v2, None)
    if verdict.get("schema") == "legacy" or "safe_verdict" not in verdict:
        return verdict.get("accept") is True
    v2["schema"] = "v2"
    return _output_passes_filter(v2, None)


def _prior_safe_requests_file_order(all_rows: List[Dict], index: int) -> List[str]:
    return [str(all_rows[j].get("safe_request") or "") for j in range(index)]


def _mock_log_meta() -> Dict[str, Any]:
    return {"latency_sec": 0.0, "usage": {}}


def mock_validate(
    row: Dict,
    accepted_norms: set,
) -> Tuple[bool, str, str, Dict[str, Any], Dict[str, Any]]:
    """Heuristic mock: drop exact / near-duplicate safe_request; otherwise accept."""
    s = normalize_for_dedup(str(row.get("safe_request", "")))[:2000]
    base = {
        "schema": "v2",
        "safe_verdict": "borderline_allowed",
        "unsafe_verdict": "disallowed",
        "pair_minimal": True,
        "borderline_quality": "medium",
        "fp_risk": "medium",
    }
    if not s:
        ex = {**base, "keep": False, "pair_minimal": False}
        return False, "reject:empty", "пустой safe_request", ex, _mock_log_meta()
    if s in accepted_norms:
        ex = {**base, "keep": False, "pair_minimal": False, "fp_risk": "low"}
        return False, "reject:duplicate", "mock: дубликат safe_request", ex, _mock_log_meta()
    accepted_norms.add(s)
    ex = {**base, "keep": True}
    return True, "kept", "mock: принято", ex, _mock_log_meta()


def real_validate(
    client: Any,
    model: str,
    row: Dict,
    prior_safe_requests: List[str],
    max_retries: int,
    logger: Any,
) -> Tuple[bool, str, str, Dict[str, Any], Dict[str, Any]]:
    risk_triggers = str(row.get("risk_triggers") or "").strip() or "—"
    user = VALIDATE_PAIRS_USER_TEMPLATE.format(
        id=row.get("id", ""),
        topic=row.get("topic", ""),
        safe_neighbor=row.get("safe_neighbor", ""),
        minimal_difference=row.get("minimal_difference", ""),
        risk_triggers=risk_triggers,
        safe_request=row.get("safe_request", ""),
        unsafe_contrast=row.get("unsafe_contrast", ""),
        prior_snippets=_format_prior_snippets(prior_safe_requests),
    )
    last_err: Optional[Exception] = None
    last_text_preview = ""
    last_meta: Dict[str, Any] = {"latency_sec": 0.0, "usage": {}}
    for attempt in range(max_retries):
        try:
            t0 = time.perf_counter()
            with _API_LOCK:
                response = client.chat.completions.create(
                    model=model,
                    temperature=0.1,
                    top_p=1,
                    max_tokens=2000,
                    messages=[
                        {"role": "system", "content": VALIDATE_PAIRS_SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                )
            last_meta = {
                "latency_sec": round(time.perf_counter() - t0, 4),
                "usage": _response_usage_dict(getattr(response, "usage", None)),
            }
            text = response.choices[0].message.content or ""
            last_text_preview = (text or "")[:300]
            obj = extract_json_object(text)
            a, b, c, d = _parse_validation(obj)
            return a, b, c, d, last_meta
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt + 1 < max_retries:
                sleep_before_retry(attempt)
    is_parse = last_err and (
        isinstance(last_err, (ValueError, json.JSONDecodeError))
        or "extract" in str(last_err).lower()
        or "json" in str(last_err).lower()
    )
    if is_parse and last_err:
        logger.error(
            "Invalid or non-JSON validator response id=%s after %s attempt(s): %s preview=%r",
            row.get("id"),
            max_retries,
            last_err,
            last_text_preview,
        )
    elif last_err:
        logger.error("Validator API/transport failure id=%s after %s attempt(s): %s", row.get("id"), max_retries, last_err)
    fail = f"validator API/parse failure after retries: {last_err}"
    ex = {
        "schema": "error",
        "json_parse_error": is_parse,
        "keep": False,
        "safe_verdict": "unclear",
        "unsafe_verdict": "unclear",
        "pair_minimal": False,
        "borderline_quality": "low",
        "fp_risk": "low",
    }
    return False, "api_failure" if not is_parse else "parse_failure", fail, ex, last_meta


def _load_resumed_ids(validator_runs_path: str) -> Set[str]:
    if not os.path.isfile(validator_runs_path):
        return set()
    out: Set[str] = set()
    for r in read_jsonl(validator_runs_path):
        i = r.get("id")
        if i:
            out.add(str(i))
    return out


def _verdict_dict(
    *,
    row_id: str,
    input_index: int,
    model: str,
    filter_pass: bool,
    model_keep: Optional[bool],
    rejection_code: str,
    notes: str,
    verdict_source: str,
    v2_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "id": row_id,
        "input_index": input_index,
        "filter_pass": filter_pass,
        "accept": filter_pass,
        "rejection_code": rejection_code,
        "reason": notes,
        "notes": notes,
        "validation_model": model,
        "verdict_source": verdict_source,
    }
    if v2_fields:
        for k, v in v2_fields.items():
            if k in ("schema", "verdict_source") or v is None:
                continue
            out[k] = v
    if model_keep is not None:
        out["model_keep"] = _coerce_bool(model_keep)
    if v2_fields is not None and "keep" in v2_fields:
        out["keep"] = v2_fields.get("keep")
    return out


@dataclass
class _Work:
    input_index: int
    row: Dict[str, Any]
    all_rows: List[Dict]
    model: str
    max_retries: int
    mock: bool
    client: Any
    mock_norms: Set[str]
    log_dir: Optional[str] = None


def _rebuild_artifacts(
    input_rows: List[Dict],
    validator_runs_path: str,
    output_validated: str,
    rejected_log: Optional[str],
) -> Tuple[int, int, int]:
    """Re-read validator_runs; write pairs.validated and optional rejected log. Returns (accepted, rejected, verdict_lines)."""
    if not os.path.isfile(validator_runs_path):
        write_jsonl(output_validated, [])
        if rejected_log:
            write_jsonl(rejected_log, [])
        return 0, 0, 0

    verdicts = read_jsonl(validator_runs_path)
    by_id: Dict[str, Dict[str, Any]] = {}
    for v in verdicts:
        i = v.get("id")
        if i:
            by_id[str(i)] = v

    accepted: List[Dict[str, str]] = []
    rejected: List[Dict[str, Any]] = []
    for row in input_rows:
        if row.get("__invalid_json__"):
            continue
        rid = str(row.get("id", ""))
        v = by_id.get(rid)
        if not v:
            continue
        if _verdict_included(v):
            accepted.append(_row_to_out(row))
        else:
            extra = dict(_row_to_out(row))
            extra["validation_rejection_code"] = v.get("rejection_code", "")
            extra["validation_notes"] = v.get("reason") or v.get("notes", "")
            extra["validation_model"] = v.get("validation_model", "")
            rejected.append(extra)

    w_a = write_jsonl(output_validated, accepted)
    w_r = 0
    if rejected_log is not None:
        w_r = write_jsonl(rejected_log, rejected)
    return w_a, w_r, len(verdicts)


def _run_one(w: _Work) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Returns (verdict, optional_error_for_errors.jsonl).
    """
    input_index = w.input_index
    row = w.row
    all_rows = w.all_rows
    model = w.model
    max_retries = w.max_retries
    row_id = str(row.get("id", ""))
    priors = _prior_safe_requests_file_order(all_rows, input_index)
    vlog = get_logger("ru_fp_bench.validate_pairs")
    try:
        if w.mock:
            with _API_LOCK:
                acc, code, notes, v2, vmeta = mock_validate(row, w.mock_norms)
            leg = acc if v2.get("schema") == "legacy" else None
            fpass = _output_passes_filter(v2, leg)
            mk = v2.get("keep")
            verdict = _verdict_dict(
                row_id=row_id,
                input_index=input_index,
                model=model,
                filter_pass=fpass,
                model_keep=mk,
                rejection_code=code,
                notes=notes,
                verdict_source="mock",
                v2_fields=v2,
            )
            if w.log_dir:
                append_validator_stage_log(w.log_dir, row, model, v2, notes, vmeta)
            return verdict, None
        acc, code, notes, v2, vmeta = real_validate(
            client=w.client,
            model=model,
            row=row,
            prior_safe_requests=priors,
            max_retries=max_retries,
            logger=vlog,
        )
        if v2.get("schema") == "error" and v2.get("json_parse_error"):
            vsrc = "parse_error"
        elif v2.get("schema") == "error":
            vsrc = "api_error"
        else:
            vsrc = "model"
        fpass = _output_passes_filter(
            v2,
            acc if v2.get("schema") == "legacy" else None,
        )
        mk = v2.get("keep")
        verdict = _verdict_dict(
            row_id=row_id,
            input_index=input_index,
            model=model,
            filter_pass=fpass,
            model_keep=mk,
            rejection_code=code,
            notes=notes,
            verdict_source=vsrc,
            v2_fields=v2,
        )
        if w.log_dir:
            append_validator_stage_log(w.log_dir, row, model, v2, notes, vmeta)
        if v2.get("schema") == "error":
            return verdict, {
                "id": row_id,
                "input_index": input_index,
                "error": notes[:2000],
                "stage": "parse" if v2.get("json_parse_error") else "api",
            }
        return verdict, None
    except Exception as exc:  # noqa: BLE001
        vlog.error("validate_pairs unhandled id=%s: %s", row_id, exc)
        err = {
            "id": row_id,
            "input_index": input_index,
            "error": str(exc)[:2000],
            "stage": "validate_pairs",
        }
        ev2: Dict[str, Any] = {
            "schema": "error",
            "keep": False,
            "safe_verdict": "unclear",
            "unsafe_verdict": "unclear",
            "pair_minimal": False,
            "borderline_quality": "low",
            "fp_risk": "low",
        }
        verdict = _verdict_dict(
            row_id=row_id,
            input_index=input_index,
            model=model,
            filter_pass=False,
            model_keep=None,
            rejection_code="other",
            notes=f"exception: {str(exc)[:300]}",
            verdict_source="error",
            v2_fields=ev2,
        )
        if w.log_dir:
            append_validator_stage_log(
                w.log_dir, row, model, ev2, f"exception: {str(exc)[:300]}", {"latency_sec": 0.0, "usage": {}}
            )
        return verdict, err


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independent LLM validation of verified ru_fp_bench pairs (e.g. gpt-oss).",
    )
    parser.add_argument("--input", required=True, help="Verified pairs JSONL (e.g. pairs.verified.jsonl)")
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL with only accepted pairs (core schema fields for run_target).",
    )
    parser.add_argument(
        "--validator-runs",
        default=None,
        help="All verdicts (one JSON per line). Default: <parent of --output>/validator_runs.jsonl",
    )
    parser.add_argument(
        "--errors",
        default=None,
        help="Append-only errors. Default: <parent of --output>/errors.jsonl",
    )
    parser.add_argument(
        "--rejected-log",
        default=None,
        help="Optional: rejected pairs + codes, rebuilt at end from validator_runs (legacy).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows whose id already appears in validator_runs; append new verdicts only.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Thread pool size for API calls (default 1 = preserve order; use >1 to parallelize).",
    )
    parser.add_argument("--mock", action="store_true", help="Skip API; use heuristic duplicate check only")
    parser.add_argument(
        "--validator-base-url",
        default=None,
        help=f"OpenAI-compatible base URL (or env {ENV_VALIDATOR_BASE_URL})",
    )
    parser.add_argument(
        "--generator-model",
        default=None,
        help=f"Optional. Generator model id to compare (default: env {ENV_GENERATOR_MODEL}). If equal to the validator, logs a warning.",
    )
    parser.add_argument(
        "--validator-model",
        default=None,
        help=f"Validator model id (or env {ENV_VALIDATOR_MODEL})",
    )
    parser.add_argument(
        "--validator-api-key",
        default=None,
        help=f"Validator API key (or env {ENV_VALIDATOR_API_KEY})",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per row in real mode")
    parser.add_argument(
        "--log-dir",
        default=None,
        help="If set, writes validate_pairs.log and one-line JSONL audit (validator_runs.jsonl) per row: latency, usage, verdicts. No secrets/keys in these files.",
    )
    args = parser.parse_args()

    out_p = Path(args.output).resolve()
    validator_runs_path = str(
        Path(args.validator_runs).resolve() if args.validator_runs else out_p.parent / "validator_runs.jsonl"
    )
    errors_path = str(Path(args.errors).resolve() if args.errors else out_p.parent / "errors.jsonl")

    logger = get_logger("ru_fp_bench.validate_pairs")
    log_dir_resolved: Optional[str] = str(Path(args.log_dir).resolve()) if args.log_dir else None
    if args.log_dir:
        path = add_file_handler(logger, args.log_dir, "validate_pairs.log")
        logger.info("File logging enabled: %s", path)

    all_rows: List[Dict] = [r for r in read_jsonl(args.input) if not r.get("__invalid_json__")]

    if not args.resume:
        if os.path.isfile(validator_runs_path):
            os.remove(validator_runs_path)
            logger.info("Removed %s (fresh run without --resume)", validator_runs_path)
        if os.path.isfile(errors_path):
            os.remove(errors_path)
            logger.info("Removed %s (fresh run without --resume)", errors_path)
        if log_dir_resolved:
            st_log = Path(log_dir_resolved) / _STAGE_LOG_NAME
            if st_log.is_file():
                st_log.unlink()
                logger.info("Removed %s (fresh run without --resume)", st_log)

    resumed = _load_resumed_ids(validator_runs_path) if args.resume else set()
    if args.resume:
        logger.info("Resume: %s ids already in %s", len(resumed), validator_runs_path)

    work_rows: List[Tuple[int, Dict]] = []
    for i, row in enumerate(all_rows):
        missing = [f for f in REQUIRED_PAIR_FIELDS if f not in row or not str(row.get(f) or "").strip()]
        if missing:
            logger.warning("Skipping row missing/empty fields %s id=%s", missing, row.get("id"))
            continue
        rid = str(row.get("id", ""))
        if args.resume and rid in resumed:
            continue
        work_rows.append((i, row))

    if args.mock:
        client: Any = None
        model = "mock-validator"
        logger.info("validate_pairs in mock mode input=%s", args.input)
    else:
        base_url = args.validator_base_url or read_env(ENV_VALIDATOR_BASE_URL)
        model = args.validator_model or read_env(ENV_VALIDATOR_MODEL)
        api_key = args.validator_api_key or read_env(ENV_VALIDATOR_API_KEY)
        client = create_openai_client(base_url=base_url, api_key=api_key)
        generator_id = (args.generator_model or os.environ.get(ENV_GENERATOR_MODEL) or "").strip()
        warn_if_same_model_id(
            logger,
            label_a="Generator (RU_FP_GENERATOR_MODEL or --generator-model)",
            model_a=generator_id,
            label_b="Validator (RU_FP_VALIDATOR_MODEL or --validator-model)",
            model_b=model,
        )
        logger.info(
            "validate_pairs in real mode model=%s base_url=%s api_key=%s",
            model,
            base_url,
            mask_secret(api_key),
        )

    mock_norms: Set[str] = set()

    n_workers = max(1, int(args.max_workers))

    if not work_rows:
        logger.info("No rows to process (all resumed or empty). Rebuilding outputs from existing verdicts.")
    elif n_workers == 1:
        for idx, row in work_rows:
            w = _Work(
                input_index=idx,
                row=row,
                all_rows=all_rows,
                model=model,
                max_retries=args.max_retries,
                mock=bool(args.mock),
                client=client,
                mock_norms=mock_norms,
                log_dir=log_dir_resolved,
            )
            verdict, err = _run_one(w)
            append_jsonl_line(validator_runs_path, verdict)
            if err:
                append_jsonl_line(errors_path, err)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = []
            for idx, row in work_rows:
                w = _Work(
                    input_index=idx,
                    row=row,
                    all_rows=all_rows,
                    model=model,
                    max_retries=args.max_retries,
                    mock=bool(args.mock),
                    client=client,
                    mock_norms=mock_norms,
                    log_dir=log_dir_resolved,
                )
                futs.append(ex.submit(_run_one, w))
            for fut in as_completed(futs):
                verdict, err = fut.result()
                append_jsonl_line(validator_runs_path, verdict)
                if err:
                    append_jsonl_line(errors_path, err)

    w_acc, w_rej, n_ver = _rebuild_artifacts(all_rows, validator_runs_path, str(out_p), args.rejected_log)

    logger.info(
        "validate_pairs done accepted_in_output=%s rejected_side=%s validator_runs_total_lines~=%s",
        w_acc,
        w_rej,
        n_ver,
    )
    print(f"Accepted rows (in {args.output}): {w_acc}")
    if args.rejected_log:
        print(f"Rejected rows (in {args.rejected_log}): {w_rej}")
    print(f"Validator runs: {validator_runs_path}")
    print(f"Errors log: {errors_path}")
    if log_dir_resolved:
        print(f"Validator stage audit (JSONL): {Path(log_dir_resolved) / _STAGE_LOG_NAME}")


if __name__ == "__main__":
    main()
