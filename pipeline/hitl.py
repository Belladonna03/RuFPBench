from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.io import read_table

DECIDED_REVIEW_STATUSES = {"accepted_auto", "corrected", "skipped"}


def _review_id_col(df: pd.DataFrame) -> str | None:
    for col in ("sample_id", "uid"):
        if col in df.columns:
            return col
    return None


def should_stop_for_hitl(cfg: dict[str, Any], root: Path) -> tuple[bool, str]:
    """Return (stop, reason) if pipeline must pause for human review."""
    hitl = cfg.get("hitl") or {}
    mode = str(hitl.get("human_mode", ""))
    if mode != "stop_if_missing":
        return False, ""

    rq = Path(hitl.get("review_queue_path", root / "data/labeled/review_queue.jsonl"))
    if not rq.exists():
        return False, ""

    try:
        q = read_table(rq)
    except Exception as exc:
        return False, f"Could not read review queue: {exc}"

    if len(q) == 0:
        return False, ""

    corrected = Path(hitl.get("corrected_queue_path", root / "data/labeled/review_results.jsonl"))
    if not corrected.exists():
        return (
            True,
            (
                f"HITL: {rq} has {len(q)} row(s); run console review and write decisions to {corrected}, "
                "then re-run."
            ),
        )

    try:
        c = read_table(corrected)
    except Exception as exc:
        return True, f"HITL: could not read {corrected}: {exc}"

    qid = _review_id_col(q)
    cid = _review_id_col(c)
    if qid is None:
        return True, f"HITL: {rq} must include sample_id or uid."
    if cid is None:
        return True, f"HITL: {corrected} must include sample_id or uid."

    status_col = "review_status" if "review_status" in c.columns else None
    if status_col:
        decided_df = c[c[status_col].astype(str).isin(DECIDED_REVIEW_STATUSES)]
    else:
        decided_df = c
    covered = set(decided_df[cid].astype(str))
    pending = [str(u) for u in q[qid].astype(str) if str(u) not in covered]
    if pending:
        return (
            True,
            f"HITL: {len(pending)} item(s) from review queue are still pending in {corrected}.",
        )

    return False, ""


def merge_hitl_labels(
    labeled_df: pd.DataFrame,
    cfg: dict[str, Any],
    root: Path,
) -> pd.DataFrame:
    """Apply human corrections from review_results.jsonl when present."""
    hitl = cfg.get("hitl") or {}
    path = Path(hitl.get("corrected_queue_path", root / "data/labeled/review_results.jsonl"))
    if not path.exists():
        out = labeled_df.copy()
        if "final_label" not in out.columns:
            if "predicted_label" in out.columns:
                out["final_label"] = out["predicted_label"]
            elif "label" in out.columns:
                out["final_label"] = out["label"]
        return out

    from agents.annotation_agent import AnnotationAgent

    return AnnotationAgent(modality="text", config=cfg).merge_review_decisions(labeled_df, path)
