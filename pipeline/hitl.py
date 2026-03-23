from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.io import read_table


def should_stop_for_hitl(cfg: dict[str, Any], root: Path) -> tuple[bool, str]:
    """Return (stop, reason) if pipeline must pause for human review."""
    hitl = cfg.get("hitl") or {}
    mode = str(hitl.get("human_mode", ""))
    if mode != "stop_if_missing":
        return False, ""

    rq = Path(hitl.get("review_queue_path", root / "data/labeled/review_queue.csv"))
    if not rq.exists():
        return False, ""

    try:
        q = pd.read_csv(rq)
    except Exception as exc:
        return False, f"Could not read review queue: {exc}"

    if len(q) == 0:
        return False, ""

    corrected = Path(hitl.get("corrected_queue_path", root / "data/labeled/review_queue_corrected.csv"))
    if not corrected.exists():
        return (
            True,
            f"HITL: {rq} has {len(q)} row(s); create {corrected} with reviewed labels then re-run.",
        )

    try:
        c = pd.read_csv(corrected)
    except Exception as exc:
        return True, f"HITL: could not read {corrected}: {exc}"

    if "uid" not in c.columns:
        return True, f"HITL: {corrected} must include a uid column."

    covered = set(c["uid"].astype(str))
    pending = [str(u) for u in q["uid"].astype(str) if str(u) not in covered]
    if pending:
        return (
            True,
            f"HITL: {len(pending)} uid(s) from review queue are missing in corrected file.",
        )

    return False, ""


def merge_hitl_labels(
    labeled_df: pd.DataFrame,
    cfg: dict[str, Any],
    root: Path,
) -> pd.DataFrame:
    """Apply human corrections from review_queue_corrected.csv when present."""
    hitl = cfg.get("hitl") or {}
    path = Path(hitl.get("corrected_queue_path", root / "data/labeled/review_queue_corrected.csv"))
    if not path.exists():
        out = labeled_df.copy()
        if "final_label" not in out.columns and "label" in out.columns:
            out["final_label"] = out["label"]
        return out

    corr = read_table(path)
    if "uid" not in corr.columns:
        raise ValueError("Corrected queue must contain uid")
    label_col = "final_label" if "final_label" in corr.columns else "label"
    if label_col not in corr.columns:
        raise ValueError("Corrected queue must contain final_label or label")

    cmap = dict(zip(corr["uid"].astype(str), corr[label_col].astype(str)))
    out = labeled_df.copy()
    if "final_label" not in out.columns:
        out["final_label"] = out["label"].astype(str) if "label" in out.columns else None

    def _apply(row: pd.Series) -> str:
        uid = str(row.get("uid", ""))
        if uid in cmap:
            return str(cmap[uid])
        fl = row.get("final_label")
        if fl is not None and str(fl).strip():
            return str(fl)
        return str(row.get("label", ""))

    out["final_label"] = out.apply(_apply, axis=1)
    return out
