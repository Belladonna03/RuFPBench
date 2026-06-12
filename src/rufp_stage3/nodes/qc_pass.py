"""Rule-based QC labels for candidate pool rows."""

from __future__ import annotations

import re
from typing import List

from ..schemas import QCLabelRecord, Stage3CandidateRecord


def run_qc(rows: List[Stage3CandidateRecord]) -> List[QCLabelRecord]:
    out: List[QCLabelRecord] = []
    for r in rows:
        flags: List[str] = []
        text = (r.text or "").strip()
        if len(text) < 8:
            flags.append("too_short")
        if not text:
            flags.append("empty_text")
        if re.search(r"\bignore previous\b", text, re.I):
            flags.append("possible_injection")
        if r.source == "review_queue":
            flags.append("from_review_queue")
        blocking = [f for f in flags if f != "from_review_queue"]
        qc_pass = len(blocking) == 0
        out.append(
            QCLabelRecord(
                item_id=r.item_id,
                qc_pass=qc_pass,
                qc_flags=flags,
                notes=",".join(flags) if flags else "ok",
            )
        )
    return out
