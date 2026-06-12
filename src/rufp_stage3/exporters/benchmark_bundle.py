"""Write ``stage3_exports/`` and optional ``stage3_final_export.jsonl`` manifest rows."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

from ..io import save_jsonl
from ..schemas import FinalExportRecord, Stage3CandidateRecord, SplitAssignmentRecord


def write_benchmark_bundle(
    export_dir: str,
    pool: List[Stage3CandidateRecord],
    splits: List[SplitAssignmentRecord],
    *,
    write_manifest: bool = True,
) -> List[FinalExportRecord]:
    """
    Emit ``benchmark_<split>.jsonl`` copies per split and optional ``final_export.jsonl``
    (``FinalExportRecord`` rows).
    """
    Path(export_dir).mkdir(parents=True, exist_ok=True)
    by_id = {r.item_id: r for r in pool}
    manifest: List[FinalExportRecord] = []

    for split_name in ("dev", "test", "review_holdout"):
        ids = {s.item_id for s in splits if s.split == split_name}
        rows = [by_id[i] for i in ids if i in by_id]
        out_path = Path(export_dir) / f"benchmark_{split_name}.jsonl"
        save_jsonl(str(out_path), rows)
        for r in rows:
            h = hashlib.sha256(r.text.encode("utf-8")).hexdigest()
            manifest.append(
                FinalExportRecord(
                    item_id=r.item_id,
                    split=split_name,
                    export_relpath=str(out_path.name),
                    sha256_text=h,
                )
            )

    readme = Path(export_dir) / "README.txt"
    readme.write_text(
        "Stage 3 export: benchmark_*.jsonl (Stage3CandidateRecord). "
        "See final_export.jsonl for FinalExportRecord pointers.\n",
        encoding="utf-8",
    )

    if write_manifest:
        save_jsonl(str(Path(export_dir) / "final_export.jsonl"), manifest)

    return manifest
