"""Write release-oriented artifacts under ``stage3_exports/``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..config import ensure_stage3_run_dir
from ..io import file_nonempty, load_jsonl, save_jsonl
from ..policy.loader import load_stage3_policy
from ..policy.schema import Stage3Policy
from ..schemas import (
    FinalLineageRecord,
    HardSubsetRecord,
    QCLabelRecord,
    SplitAssignmentRecord,
    Stage3CandidateRecord,
)
from .checklist import build_release_checklist_markdown
from .dataset_card import build_dataset_card_markdown
from .lineage import build_final_lineage_records


def _index_qc(rows: List[QCLabelRecord]) -> Dict[str, QCLabelRecord]:
    return {q.item_id: q for q in rows}


def _index_split(rows: List[SplitAssignmentRecord]) -> Dict[str, SplitAssignmentRecord]:
    return {s.item_id: s for s in rows}


def _hard_id_set(hard: List[HardSubsetRecord]) -> Set[str]:
    return {h.item_id for h in hard}


def _load_export_meta(export_dir: Path) -> Dict[str, Any]:
    p = export_dir / "metadata.json"
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def write_stage3_release_docs(
    *,
    run_id: str,
    run_dir: str | Path,
    export_dir: str | Path,
    balanced_pool: List[Stage3CandidateRecord],
    splits: List[SplitAssignmentRecord],
    hard_rows: List[HardSubsetRecord],
    policy: Stage3Policy,
    qc_labels: Optional[List[QCLabelRecord]] = None,
) -> Dict[str, str]:
    """
    Writes ``final_lineage.jsonl``, ``dataset_card.md``, ``release_checklist.md`` under ``export_dir``.

    If ``qc_labels`` is None, loads ``<run_dir>/stage3_qc_labels.jsonl`` when present.
    """
    rdir = Path(run_dir)
    exp = Path(export_dir)
    exp.mkdir(parents=True, exist_ok=True)

    qc_list = qc_labels
    if qc_list is None:
        qpath = rdir / "stage3_qc_labels.jsonl"
        if not qpath.is_file():
            raise FileNotFoundError(
                f"QC labels required for lineage: missing {qpath} (pass qc_labels= or run Node 2)"
            )
        qc_list = load_jsonl(str(qpath), QCLabelRecord)

    qc_by = _index_qc(qc_list)
    sp_by = _index_split(splits)
    hard_ids = _hard_id_set(hard_rows)

    lineage_rows: List[FinalLineageRecord] = build_final_lineage_records(
        stage3_run_id=run_id,
        balanced_pool=balanced_pool,
        qc_by_item=qc_by,
        split_by_item=sp_by,
        hard_ids=hard_ids,
    )
    lineage_path = exp / "final_lineage.jsonl"
    save_jsonl(str(lineage_path), lineage_rows)

    meta = _load_export_meta(exp)
    if not meta.get("counts"):
        meta = {
            **meta,
            "counts": {
                "balanced_total": len(balanced_pool),
                "dev": sum(1 for s in splits if s.split == "dev"),
                "test": sum(1 for s in splits if s.split == "test"),
                "review_holdout": sum(1 for s in splits if s.split == "review_holdout"),
                "hard_subset_export_rows": len(hard_ids),
            },
        }

    card_md = build_dataset_card_markdown(
        run_id=run_id,
        policy=policy,
        balanced_pool=balanced_pool,
        splits=splits,
        hard_rows=hard_rows,
        export_meta=meta,
    )
    card_path = exp / "dataset_card.md"
    card_path.write_text(card_md, encoding="utf-8")

    checklist_md = build_release_checklist_markdown(
        run_id=run_id,
        exports_dir=exp.resolve(),
        extra={},
    )
    chk_path = exp / "release_checklist.md"
    chk_path.write_text(checklist_md, encoding="utf-8")

    return {
        "final_lineage": str(lineage_path),
        "dataset_card": str(card_path),
        "release_checklist": str(chk_path),
    }


def prepare_stage3_release_from_disk(
    run_id: str,
    *,
    config_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    Load artifacts from ``artifacts/stage3/<run_id>/`` and write release docs.

    Requires: ``stage3_balanced_pool.jsonl``, ``split_assignment.jsonl`` (or ``stage3_splits.jsonl``),
    ``stage3_qc_labels.jsonl``, ``stage3_hard_subset.jsonl`` (may be empty).
    """
    base = ensure_stage3_run_dir(run_id)
    policy = load_stage3_policy(config_path)

    bal_path = base / "stage3_balanced_pool.jsonl"
    if not file_nonempty(str(bal_path)):
        raise FileNotFoundError(f"Missing or empty balanced pool: {bal_path}")

    sp_path = base / "split_assignment.jsonl"
    if not file_nonempty(str(sp_path)):
        alt = base / "stage3_splits.jsonl"
        if file_nonempty(str(alt)):
            sp_path = alt
        else:
            raise FileNotFoundError(f"Missing split assignment: {sp_path}")

    balanced = load_jsonl(str(bal_path), Stage3CandidateRecord)
    splits = load_jsonl(str(sp_path), SplitAssignmentRecord)

    hpath = base / "stage3_hard_subset.jsonl"
    hard: List[HardSubsetRecord] = (
        load_jsonl(str(hpath), HardSubsetRecord) if file_nonempty(str(hpath)) else []
    )

    exports = base / "stage3_exports"
    return write_stage3_release_docs(
        run_id=run_id,
        run_dir=base,
        export_dir=exports,
        balanced_pool=balanced,
        splits=splits,
        hard_rows=hard,
        policy=policy,
        qc_labels=None,
    )
