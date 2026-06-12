"""Export bundle (delegates to ``split_builder_exporter.write_stage3_split_outputs`` when used from pipeline)."""

from __future__ import annotations

from typing import List

from ..exporters.benchmark_bundle import write_benchmark_bundle
from ..schemas import Stage3CandidateRecord, SplitAssignmentRecord


def write_exports(
    export_dir: str,
    pool: List[Stage3CandidateRecord],
    splits: List[SplitAssignmentRecord],
    *,
    write_manifest: bool = True,
) -> None:
    """
    Legacy helper: JSONL-only bundle under ``export_dir`` (no CSV / manifest JSON).

    Full Node 5 output is produced by ``write_stage3_split_outputs`` in ``split_builder_exporter``.
    """
    write_benchmark_bundle(export_dir, pool, splits, write_manifest=write_manifest)
