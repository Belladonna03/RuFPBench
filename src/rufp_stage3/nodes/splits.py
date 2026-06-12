"""Split assignment (delegates to ``split_builder_exporter``)."""

from __future__ import annotations

from typing import List, Optional

from ..policy.schema import Stage3Policy
from ..schemas import DedupClusterRecord, SplitAssignmentRecord, Stage3CandidateRecord
from .split_builder_exporter import run_splits as _run_splits


def run_splits(
    rows: List[Stage3CandidateRecord],
    policy: Stage3Policy,
    dedup_clusters: Optional[List[DedupClusterRecord]] = None,
) -> List[SplitAssignmentRecord]:
    return _run_splits(rows, policy, dedup_clusters)
