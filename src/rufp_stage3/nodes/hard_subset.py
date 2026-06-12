"""Hard subset selection (delegates to ``hard_subset_selector``)."""

from __future__ import annotations

from typing import List

from ..policy.schema import Stage3Policy
from ..schemas import HardSubsetRecord, Stage3CandidateRecord
from .hard_subset_selector import run_hard_subset as _run_hard_subset_impl


def run_hard_subset(rows: List[Stage3CandidateRecord], policy: Stage3Policy) -> List[HardSubsetRecord]:
    return _run_hard_subset_impl(rows, policy)
