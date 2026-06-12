"""Balanced pool + strata metadata (delegates to ``balancer_slice_builder``)."""

from __future__ import annotations

from typing import List, Tuple

from ..policy.schema import Stage3Policy
from ..schemas import BalancedRecord, Stage3CandidateRecord
from .balancer_slice_builder import run_balance as _run_balance


def run_balance(rows: List[Stage3CandidateRecord], policy: Stage3Policy) -> Tuple[List[Stage3CandidateRecord], List[BalancedRecord]]:
    return _run_balance(rows, policy)
