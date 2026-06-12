"""Merge Stage 2 + Stage 2.5 artifacts into ``stage3_candidate_pool.jsonl`` (delegates to candidate_merger)."""

from __future__ import annotations

from typing import List

from ..policy.schema import Stage3Policy
from ..schemas import Stage3CandidateRecord
from .candidate_merger import merge_candidate_pool


def build_candidate_pool(policy: Stage3Policy, *, dry_run: bool) -> List[Stage3CandidateRecord]:
    return merge_candidate_pool(policy, dry_run=dry_run)
