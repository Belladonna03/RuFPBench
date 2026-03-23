"""Post-merge downsampling for collection: stratify by ``source`` with proportional budget."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from shared.logging_utils import get_logger

_log = get_logger("shared.collection_merge_sample")


def stratified_sample_by_source(
    df: pd.DataFrame,
    target_n: int,
    *,
    source_col: str = "source",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Reduce ``df`` to at most ``target_n`` rows while keeping all sources represented when possible.

    - If ``df`` is empty or ``len(df) <= target_n``, returns a copy with ``reset_index(drop=True)``.
    - If ``source_col`` is missing, uses a single ``sample(n=target_n, random_state=...)``.
    - If ``target_n < number of distinct sources``, logs a warning and takes one row each from
      the first ``target_n`` sources in sorted order (deterministic).
    - Otherwise: at least one row per source (if ``target_n`` allows), remainder split proportionally
      to source sizes (largest-remainder), then rows sampled per source with a shared RNG.

    Does not mutate the input dataframe.
    """
    out = df.reset_index(drop=True)
    if out.empty or target_n <= 0:
        return out
    if len(out) <= target_n:
        return out

    rng = np.random.RandomState(random_state)

    if source_col not in out.columns:
        _log.info(
            "merged sample: column %r missing; using uniform random sample (no stratification)",
            source_col,
        )
        return out.sample(n=target_n, random_state=rng).reset_index(drop=True)

    counts = out.groupby(source_col, dropna=False).size()
    sources = sorted(list(counts.index), key=str)
    S = len(sources)
    if S == 0:
        return out.sample(n=min(target_n, len(out)), random_state=rng).reset_index(drop=True)

    if target_n < S:
        _log.warning(
            "merged sample: target_n=%s < unique_sources=%s; taking 1 row from first %s sources "
            "(alphabetically sorted source ids)",
            target_n,
            S,
            target_n,
        )
        chosen = sources[:target_n]
        parts: list[pd.DataFrame] = []
        for s in chosen:
            sub = out[out[source_col] == s]
            if sub.empty:
                continue
            parts.append(sub.sample(n=1, random_state=rng))
        if not parts:
            return out.iloc[:0].copy()
        return pd.concat(parts, ignore_index=True)

    # target_n >= S: min 1 per source, rest proportional (largest remainder on fractional parts)
    names = sources  # already sorted by str
    c = np.array([int(counts[n]) for n in names], dtype=np.int64)
    total = int(c.sum())
    R = target_n - S  # extra rows after reserving 1 per source
    max_extra = c - 1

    raw_extra = (R * c / float(total)) if total > 0 else np.zeros(S, dtype=float)
    extra = np.floor(raw_extra).astype(np.int64)
    extra = np.minimum(extra, max_extra)
    rem = R - int(extra.sum())
    frac = raw_extra - np.floor(raw_extra)
    order = list(np.argsort(-frac))  # largest fractional parts first

    while rem > 0:
        progressed = False
        for i in order:
            if rem <= 0:
                break
            if extra[i] < max_extra[i]:
                extra[i] += 1
                rem -= 1
                progressed = True
        if not progressed:
            _log.warning(
                "merged sample: could not place %s extra rows (slack exhausted); output may be shorter than target_n",
                rem,
            )
            break

    n_take = np.minimum((1 + extra).astype(np.int64), c)

    parts = []
    for i, s in enumerate(names):
        k = int(min(n_take[i], c[i]))
        if k <= 0:
            continue
        sub = out[out[source_col] == s]
        if len(sub) <= k:
            parts.append(sub)
        else:
            parts.append(sub.sample(n=k, random_state=rng))
    if not parts:
        return out.iloc[:0].copy()
    return pd.concat(parts, ignore_index=True)


# Alias for callers that prefer a dataset-oriented name
sample_merged_dataset = stratified_sample_by_source


__all__ = ["stratified_sample_by_source", "sample_merged_dataset"]
