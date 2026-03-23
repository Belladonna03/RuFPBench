from __future__ import annotations

from typing import Any

import pandas as pd

from shared.logging_utils import get_logger

_log = get_logger("agents.data_quality")


class DataQualityAgent:
    """Detect duplicate / missing / length-outlier issues and apply cleanup strategies."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def detect_issues(self, df: pd.DataFrame) -> dict[str, Any]:
        _log.info("detect_issues input_rows=%s", len(df))
        dup_subset = self.config.get("duplicate_subset") or ["text"]
        cols = [c for c in dup_subset if c in df.columns]
        dup_mask = df.duplicated(subset=cols, keep=False) if cols else pd.Series(False, index=df.index)
        n_dup = int(dup_mask.sum())

        text = df["text"] if "text" in df.columns else pd.Series([""] * len(df))
        missing_text = int((text.isna() | (text.astype(str).str.strip() == "")).sum())

        lengths = text.fillna("").astype(str).str.len().astype(float)
        k = float(self.config.get("outlier_k", 1.5))
        q1, q3 = lengths.quantile(0.25), lengths.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - k * iqr, q3 + k * iqr
        outlier_mask = (lengths > 0) & ((lengths < low) | (lengths > high))
        n_out = int(outlier_mask.sum())

        summary = {
            "duplicates": {"count": n_dup, "subset": cols},
            "missing_text": {"count": missing_text},
            "length_outliers": {"count": n_out, "iqr": [float(low), float(high)]},
            "rows": len(df),
        }
        _log.info(
            "issues duplicates=%s missing_text=%s length_outliers=%s",
            n_dup,
            missing_text,
            n_out,
        )
        return summary

    def fix(self, df: pd.DataFrame, strategy: dict[str, str] | None = None) -> pd.DataFrame:
        strategy = strategy or (self.config.get("strategy") or {})
        n0 = len(df)
        _log.info("fix start rows=%s strategy=%s", n0, strategy)
        out = df.copy()

        if strategy.get("missing") == "fill" and "text" in out.columns:
            out["text"] = out["text"].fillna("")

        dup_subset = self.config.get("duplicate_subset") or ["text"]
        cols = [c for c in dup_subset if c in out.columns]
        if strategy.get("duplicates") == "drop" and cols:
            out = out.drop_duplicates(subset=cols, keep="first")

        if strategy.get("outliers") == "drop_iqr" and "text" in out.columns:
            lengths = out["text"].fillna("").astype(str).str.len().astype(float)
            k = float(self.config.get("outlier_k", 1.5))
            q1, q3 = lengths.quantile(0.25), lengths.quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - k * iqr, q3 + k * iqr
            keep = ~((lengths > 0) & ((lengths < low) | (lengths > high)))
            out = out.loc[keep].reset_index(drop=True)

        _log.info("fix done rows=%s (removed %s)", len(out), n0 - len(out))
        return out

    def compare(self, before: pd.DataFrame, after: pd.DataFrame) -> dict[str, Any]:
        return {
            "rows_before": len(before),
            "rows_after": len(after),
            "delta": len(before) - len(after),
            "table": [
                {"metric": "rows", "before": len(before), "after": len(after)},
            ],
        }


__all__ = ["DataQualityAgent"]
