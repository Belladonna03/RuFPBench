from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Report dataclasses
# ----------------------------

@dataclass
class OutlierSummary:
    feature: str
    method: str
    lower: float
    upper: float
    n_outliers: int
    outlier_rate: float
    example_indices: List[int]


@dataclass
class QualityReport:
    missing: Dict[str, Dict[str, Any]]
    duplicates: int
    outliers: List[Dict[str, Any]]
    imbalance: Dict[str, Any]
    meta: Dict[str, Any]


@dataclass
class ComparisonReport:
    """Tabular 'before vs after' quality metrics."""
    table: pd.DataFrame
    meta: Dict[str, Any]


# ----------------------------
# Agent
# ----------------------------

class DataQualityAgent:
    """
    DataQualityAgent ("Data Detective") for text datasets.

    Contract:
        agent = DataQualityAgent()
        report = agent.detect_issues(df)
        df_clean = agent.fix(df, strategy={...})
        comparison = agent.compare(df, df_clean)

    Notes for RuFPBench-MVP:
      - This agent is intentionally generic: it works for any text dataset following
        the minimal schema [text, label, source, collected_at].
      - For outliers, we detect anomalies on derived numeric features (text length in chars/words),
        plus any existing numeric columns in df.
    """

    REQUIRED_COLUMNS = ["text", "label", "source", "collected_at"]

    def __init__(
        self,
        *,
        text_col: str = "text",
        label_col: str = "label",
        duplicate_subset: Optional[List[str]] = None,
        outlier_method: str = "iqr",
        outlier_k: float = 1.5,
        imbalance_warn_threshold: float = 0.1,
        example_k: int = 5,
        random_state: int = 42,
    ) -> None:
        self.text_col = text_col
        self.label_col = label_col
        self.duplicate_subset = duplicate_subset or [text_col]
        self.outlier_method = outlier_method
        self.outlier_k = outlier_k
        self.imbalance_warn_threshold = imbalance_warn_threshold
        self.example_k = example_k
        self.random_state = random_state

    # ----------------------------
    # Public API
    # ----------------------------

    def detect_issues(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect missing values, duplicates, outliers, and class imbalance.

        Returns a JSON-serializable dict (QualityReport-like), per assignment.
        """
        self._validate_schema(df)

        df_ = df.copy()

        # Normalize basic text artifacts early for detection
        df_[self.text_col] = df_[self.text_col].astype("string")
        df_["_text_norm"] = df_[self.text_col].fillna("").str.strip()

        missing_report = self._detect_missing(df_)
        duplicates_report = self._detect_duplicates(df_)
        duplicates_count = int(duplicates_report.get('count', 0))
        outliers_report = self._detect_outliers(df_)
        imbalance_report = self._detect_imbalance(df_)

        report = QualityReport(
            missing=missing_report,
            duplicates=duplicates_count,
            outliers=outliers_report,
            imbalance=imbalance_report,
            meta={
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "n_rows": int(len(df_)),
                "required_columns": self.REQUIRED_COLUMNS,
                "text_col": self.text_col,
                "label_col": self.label_col,
                "duplicate_subset": self.duplicate_subset,
                "duplicates_details": duplicates_report,
                "outlier_method": self.outlier_method,
                "outlier_k": self.outlier_k,
            },
        )
        return self._to_jsonable(report)

    def fix(self, df: pd.DataFrame, strategy: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply data cleaning strategies.

        strategy example:
            {
              "missing": "fill",           # fill | drop | median | mode | constant:<value>
              "duplicates": "drop",        # drop | keep_longest | none
              "outliers": "drop_iqr",      # drop_iqr | clip_iqr | none
              "imbalance": "none"          # none | downsample | upsample
            }
        """
        self._validate_schema(df)

        df_clean = df.copy()

        # Ensure stable types
        df_clean[self.text_col] = df_clean[self.text_col].astype("string")
        df_clean[self.label_col] = df_clean[self.label_col].astype("string")

        # 1) Missing
        missing_strategy = (strategy.get("missing") or "fill").lower()
        df_clean = self._fix_missing(df_clean, missing_strategy, strategy.get("missing_fill_values"))

        # 2) Duplicates
        dup_strategy = (strategy.get("duplicates") or "drop").lower()
        df_clean = self._fix_duplicates(df_clean, dup_strategy)

        # 3) Outliers
        out_strategy = (strategy.get("outliers") or "none").lower()
        df_clean = self._fix_outliers(df_clean, out_strategy)

        # 4) Imbalance (optional)
        imb_strategy = (strategy.get("imbalance") or "none").lower()
        df_clean = self._fix_imbalance(df_clean, imb_strategy, target_col=self.label_col)

        # Final: strip empties & keep only meaningful rows
        df_clean[self.text_col] = df_clean[self.text_col].fillna("").str.strip()
        df_clean = df_clean[df_clean[self.text_col] != ""].reset_index(drop=True)

        return df_clean

    def compare(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict[str, Any]:
        """
        Return a table 'before vs after' for each quality metric.

        Returns a JSON-serializable dict with an embedded table (as records).
        """
        self._validate_schema(df_before)
        self._validate_schema(df_after)

        before = self._compute_quality_metrics(df_before)
        after = self._compute_quality_metrics(df_after)

        metrics = sorted(set(before.keys()) | set(after.keys()))
        rows = []
        for m in metrics:
            b = before.get(m, np.nan)
            a = after.get(m, np.nan)
            delta = a - b if self._is_number(b) and self._is_number(a) else None
            rows.append({"metric": m, "before": b, "after": a, "delta": delta})

        table = pd.DataFrame(rows).sort_values("metric").reset_index(drop=True)

        report = ComparisonReport(
            table=table,
            meta={
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "n_rows_before": int(len(df_before)),
                "n_rows_after": int(len(df_after)),
            },
        )
        return {
            "table": table.to_dict(orient="records"),
            "meta": report.meta,
        }

    # ----------------------------
    # Missing values
    # ----------------------------

    def _detect_missing(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        missing = {}
        n = len(df) if len(df) else 1

        for col in df.columns:
            is_missing = df[col].isna()
            # For text column, treat empty/whitespace as missing-like (quality issue)
            if col == self.text_col:
                is_missing = is_missing | (df["_text_norm"] == "")
            cnt = int(is_missing.sum())
            if cnt > 0:
                missing[col] = {
                    "count": cnt,
                    "rate": float(cnt / n),
                }

        # Required columns summary
        required_missing_rows = self._rows_missing_required(df)
        missing["_required_rows"] = {
            "count": int(required_missing_rows.sum()),
            "rate": float(required_missing_rows.mean()) if len(df) else 0.0,
            "required_columns": self.REQUIRED_COLUMNS,
        }
        return missing

    def _fix_missing(
        self,
        df: pd.DataFrame,
        missing_strategy: str,
        fill_values: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        fill_values = fill_values or {}

        # Drop rows with missing required columns (strong, safe default)
        if missing_strategy in {"drop", "drop_rows"}:
            mask_bad = self._rows_missing_required(df)
            df = df[~mask_bad].copy()

        # Fill: a generic approach suitable for text datasets
        if missing_strategy in {"fill", "median", "mode"} or missing_strategy.startswith("constant:"):
            if missing_strategy.startswith("constant:"):
                const_val = missing_strategy.split("constant:", 1)[1]
            else:
                const_val = None

            for col in df.columns:
                if col in {"_text_norm"}:
                    continue

                if col in fill_values:
                    df[col] = df[col].fillna(fill_values[col])
                    continue

                if pd.api.types.is_numeric_dtype(df[col]):
                    if missing_strategy in {"median", "fill"}:
                        val = df[col].median()
                    elif missing_strategy == "mode":
                        val = df[col].mode(dropna=True)
                        val = val.iloc[0] if len(val) else df[col].median()
                    else:
                        val = const_val
                    df[col] = df[col].fillna(val)
                else:
                    # categorical / strings
                    if missing_strategy == "mode":
                        mode = df[col].mode(dropna=True)
                        val = mode.iloc[0] if len(mode) else (const_val if const_val is not None else "unknown")
                    else:
                        val = const_val if const_val is not None else "unknown"
                    df[col] = df[col].fillna(val)

            # Also handle empty strings in text
            if self.text_col in df.columns:
                df[self.text_col] = df[self.text_col].astype("string").fillna("").str.strip()
                df = df[df[self.text_col] != ""].copy()

        return df.reset_index(drop=True)

    # ----------------------------
    # Duplicates
    # ----------------------------

    def _detect_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        subset = [c for c in self.duplicate_subset if c in df.columns]
        if not subset:
            subset = [self.text_col]

        # normalize subset if it includes text
        df_tmp = df.copy()
        if self.text_col in subset:
            df_tmp[self.text_col] = df_tmp[self.text_col].astype("string").fillna("").str.strip()

        dup_mask = df_tmp.duplicated(subset=subset, keep="first")
        dup_count = int(dup_mask.sum())
        dup_rate = float(dup_mask.mean()) if len(df_tmp) else 0.0

        examples = []
        if dup_count > 0:
            example_idx = df_tmp.index[dup_mask].tolist()[: self.example_k]
            examples = [{"index": int(i), "text": str(df_tmp.loc[i, self.text_col])[:200]} for i in example_idx]

        return {
            "count": dup_count,
            "rate": dup_rate,
            "subset": subset,
            "examples": examples,
        }

    def _fix_duplicates(self, df: pd.DataFrame, dup_strategy: str) -> pd.DataFrame:
        subset = [c for c in self.duplicate_subset if c in df.columns]
        if not subset:
            subset = [self.text_col]

        df = df.copy()
        if dup_strategy in {"none", "keep"}:
            return df.reset_index(drop=True)

        if dup_strategy == "drop":
            # keep first
            df[self.text_col] = df[self.text_col].astype("string").fillna("").str.strip()
            df = df.drop_duplicates(subset=subset, keep="first").copy()
            return df.reset_index(drop=True)

        if dup_strategy == "keep_longest":
            # For each duplicate group, keep row with longest text
            df[self.text_col] = df[self.text_col].astype("string").fillna("").str.strip()
            df["_len_chars"] = df[self.text_col].str.len()
            df = (
                df.sort_values("_len_chars", ascending=False)
                  .drop_duplicates(subset=subset, keep="first")
                  .drop(columns=["_len_chars"])
                  .copy()
            )
            return df.reset_index(drop=True)

        raise ValueError(f"Unknown duplicates strategy: {dup_strategy}")

    # ----------------------------
    # Outliers
    # ----------------------------

    def _detect_outliers(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        numeric_df = self._numeric_features(df)
        if numeric_df.empty:
            return []

        summaries: List[OutlierSummary] = []
        outlier_masks: Dict[str, np.ndarray] = {}

        for col in numeric_df.columns:
            series = numeric_df[col].astype(float)
            mask, (lo, hi) = self._iqr_outlier_mask(series, k=self.outlier_k)
            outlier_masks[col] = mask
            n_out = int(mask.sum())
            if n_out == 0:
                continue

            idx = np.where(mask)[0].tolist()
            example_idx = idx[: self.example_k]

            summaries.append(
                OutlierSummary(
                    feature=col,
                    method="iqr",
                    lower=float(lo),
                    upper=float(hi),
                    n_outliers=n_out,
                    outlier_rate=float(n_out / max(len(series), 1)),
                    example_indices=[int(i) for i in example_idx],
                )
            )

        # JSON-serializable
        return [self._to_jsonable(s) for s in summaries]

    def _fix_outliers(self, df: pd.DataFrame, out_strategy: str) -> pd.DataFrame:
        if out_strategy in {"none", "keep"}:
            return df.reset_index(drop=True)

        df = df.copy()
        numeric_df = self._numeric_features(df)

        if numeric_df.empty:
            return df.reset_index(drop=True)

        # Compute bounds for each numeric feature
        bounds: Dict[str, Tuple[float, float]] = {}
        for col in numeric_df.columns:
            mask, (lo, hi) = self._iqr_outlier_mask(numeric_df[col].astype(float), k=self.outlier_k)
            bounds[col] = (float(lo), float(hi))

        if out_strategy in {"drop_iqr", "drop"}:
            # Drop rows that are outliers in ANY derived numeric feature
            any_out = np.zeros(len(df), dtype=bool)
            for col, (lo, hi) in bounds.items():
                s = numeric_df[col].astype(float)
                any_out |= (s < lo) | (s > hi)
            df = df[~any_out].copy()
            return df.reset_index(drop=True)

        if out_strategy in {"clip_iqr", "clip"}:
            # Clip numeric columns; for text length derived features, we do NOT rewrite text here.
            # We only clip existing numeric columns (if any).
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]) and col in bounds:
                    lo, hi = bounds[col]
                    df[col] = df[col].clip(lower=lo, upper=hi)
            return df.reset_index(drop=True)

        raise ValueError(f"Unknown outliers strategy: {out_strategy}")

    def _numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a numeric feature frame for outlier detection.
        Includes:
          - any numeric columns already in df
          - derived features from text: char length and word count
        """
        feats = {}

        # Derived from text
        if self.text_col in df.columns:
            text = df[self.text_col].astype("string").fillna("")
            feats["_text_len_chars"] = text.str.len().astype(float)
            feats["_text_len_words"] = (
                text.str.replace(r"\s+", " ", regex=True).str.strip().str.split(" ").apply(lambda x: 0 if x == [""] else len(x)).astype(float)
            )

        # Existing numeric columns
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and col not in feats:
                feats[col] = df[col].astype(float)

        return pd.DataFrame(feats)

    @staticmethod
    def _iqr_outlier_mask(series: pd.Series, k: float = 1.5) -> Tuple[np.ndarray, Tuple[float, float]]:
        s = series.dropna()
        if len(s) < 4:
            # Not enough data to estimate IQR; return no outliers
            return np.zeros(len(series), dtype=bool), (float("-inf"), float("inf"))
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lo = q1 - k * iqr
        hi = q3 + k * iqr
        mask = (series < lo) | (series > hi)
        return mask.to_numpy(dtype=bool), (float(lo), float(hi))

    # ----------------------------
    # Class imbalance
    # ----------------------------

    def _detect_imbalance(self, df: pd.DataFrame) -> Dict[str, Any]:
        if self.label_col not in df.columns:
            return {"note": f"label column '{self.label_col}' not found"}

        labels = df[self.label_col].astype("string").fillna("unknown")
        counts = labels.value_counts(dropna=False)
        total = int(counts.sum()) if len(counts) else 0
        rates = (counts / total).to_dict() if total else {}

        # "Imbalance" heuristic: classes with rate < threshold
        rare = {k: float(v) for k, v in rates.items() if v < self.imbalance_warn_threshold}
        majority_rate = float(max(rates.values())) if rates else 0.0
        minority_rate = float(min(rates.values())) if rates else 0.0
        ratio = float(majority_rate / minority_rate) if minority_rate > 0 else float("inf")

        return {
            "label_col": self.label_col,
            "counts": {str(k): int(v) for k, v in counts.to_dict().items()},
            "rates": {str(k): float(v) for k, v in rates.items()},
            "majority_rate": majority_rate,
            "minority_rate": minority_rate,
            "majority_to_minority_ratio": ratio,
            "rare_classes": rare,
        }

    def _fix_imbalance(self, df: pd.DataFrame, imb_strategy: str, *, target_col: str) -> pd.DataFrame:
        if imb_strategy in {"none", "keep"}:
            return df.reset_index(drop=True)

        if target_col not in df.columns:
            return df.reset_index(drop=True)

        df = df.copy()
        rng = np.random.default_rng(self.random_state)

        counts = df[target_col].astype("string").fillna("unknown").value_counts()
        if len(counts) < 2:
            return df.reset_index(drop=True)

        if imb_strategy == "downsample":
            target_n = int(counts.min())
            parts = []
            for cls, grp in df.groupby(target_col, dropna=False):
                if len(grp) <= target_n:
                    parts.append(grp)
                else:
                    idx = rng.choice(grp.index.to_numpy(), size=target_n, replace=False)
                    parts.append(df.loc[idx])
            return pd.concat(parts, ignore_index=True).reset_index(drop=True)

        if imb_strategy == "upsample":
            target_n = int(counts.max())
            parts = []
            for cls, grp in df.groupby(target_col, dropna=False):
                if len(grp) >= target_n:
                    parts.append(grp)
                else:
                    idx = rng.choice(grp.index.to_numpy(), size=target_n, replace=True)
                    parts.append(df.loc[idx])
            return pd.concat(parts, ignore_index=True).reset_index(drop=True)

        raise ValueError(f"Unknown imbalance strategy: {imb_strategy}")

    # ----------------------------
    # Metrics helpers
    # ----------------------------

    def _compute_quality_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        df_ = df.copy()
        df_[self.text_col] = df_[self.text_col].astype("string")
        df_["_text_norm"] = df_[self.text_col].fillna("").str.strip()

        missing = self._detect_missing(df_)
        duplicates = self._detect_duplicates(df_)
        outliers = self._detect_outliers(df_)
        imbalance = self._detect_imbalance(df_)

        # Outlier total across derived features
        outlier_total = 0
        if outliers:
            outlier_total = int(sum(o.get("n_outliers", 0) for o in outliers))

        metrics = {
            "n_rows": float(len(df_)),
            "missing_required_rows": float(missing.get("_required_rows", {}).get("count", 0)),
            "missing_total_columns_with_missing": float(len([k for k in missing.keys() if k not in {"_required_rows"}])),
            "duplicates_count": float(duplicates.get("count", 0)),
            "outliers_total": float(outlier_total),
            "n_classes": float(len(imbalance.get("counts", {}))) if isinstance(imbalance, dict) else np.nan,
            "majority_to_minority_ratio": float(imbalance.get("majority_to_minority_ratio", np.nan)) if isinstance(imbalance, dict) else np.nan,
        }
        return metrics

    def _rows_missing_required(self, df: pd.DataFrame) -> pd.Series:
        # required columns must exist; otherwise _validate_schema would raise
        text_bad = df[self.text_col].isna()
        if "_text_norm" in df.columns:
            text_bad = text_bad | (df["_text_norm"] == "")
        required_bad = text_bad
        for col in [c for c in self.REQUIRED_COLUMNS if c != self.text_col]:
            required_bad = required_bad | df[col].isna()
        return required_bad

    # ----------------------------
    # Utilities
    # ----------------------------

    @staticmethod
    def _is_number(x: Any) -> bool:
        return isinstance(x, (int, float, np.number)) and not (isinstance(x, float) and np.isnan(x))

    def _validate_schema(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"DataQualityAgent requires columns {self.REQUIRED_COLUMNS}, missing: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

    @staticmethod
    def _to_jsonable(obj: Any) -> Any:
        """Convert dataclasses / numpy types into plain JSON-serializable structures."""
        if hasattr(obj, "__dataclass_fields__"):
            return {k: DataQualityAgent._to_jsonable(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, pd.Series):
            return obj.to_list()
        if isinstance(obj, dict):
            return {str(k): DataQualityAgent._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [DataQualityAgent._to_jsonable(v) for v in obj]
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
