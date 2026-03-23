from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from shared.logging_utils import get_logger

_log = get_logger("agents.data_quality")

StrategyDict = dict[str, Any]
IssuesDict = dict[str, Any]


def _to_python_scalars(obj: Any) -> Any:
    """Recursively convert numpy scalars for JSON-safe reports."""
    if isinstance(obj, dict):
        return {k: _to_python_scalars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python_scalars(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def _blank_text_mask(s: pd.Series) -> pd.Series:
    st = s.astype("string")
    return st.isna() | (st.str.strip() == "")


def _word_counts(s: pd.Series) -> pd.Series:
    st = s.astype("string").fillna("")
    return st.str.split().str.len().astype(float)


def _iqr_fences(series: pd.Series, k: float) -> tuple[float, float] | None:
    clean = series.dropna()
    if len(clean) == 0:
        return None
    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    low = q1 - k * iqr
    high = q3 + k * iqr
    return low, high


def _zscore_outlier_mask(series: pd.Series, threshold: float) -> pd.Series:
    clean = series.dropna()
    if len(clean) < 2:
        return pd.Series(False, index=series.index)
    mu = float(clean.mean())
    sigma = float(clean.std(ddof=0))
    if sigma == 0.0 or math.isnan(sigma):
        return pd.Series(False, index=series.index)
    z = ((series - mu) / sigma).abs()
    return z > threshold


class DataQualityAgent:
    """
    Rule-based data-quality agent for tabular data, tuned for text classification.

    Observes issues (``detect_issues``), decides a cleaning plan (``choose_strategy``),
    applies it (``fix``), and evaluates before/after (``compare``). ``run`` ties the
    loop together without LLM calls in the core path.
    """

    def __init__(
        self,
        task_type: str | dict[str, Any] = "text_classification",
        text_column: str = "text",
        label_column: str = "label",
        duplicate_subset: list[str] | None = None,
        outlier_method: str = "iqr",
        iqr_k: float = 1.5,
        zscore_threshold: float = 3.0,
    ) -> None:
        cfg: dict[str, Any] = {}
        if isinstance(task_type, dict):
            cfg = task_type
            task_type = str(cfg.get("task_type", "text_classification"))

        self.task_type: str = task_type
        self.text_column: str = str(cfg.get("text_column", text_column))
        self.label_column: str = str(cfg.get("label_column", label_column))

        dup = duplicate_subset
        if dup is None and "duplicate_subset" in cfg:
            dup = cfg.get("duplicate_subset")
        if dup is None and self.task_type == "text_classification":
            dup = [self.text_column]
        self.duplicate_subset: list[str] = list(dup) if dup else []

        self.outlier_method: str = str(cfg.get("outlier_method", outlier_method))
        self.iqr_k: float = float(cfg.get("outlier_k", cfg.get("iqr_k", iqr_k)))
        self.zscore_threshold: float = float(cfg.get("zscore_threshold", zscore_threshold))

    # --- diagnostics ---

    def _missing_report(self, df: pd.DataFrame) -> dict[str, Any]:
        if df.empty or len(df.columns) == 0:
            return {
                "per_column": {},
                "total_cells_missing": 0,
                "critical": {
                    self.text_column: 0,
                    self.label_column: 0,
                },
            }
        per: dict[str, int] = {}
        for col in df.columns:
            s = df[col]
            if s.dtype == object or pd.api.types.is_string_dtype(s):
                n = int((s.isna() | (s.astype(str).str.strip() == "")).sum())
            else:
                n = int(s.isna().sum())
            per[str(col)] = n
        critical: dict[str, int] = {}
        if self.text_column in df.columns:
            critical[self.text_column] = int(_blank_text_mask(df[self.text_column]).sum())
        else:
            critical[self.text_column] = 0
        if self.label_column in df.columns:
            lb = df[self.label_column]
            if pd.api.types.is_numeric_dtype(lb):
                critical[self.label_column] = int(lb.isna().sum())
            else:
                critical[self.label_column] = int(
                    (lb.isna() | _blank_text_mask(lb.astype("string"))).sum()
                )
        else:
            critical[self.label_column] = 0
        total_cells_missing = int(sum(per.values()))
        return {
            "per_column": per,
            "total_cells_missing": total_cells_missing,
            "critical": critical,
        }

    def _duplicate_count(self, df: pd.DataFrame) -> int:
        cols = [c for c in self.duplicate_subset if c in df.columns]
        if not cols or df.empty:
            return 0
        return int(df.duplicated(subset=cols, keep="first").sum())

    def _outliers_for_series(
        self,
        name: str,
        series: pd.Series,
        *,
        use_zscore: bool,
    ) -> dict[str, Any]:
        fences = _iqr_fences(series, self.iqr_k)
        if fences is None:
            return {
                "n_iqr": 0,
                "iqr_bounds": None,
                "n_zscore": 0,
            }
        low, high = fences
        mask_iqr = series.notna() & ((series < low) | (series > high))
        n_iqr = int(mask_iqr.sum())
        n_z = 0
        if use_zscore:
            n_z = int(_zscore_outlier_mask(series, self.zscore_threshold).sum())
        return {
            "n_iqr": n_iqr,
            "iqr_bounds": [low, high],
            "n_zscore": n_z,
        }

    def _numeric_columns(self, df: pd.DataFrame) -> list[str]:
        out: list[str] = []
        for c in df.columns:
            if c in (self.text_column, self.label_column):
                continue
            s = df[c]
            if pd.api.types.is_bool_dtype(s):
                continue
            if pd.api.types.is_numeric_dtype(s):
                out.append(str(c))
        return out

    def detect_issues(self, df: pd.DataFrame) -> IssuesDict:
        """
        Scan the frame for missing values, duplicates, length/numeric outliers, and class imbalance.

        Returns a dict with keys ``missing``, ``duplicates``, ``outliers``, ``imbalance``,
        ``dataset_profile`` (JSON-friendly scalars).
        """
        _log.info("detect_issues input_rows=%s cols=%s", len(df), len(df.columns))

        missing = self._missing_report(df)
        dup_n = self._duplicate_count(df)

        outliers_detail: dict[str, Any] = {}
        use_z = self.outlier_method in ("both", "zscore", "iqr+zscore")

        if self.text_column in df.columns:
            t = df[self.text_column]
            char_len = t.astype("string").fillna("").str.len().astype(float)
            word_len = _word_counts(t)
            outliers_detail["text_len_chars"] = self._outliers_for_series(
                "text_len_chars", char_len, use_zscore=use_z
            )
            outliers_detail["text_len_words"] = self._outliers_for_series(
                "text_len_words", word_len, use_zscore=use_z
            )
        for ncol in self._numeric_columns(df):
            outliers_detail[str(ncol)] = self._outliers_for_series(
                str(ncol), df[ncol].astype(float), use_zscore=use_z
            )

        total_outlier_rows = 0
        if self.text_column in df.columns:
            t = df[self.text_column]
            char_len = t.astype("string").fillna("").str.len().astype(float)
            word_len = _word_counts(t)
            m_char = np.zeros(len(df), dtype=bool)
            m_word = np.zeros(len(df), dtype=bool)
            fc = _iqr_fences(char_len, self.iqr_k)
            if fc is not None:
                lo, hi = fc
                m_char = char_len.notna().values & ((char_len.values < lo) | (char_len.values > hi))
            fw = _iqr_fences(word_len, self.iqr_k)
            if fw is not None:
                lo, hi = fw
                m_word = word_len.notna().values & ((word_len.values < lo) | (word_len.values > hi))
            m_num = np.zeros(len(df), dtype=bool)
            for ncol in self._numeric_columns(df):
                s = df[ncol].astype(float)
                f = _iqr_fences(s, self.iqr_k)
                if f is None:
                    continue
                lo, hi = f
                m_num |= (s.notna().values & ((s.values < lo) | (s.values > hi)))
            union = m_char | m_word | m_num
            total_outlier_rows = int(union.sum())

        imbalance: dict[str, Any] | None = None
        if self.label_column in df.columns and not df.empty:
            lb = df[self.label_column]
            valid = lb.notna() & ~_blank_text_mask(lb.astype("string"))
            vc = lb[valid].astype(str).value_counts()
            counts = {str(k): int(v) for k, v in vc.items()}
            total = int(vc.sum())
            if total > 0:
                maj = str(vc.index[0])
                maj_share = float(vc.iloc[0] / total)
                imbalance = {
                    "counts": counts,
                    "majority_class": maj,
                    "majority_share": maj_share,
                }

        profile = {
            "n_rows": int(len(df)),
            "n_columns": int(len(df.columns)),
            "columns": [str(c) for c in df.columns],
            "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
            "duplicate_subset_effective": [c for c in self.duplicate_subset if c in df.columns],
        }

        report: IssuesDict = {
            "missing": missing,
            "duplicates": dup_n,
            "outliers": {
                "method": self.outlier_method,
                "by_feature": outliers_detail,
                "rows_flagged_iqr_union": total_outlier_rows,
            },
            "imbalance": imbalance,
            "dataset_profile": profile,
        }
        _log.info(
            "detect_issues duplicates=%s outlier_rows_union=%s",
            dup_n,
            total_outlier_rows,
        )
        return _to_python_scalars(report)

    # --- decision ---

    def choose_strategy(self, report: IssuesDict, df: pd.DataFrame) -> StrategyDict:
        """
        Choose cleaning actions from diagnostics (rule-based, no LLM).

        For ``text_classification``, applies the user's prescribed policy for missing
        values, duplicates, and text-length outliers, and appends human-readable
        ``reasoning`` strings.
        """
        reasoning: list[str] = []
        strategy: StrategyDict = {
            "missing": "keep",
            "duplicates": "keep",
            "outliers": "keep",
            "reasoning": reasoning,
        }

        miss = report.get("missing") or {}
        critical = miss.get("critical") or {}
        n_text_crit = int(critical.get(self.text_column, 0))
        n_label_crit = int(critical.get(self.label_column, 0))

        if self.task_type == "text_classification":
            if n_text_crit > 0 or n_label_crit > 0:
                strategy["missing"] = "drop_critical_fill_optional"
                reasoning.append(
                    "Строки без текста или метки непригодны для текстовой классификации и должны быть удалены."
                )
            elif int(miss.get("total_cells_missing", 0) or 0) > 0:
                strategy["missing"] = "fill"
                reasoning.append(
                    "Пропуски только в некритичных полях: заполним заглушками, чтобы сохранить строки."
                )
            else:
                strategy["missing"] = "keep"
                reasoning.append("Критичных пропусков нет; шаг по missing не требуется.")
        else:
            if n_text_crit > 0 or n_label_crit > 0:
                strategy["missing"] = "drop"
                reasoning.append("Удаляем строки с пропусками в ключевых полях.")
            elif int(miss.get("total_cells_missing", 0) or 0) > 0:
                strategy["missing"] = "median"
                reasoning.append("Заполняем числовые пропуски медианой.")
            else:
                strategy["missing"] = "keep"

        dup_n = int(report.get("duplicates", 0) or 0)
        if dup_n > 0:
            strategy["duplicates"] = "drop"
            reasoning.append("Дубликаты могут искажать обучение и оценку модели.")
        else:
            strategy["duplicates"] = "keep"
            reasoning.append("Дубликатов для выбранного subset нет.")

        ob = report.get("outliers") or {}
        by_f = ob.get("by_feature") or {}
        text_len_out = 0
        for key in ("text_len_chars", "text_len_words"):
            block = by_f.get(key) or {}
            text_len_out += int(block.get("n_iqr", 0) or 0)

        if self.task_type == "text_classification" and text_len_out > 0:
            strategy["outliers"] = "drop_iqr"
            reasoning.append(
                "Экстремальные длины текста часто являются шумом; по умолчанию удаляем строки по IQR."
            )
        else:
            any_iqr = any(
                int((v or {}).get("n_iqr", 0) or 0) > 0
                for k, v in by_f.items()
                if isinstance(v, dict)
            )
            if any_iqr and self.task_type != "text_classification":
                strategy["outliers"] = "drop_iqr"
                reasoning.append("Обнаружены выбросы по IQR в числовых признаках; удаляем строки.")
            else:
                strategy["outliers"] = "keep"
                reasoning.append("Выбросов по выбранным правилам нет или они не критичны для задачи.")

        imb = report.get("imbalance")
        if isinstance(imb, dict):
            share = float(imb.get("majority_share", 0) or 0)
            if share >= 0.9 and len(imb.get("counts") or {}) > 1:
                reasoning.append(
                    f"Сильный дисбаланс классов (доля мажоритарного класса ~{share:.2f}); "
                    "стоит учесть при обучении (веса, метрики, сэмплирование)."
                )

        return strategy

    # --- actions ---

    def _drop_critical_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        mask = pd.Series(True, index=out.index)
        if self.text_column in out.columns:
            mask &= ~_blank_text_mask(out[self.text_column])
        if self.label_column in out.columns:
            lb = out[self.label_column]
            if pd.api.types.is_numeric_dtype(lb):
                mask &= lb.notna()
            else:
                mask &= lb.notna() & ~_blank_text_mask(lb.astype("string"))
        return out.loc[mask].reset_index(drop=True)

    def _fill_optional(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        critical = {self.text_column, self.label_column}
        for c in out.columns:
            if c in critical:
                continue
            s = out[c]
            if pd.api.types.is_numeric_dtype(s):
                out[c] = s.fillna(0.0)
            else:
                out[c] = s.astype("string").fillna("").replace({pd.NA: ""})
        return out

    def _apply_missing(self, df: pd.DataFrame, mode: str) -> pd.DataFrame:
        if mode in ("keep", None):
            return df.copy()
        if mode == "drop_critical_fill_optional":
            out = self._drop_critical_rows(df)
            return self._fill_optional(out)
        if mode == "drop":
            return self._drop_critical_rows(df)
        if mode == "fill":
            out = df.copy()
            for c in out.columns:
                s = out[c]
                if pd.api.types.is_numeric_dtype(s):
                    out[c] = s.fillna(0.0)
                else:
                    out[c] = s.astype("string").fillna("").replace({pd.NA: ""})
            return out
        if mode == "median":
            out = df.copy()
            for c in out.columns:
                if pd.api.types.is_numeric_dtype(out[c]):
                    med = out[c].median()
                    if pd.isna(med):
                        med = 0.0
                    out[c] = out[c].fillna(med)
            return out
        _log.warning("unknown missing strategy=%s; no change", mode)
        return df.copy()

    def _apply_duplicates(self, df: pd.DataFrame, mode: str) -> pd.DataFrame:
        if mode != "drop":
            return df.copy()
        cols = [c for c in self.duplicate_subset if c in df.columns]
        if not cols:
            return df.copy()
        return df.drop_duplicates(subset=cols, keep="first").reset_index(drop=True)

    def _outlier_drop_mask(self, df: pd.DataFrame) -> pd.Series:
        """Rows to DROP (True = drop) based on IQR on text lengths and numeric columns."""
        if df.empty:
            return pd.Series(dtype=bool, index=df.index)
        drop = pd.Series(False, index=df.index)
        if self.text_column in df.columns:
            t = df[self.text_column]
            char_len = t.astype("string").fillna("").str.len().astype(float)
            word_len = _word_counts(t)
            for series in (char_len, word_len):
                f = _iqr_fences(series, self.iqr_k)
                if f is None:
                    continue
                lo, hi = f
                bad = series.notna() & ((series < lo) | (series > hi))
                drop |= bad
        for ncol in self._numeric_columns(df):
            s = df[ncol].astype(float)
            f = _iqr_fences(s, self.iqr_k)
            if f is None:
                continue
            lo, hi = f
            bad = s.notna() & ((s < lo) | (s > hi))
            drop |= bad
        return drop

    def _apply_outliers(self, df: pd.DataFrame, mode: str) -> pd.DataFrame:
        if mode in ("keep", None):
            return df.copy()
        if mode == "drop_iqr":
            out = df.copy()
            bad = self._outlier_drop_mask(out)
            return out.loc[~bad].reset_index(drop=True)
        if mode == "clip_iqr":
            out = df.copy()
            if self.text_column in out.columns:
                t = out[self.text_column].astype("string").fillna("")
                char_len = t.str.len().astype(float)
                f = _iqr_fences(char_len, self.iqr_k)
                if f is not None:
                    lo_f, hi_f = f
                    lo_i = max(0, int(round(lo_f)))
                    hi_i = max(0, int(round(hi_f)))

                    def _clip_one(x: str) -> str:
                        if len(x) > hi_i:
                            return x[:hi_i]
                        return x

                    out[self.text_column] = t.map(_clip_one)
            for ncol in self._numeric_columns(out):
                s = out[ncol].astype(float)
                f = _iqr_fences(s, self.iqr_k)
                if f is None:
                    continue
                lo, hi = f
                out[ncol] = s.clip(lower=lo, upper=hi)
            return out
        _log.warning("unknown outliers strategy=%s; no change", mode)
        return df.copy()

    def fix(self, df: pd.DataFrame, strategy: StrategyDict) -> pd.DataFrame:
        """
        Apply ``strategy`` without mutating the input frame.

        Order: missing handling → duplicate policy → outlier policy.
        Unknown keys in ``strategy`` (e.g. ``reasoning``) are ignored.
        """
        n0 = len(df)
        strat = {k: v for k, v in strategy.items() if k != "reasoning"}
        _log.info("fix start rows=%s strategy=%s", n0, strat)

        out = self._apply_missing(df, str(strat.get("missing", "keep")))
        out = self._apply_duplicates(out, str(strat.get("duplicates", "keep")))
        out = self._apply_outliers(out, str(strat.get("outliers", "keep")))

        _log.info("fix done rows=%s (removed %s)", len(out), n0 - len(out))
        return out

    # --- evaluation ---

    def compare(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
        """
        Build a compact before/after table (rows, duplicates, missing per column, outlier features).
        """
        rep_b = self.detect_issues(df_before)
        rep_a = self.detect_issues(df_after)

        rows: list[dict[str, Any]] = [
            {
                "metric": "rows",
                "before": int(rep_b["dataset_profile"]["n_rows"]),
                "after": int(rep_a["dataset_profile"]["n_rows"]),
            },
            {
                "metric": "duplicates",
                "before": int(rep_b.get("duplicates", 0)),
                "after": int(rep_a.get("duplicates", 0)),
            },
        ]

        miss_b = (rep_b.get("missing") or {}).get("per_column") or {}
        miss_a = (rep_a.get("missing") or {}).get("per_column") or {}
        for col in sorted(set(miss_b) | set(miss_a)):
            rows.append(
                {
                    "metric": f"missing::{col}",
                    "before": int(miss_b.get(col, 0)),
                    "after": int(miss_a.get(col, 0)),
                }
            )

        ob = rep_b.get("outliers") or {}
        oa = rep_a.get("outliers") or {}
        fb = ob.get("by_feature") or {}
        fa = oa.get("by_feature") or {}
        for feat in sorted(set(fb) | set(fa)):
            bb = (fb.get(feat) or {}).get("n_iqr", 0)
            aa = (fa.get(feat) or {}).get("n_iqr", 0)
            rows.append(
                {
                    "metric": f"outliers::iqr::{feat}",
                    "before": int(bb or 0),
                    "after": int(aa or 0),
                }
            )

        rows.append(
            {
                "metric": "outliers::rows_flagged_iqr_union",
                "before": int((ob.get("rows_flagged_iqr_union") or 0)),
                "after": int((oa.get("rows_flagged_iqr_union") or 0)),
            }
        )

        imb_b = rep_b.get("imbalance")
        imb_a = rep_a.get("imbalance")
        if imb_b or imb_a:
            rows.append(
                {
                    "metric": "imbalance::majority_share",
                    "before": float((imb_b or {}).get("majority_share", float("nan")))
                    if isinstance(imb_b, dict)
                    else float("nan"),
                    "after": float((imb_a or {}).get("majority_share", float("nan")))
                    if isinstance(imb_a, dict)
                    else float("nan"),
                }
            )

        return pd.DataFrame(rows)

    # --- orchestration ---

    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Full agent loop: diagnose → decide → fix → compare.

        Returns a dict with ``report_before``, ``chosen_strategy``, ``df_clean``, ``comparison``.
        """
        report_before = self.detect_issues(df)
        chosen = self.choose_strategy(report_before, df)
        df_clean = self.fix(df, chosen)
        comparison = self.compare(df, df_clean)
        return {
            "report_before": report_before,
            "chosen_strategy": chosen,
            "df_clean": df_clean,
            "comparison": comparison,
        }

    def explain_strategy_llm(
        self,
        strategy: StrategyDict,
        report: IssuesDict,
        *,
        client: Any = None,
    ) -> str | None:
        """
        Optional hook for LLM-based narration of ``strategy``; core agent does not call this.

        Override or pass an OpenAI-compatible ``client`` in your own code if needed.
        """
        if client is None:
            return None
        return None


__all__ = ["DataQualityAgent"]
