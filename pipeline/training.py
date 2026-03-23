from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from shared.logging_utils import get_logger
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

_log = get_logger("pipeline.training")


def train_baseline_classifier(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    label_col: str = "final_label",
    test_size: float = 0.2,
    random_state: int = 42,
    output_model_path: str | Path | None = None,
    reports_dir: Path | None = None,
) -> Pipeline:
    if label_col not in df.columns:
        if "label" in df.columns:
            label_col = "label"
        else:
            raise ValueError("DataFrame must contain final_label or label")

    sub = df[[text_col, label_col]].dropna()
    sub = sub[sub[text_col].astype(str).str.strip() != ""]
    if len(sub) < 4:
        raise ValueError("Not enough labeled rows for training.")

    _log.info("training start labeled_rows=%s", len(sub))
    X = sub[text_col].astype(str)
    y = sub[label_col].astype(str)

    strat = y if y.nunique() > 1 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=strat
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=30_000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=300, random_state=random_state)),
        ]
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    report = classification_report(y_test, pred)

    if reports_dir:
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "train_report.txt").write_text(report, encoding="utf-8")

    if output_model_path:
        outp = Path(output_model_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, outp)
        _log.info("saved model path=%s", outp)

    return pipe


def run_training_from_config(df: pd.DataFrame, cfg: dict[str, Any], reports_dir: Path) -> Path | None:
    tcfg = cfg.get("training") or {}
    if not tcfg.get("enabled", False):
        _log.warning("training skipped reason=training.disabled")
        return None
    out = tcfg.get("output_model_path", "models/baseline_logreg.pkl")
    test_size = float(tcfg.get("test_size", 0.2))
    rs = int(tcfg.get("random_state", 42))
    train_baseline_classifier(
        df,
        test_size=test_size,
        random_state=rs,
        output_model_path=out,
        reports_dir=reports_dir,
    )
    return Path(out)
