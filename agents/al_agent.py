from __future__ import annotations

"""
ActiveLearningAgent (Assignment 4) — Track A (Active Learning)

Implements a minimal but solid Active Learning loop for text classification.

Skills:
- fit(labeled_df) -> model
- query(pool_df, strategy) -> indices
- evaluate(labeled_df, test_df) -> metrics (accuracy, f1)
- report(history) -> learning_curve.png

Design goals:
- Works out-of-the-box for course assignments
- Generic for other text tasks: change text/label columns
- Reproducible (random_state)
- Uses a simple baseline: TF-IDF + Logistic Regression (multinomial)

Assumptions for AL simulation:
- pool_df contains the true label in `label_col` so we can "acquire" it during the loop.
  (In a real HITL pipeline, the selected batch would be exported for human labeling.)
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ALConfig:
    model: str = "logreg"
    text_col: str = "text"
    label_col: str = "label"
    uid_col: str = "uid"
    random_state: int = 42

    # Vectorizer
    max_features: int = 30000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2

    # Logistic regression
    max_iter: int = 2000
    C: float = 2.0

    # Reporting
    reports_dir: str = "reports"
    default_curve_path: str = "reports/learning_curve.png"


class ActiveLearningAgent:
    """
    ActiveLearningAgent for Track A.

    Example:
        agent = ActiveLearningAgent(model='logreg')
        history = agent.run_cycle(
            labeled_df=df_labeled_50,
            pool_df=df_unlabeled,
            test_df=df_test,
            strategy='entropy',
            n_iterations=5,
            batch_size=20
        )
        agent.report({'entropy': history, 'random': history2})
    """

    SUPPORTED_STRATEGIES = {"entropy", "margin", "random"}

    def __init__(self, model: str = "logreg", config: Optional[Dict[str, Any]] = None) -> None:
        cfg = ALConfig(model=model)
        if config:
            for k, v in config.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        self.cfg = cfg
        Path(self.cfg.reports_dir).mkdir(parents=True, exist_ok=True)

        self.pipeline: Optional[Pipeline] = None
        self.label_list_: Optional[List[str]] = None

    # ----------------------------
    # Skills
    # ----------------------------

    def fit(self, labeled_df: pd.DataFrame) -> Pipeline:
        self._validate_df(labeled_df, require_labels=True)

        X = labeled_df[self.cfg.text_col].astype("string").fillna("").tolist()
        y = labeled_df[self.cfg.label_col].astype("string").fillna("unknown").tolist()

        self.label_list_ = sorted(list(set(y)))

        self.pipeline = self._build_model()
        self.pipeline.fit(X, y)
        return self.pipeline

    def query(self, pool_df: pd.DataFrame, strategy: str, batch_size: int) -> List[int]:
        """
        Return a list of row indices (from pool_df.index) to acquire.
        """
        strategy = strategy.lower().strip()
        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Supported: {sorted(self.SUPPORTED_STRATEGIES)}")

        self._validate_df(pool_df, require_labels=False)
        if len(pool_df) == 0:
            return []

        n = min(int(batch_size), len(pool_df))
        rng = np.random.default_rng(self.cfg.random_state)

        if strategy == "random" or self.pipeline is None:
            return rng.choice(pool_df.index.to_numpy(), size=n, replace=False).tolist()

        X_pool = pool_df[self.cfg.text_col].astype("string").fillna("").tolist()
        proba = self.pipeline.predict_proba(X_pool)  # shape: [N, K]

        if proba.ndim != 2 or proba.shape[0] != len(pool_df):
            # Fallback to random if predict_proba is not supported
            return rng.choice(pool_df.index.to_numpy(), size=n, replace=False).tolist()

        if strategy == "entropy":
            scores = self._entropy(proba)
            # higher entropy => more uncertain
            chosen = np.argsort(-scores)[:n]
        elif strategy == "margin":
            scores = self._margin(proba)
            # smaller margin => more uncertain
            chosen = np.argsort(scores)[:n]
        else:
            chosen = rng.choice(np.arange(len(pool_df)), size=n, replace=False)

        chosen_indices = pool_df.index.to_numpy()[chosen].tolist()
        return [int(i) if isinstance(i, (np.integer, int)) else i for i in chosen_indices]

    def evaluate(self, labeled_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, float]:
        """
        Train on labeled_df (current) and evaluate on test_df.

        Returns: {'accuracy': ..., 'f1': ...}
        """
        self._validate_df(labeled_df, require_labels=True)
        self._validate_df(test_df, require_labels=True)

        model = self.fit(labeled_df)

        X_test = test_df[self.cfg.text_col].astype("string").fillna("").tolist()
        y_true = test_df[self.cfg.label_col].astype("string").fillna("unknown").tolist()

        y_pred = model.predict(X_test)

        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, average="macro"))
        return {"accuracy": acc, "f1": f1}

    def report(
        self,
        history: Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]],
        *,
        metric: str = "f1",
        output_path: Optional[str] = None,
        title: str = "Active Learning Curve",
    ) -> str:
        """
        Plot learning curve(s) and save as PNG.

        - If history is a list -> single curve
        - If history is a dict[str, list] -> multiple curves on one plot (compare strategies)
        """
        if output_path is None:
            output_path = self.cfg.default_curve_path
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(9, 5))

        if isinstance(history, dict):
            for name, h in history.items():
                xs = [int(r["n_labeled"]) for r in h]
                ys = [float(r.get(metric, np.nan)) for r in h]
                plt.plot(xs, ys, marker="o", label=name)
            plt.legend()
        else:
            xs = [int(r["n_labeled"]) for r in history]
            ys = [float(r.get(metric, np.nan)) for r in history]
            plt.plot(xs, ys, marker="o")

        plt.xlabel("n_labeled")
        plt.ylabel(metric)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path.as_posix(), dpi=160)
        plt.close()
        return out_path.as_posix()

    # ----------------------------
    # AL loop
    # ----------------------------

    def run_cycle(
        self,
        *,
        labeled_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        test_df: pd.DataFrame,
        strategy: str = "entropy",
        n_iterations: int = 5,
        batch_size: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Active Learning loop.

        - Start from labeled_df (e.g., N=50)
        - For each iteration:
            1) train & eval -> record metrics
            2) query pool -> acquire batch (simulate labeling)
            3) move batch from pool to labeled
        """
        strategy = strategy.lower().strip()
        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Supported: {sorted(self.SUPPORTED_STRATEGIES)}")

        labeled = labeled_df.copy()
        pool = pool_df.copy()

        self._validate_df(labeled, require_labels=True)
        self._validate_df(pool, require_labels=True)  # for simulation; has oracle labels
        self._validate_df(test_df, require_labels=True)

        history: List[Dict[str, Any]] = []

        for it in range(int(n_iterations) + 1):
            # Evaluate current model
            metrics = self.evaluate(labeled, test_df)
            row = {
                "iteration": it,
                "n_labeled": int(len(labeled)),
                "accuracy": float(metrics["accuracy"]),
                "f1": float(metrics["f1"]),
                "strategy": strategy,
                "generated_at": utc_now_iso(),
            }
            history.append(row)

            if it == n_iterations:
                break

            # Select and acquire a batch
            selected_idx = self.query(pool, strategy=strategy, batch_size=int(batch_size))
            if not selected_idx:
                break

            acquired = pool.loc[selected_idx].copy()
            labeled = pd.concat([labeled, acquired], ignore_index=True)
            pool = pool.drop(index=selected_idx)

        # Keep last fitted model
        self.fit(labeled)
        return history

    # ----------------------------
    # Internals
    # ----------------------------

    def _build_model(self) -> Pipeline:
        if self.cfg.model != "logreg":
            raise ValueError("This MVP supports only model='logreg' (TF-IDF + LogisticRegression).")

        vectorizer = TfidfVectorizer(
            max_features=int(self.cfg.max_features),
            ngram_range=tuple(self.cfg.ngram_range),
            min_df=int(self.cfg.min_df),
        )
        clf = LogisticRegression(
            max_iter=int(self.cfg.max_iter),
            C=float(self.cfg.C),
            solver="lbfgs",
            multi_class="auto",
            n_jobs=None,
            random_state=int(self.cfg.random_state),
        )
        return Pipeline([("tfidf", vectorizer), ("clf", clf)])

    @staticmethod
    def _entropy(proba: np.ndarray) -> np.ndarray:
        p = np.clip(proba, 1e-12, 1.0)
        return -np.sum(p * np.log(p), axis=1)

    @staticmethod
    def _margin(proba: np.ndarray) -> np.ndarray:
        # margin = top1 - top2 (smaller => more uncertain)
        p = np.sort(proba, axis=1)[:, ::-1]
        if p.shape[1] == 1:
            return np.zeros(p.shape[0])
        return p[:, 0] - p[:, 1]

    def _validate_df(self, df: pd.DataFrame, *, require_labels: bool) -> None:
        for col in [self.cfg.text_col]:
            if col not in df.columns:
                raise ValueError(f"ActiveLearningAgent requires column '{col}'. Available: {list(df.columns)}")

        if require_labels and self.cfg.label_col not in df.columns:
            raise ValueError(f"ActiveLearningAgent requires label column '{self.cfg.label_col}' for this operation.")

        # Basic sanity: no empty strings for text
        if self.cfg.text_col in df.columns:
            text = df[self.cfg.text_col].astype("string").fillna("").str.strip()
            if (text == "").all():
                raise ValueError("All texts are empty after stripping; cannot proceed.")
