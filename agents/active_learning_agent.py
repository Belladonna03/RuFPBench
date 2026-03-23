from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from shared.config import as_config_dict
from shared.logging_utils import get_logger

_log = get_logger("agents.active_learning")


class ActiveLearningAgent:
    """Pool-based active learning with entropy or random query strategies."""

    def __init__(self, model: Literal["logreg"] = "logreg", config: str | Path | dict[str, Any] | None = None):
        self.model_name = model
        self.cfg = as_config_dict(config) if config else {}
        self.al_cfg = self.cfg.get("active_learning") or {}
        self.random_state = int(self.al_cfg.get("random_state", 42))
        self._last_pipeline: Pipeline | None = None
        self._last_label_encoder: LabelEncoder | None = None

    def _make_pipeline(self) -> Pipeline:
        if self.model_name == "logreg":
            return Pipeline(
                [
                    ("tfidf", TfidfVectorizer(max_features=20_000, ngram_range=(1, 2))),
                    (
                        "clf",
                        LogisticRegression(max_iter=200, random_state=self.random_state),
                    ),
                ]
            )
        raise ValueError(f"Unsupported model: {self.model_name}")

    def _score_pool(self, pipe: Pipeline, pool_df: pd.DataFrame, le: LabelEncoder) -> pd.DataFrame:
        scored = pool_df.copy()
        proba = pipe.predict_proba(pool_df["text"].astype(str))
        entropy = -(proba * np.log(proba + 1e-12)).sum(axis=1)
        best_idx = np.argmax(proba, axis=1)
        best = np.take_along_axis(proba, best_idx[:, None], axis=1).ravel()
        if proba.shape[1] > 1:
            second = np.partition(proba, -2, axis=1)[:, -2]
        else:
            second = np.zeros(len(proba))
        margin = best - second
        scored["uncertainty"] = entropy
        scored["margin"] = margin
        scored["predicted_label"] = le.inverse_transform(best_idx)
        return scored

    def _rank_scored(self, scored_df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        strategy = str(strategy).lower().strip()
        if strategy == "entropy":
            return scored_df.sort_values(["uncertainty", "margin"], ascending=[False, True])
        if strategy == "margin":
            return scored_df.sort_values(["margin", "uncertainty"], ascending=[True, False])
        raise ValueError(f"Unknown strategy: {strategy}")

    def fit(self, labeled_df: pd.DataFrame, *, label_space: pd.Series | None = None) -> Pipeline:
        """Train the baseline model on labeled rows and store the fitted encoder/model on the agent."""
        if "text" not in labeled_df.columns or "label" not in labeled_df.columns:
            raise ValueError("labeled_df must contain text and label columns")
        le = LabelEncoder()
        labels_for_encoder = (
            pd.Series(label_space, copy=False).astype(str)
            if label_space is not None
            else labeled_df["label"].astype(str)
        )
        le.fit(labels_for_encoder)
        y = le.transform(labeled_df["label"].astype(str))
        if len(np.unique(y)) < 2:
            raise ValueError("fit requires at least 2 label classes")
        pipe = self._make_pipeline()
        pipe.fit(labeled_df["text"].astype(str), y)
        self._last_pipeline = pipe
        self._last_label_encoder = le
        return pipe

    def evaluate(
        self,
        labeled_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        model: Pipeline | None = None,
    ) -> dict[str, Any]:
        """Evaluate a fitted model (or fit on labeled_df first) and return accuracy/F1 metrics."""
        pipe = model or self.fit(
            labeled_df,
            label_space=pd.concat([labeled_df["label"], test_df["label"]], axis=0),
        )
        le = self._last_label_encoder
        if le is None:
            raise RuntimeError("Label encoder is not available; call fit() first")
        y_test = le.transform(test_df["label"].astype(str))
        pred = pipe.predict(test_df["text"].astype(str))
        return {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1_macro": float(f1_score(y_test, pred, average="macro")),
            "n_train": int(len(labeled_df)),
            "n_test": int(len(test_df)),
        }

    def query(
        self,
        pool_df: pd.DataFrame,
        *,
        strategy: str,
        batch_size: int,
        model: Pipeline | None = None,
        labeled_df: pd.DataFrame | None = None,
    ) -> list[Any]:
        """Return pool indices selected by the query strategy."""
        strategy = str(strategy).lower().strip()
        if strategy == "random":
            if len(pool_df) == 0:
                return []
            out = pool_df.sample(n=min(batch_size, len(pool_df)), random_state=self.random_state)
            return out.index.tolist()

        pipe = model
        if pipe is None:
            if labeled_df is None:
                raise ValueError("query(..., strategy!=random) requires model or labeled_df")
            pipe = self.fit(labeled_df)
        le = self._last_label_encoder
        if le is None:
            raise RuntimeError("Label encoder is not available; call fit() first")
        scored = self._score_pool(pipe, pool_df, le)
        ranked = self._rank_scored(scored, strategy)
        selected = self._select_diverse_subset(ranked, pipe, batch_size)
        return selected.index.tolist()

    def _select_diverse_subset(self, scored_df: pd.DataFrame, pipe: Pipeline, batch_size: int) -> pd.DataFrame:
        if len(scored_df) <= batch_size or "text" not in scored_df.columns:
            return scored_df.head(min(batch_size, len(scored_df)))

        candidate_factor = int(self.al_cfg.get("diversity_candidate_pool_factor", 5))
        similarity_threshold = float(self.al_cfg.get("diversity_similarity_threshold", 0.85))
        candidate_n = min(len(scored_df), max(batch_size, batch_size * candidate_factor))
        candidates = scored_df.head(candidate_n).copy()
        tfidf = pipe.named_steps["tfidf"]
        matrix = tfidf.transform(candidates["text"].astype(str))
        sims = cosine_similarity(matrix)

        selected_positions: list[int] = []
        remaining = list(range(len(candidates)))
        while remaining and len(selected_positions) < batch_size:
            picked = None
            for pos in remaining:
                if not selected_positions:
                    picked = pos
                    break
                max_sim = float(max(sims[pos, prev] for prev in selected_positions))
                if max_sim < similarity_threshold:
                    picked = pos
                    break
            if picked is None:
                picked = remaining[0]
            selected_positions.append(picked)
            remaining.remove(picked)

        return candidates.iloc[selected_positions].copy()

    def run_cycle(
        self,
        labeled_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        strategy: str,
        n_iterations: int,
        batch_size: int,
    ) -> list[dict[str, Any]]:
        if "text" not in labeled_df.columns or "label" not in labeled_df.columns:
            raise ValueError("labeled_df must contain text and label columns")
        if "text" not in pool_df.columns or "label" not in pool_df.columns:
            raise ValueError("run_cycle requires pool_df with text and hidden label columns for offline AL simulation")
        rng = np.random.RandomState(self.random_state)

        history: list[dict[str, Any]] = []
        labeled = labeled_df.copy().reset_index(drop=True)
        pool = pool_df.copy().reset_index(drop=True)

        le = LabelEncoder()
        le.fit(pd.concat([labeled["label"], test_df["label"]], axis=0).astype(str))

        _log.info(
            "run_cycle strategy=%s iterations=%s batch_size=%s labeled=%s pool=%s test=%s",
            strategy,
            n_iterations,
            batch_size,
            len(labeled),
            len(pool),
            len(test_df),
        )

        for it in range(n_iterations):
            y = le.transform(labeled["label"].astype(str))
            if len(np.unique(y)) < 2:
                _log.warning("iteration=%s skipped: need at least 2 label classes", it)
                history.append(
                    {
                        "iteration": it,
                        "n_labeled": len(labeled),
                        "accuracy": float("nan"),
                        "f1_macro": float("nan"),
                        "note": "need at least 2 label classes",
                    }
                )
                break
            pipe = self.fit(
                labeled,
                label_space=pd.concat([labeled["label"], test_df["label"]], axis=0),
            )
            metrics = self.evaluate(labeled, test_df, model=pipe)
            history.append(
                {
                    "iteration": it,
                    "n_labeled": len(labeled),
                    "accuracy": float(metrics["accuracy"]),
                    "f1_macro": float(metrics["f1_macro"]),
                }
            )
            _log.info(
                "iteration=%s n_labeled=%s accuracy=%.4f f1_macro=%.4f pool_remaining=%s",
                it,
                len(labeled),
                float(metrics["accuracy"]),
                float(metrics["f1_macro"]),
                len(pool),
            )

            if it >= n_iterations - 1 or len(pool) <= 0:
                break

            if strategy == "random":
                batch = pool.sample(n=min(batch_size, len(pool)), random_state=rng).copy()
                batch["uncertainty"] = np.nan
                batch["margin"] = np.nan
            elif strategy in ("entropy", "margin"):
                scored = self._score_pool(pipe, pool, le)
                ranked = self._rank_scored(scored, strategy)
                batch = self._select_diverse_subset(ranked, pipe, batch_size)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            batch_n = len(batch)
            labeled = pd.concat([labeled, batch], axis=0)
            pool = pool.drop(index=batch.index)
            _log.info("iteration=%s added_from_pool=%s pool_remaining=%s", it, batch_n, len(pool))

        return history

    def report(
        self,
        histories: dict[str, list[dict[str, Any]]],
        *,
        metric: str = "f1",
        output_path: str | Path | None = None,
        title: str = "Active Learning",
    ) -> Path | None:
        if not output_path:
            return None
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 5))
        for name, hist in histories.items():
            xs = [h["iteration"] for h in hist]
            if metric == "accuracy":
                ys = [h.get("accuracy", 0.0) for h in hist]
            else:
                ys = [h.get(metric + "_macro", h.get("f1_macro", 0.0)) for h in hist]
            plt.plot(xs, ys, marker="o", label=name)
        plt.xlabel("iteration")
        plt.ylabel(metric)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        _log.info("saved learning_curve path=%s", output_path)
        return output_path

    def select_batch(
        self,
        pool_df: pd.DataFrame,
        labeled_df: pd.DataFrame,
        *,
        strategy: str,
        batch_size: int,
    ) -> pd.DataFrame:
        """Select the next batch from pool using a model trained on labeled rows."""
        _log.info(
            "select_batch strategy=%s batch_size=%s labeled=%s pool=%s",
            strategy,
            batch_size,
            len(labeled_df),
            len(pool_df),
        )
        if len(pool_df) == 0:
            _log.warning("select_batch empty pool")
            return pool_df
        if len(labeled_df) < 2 or "text" not in pool_df.columns:
            _log.warning("select_batch fallback: insufficient labeled rows for model")
            return pool_df.head(min(batch_size, len(pool_df)))

        le = LabelEncoder()
        pipe = self.fit(labeled_df)

        if strategy == "random":
            out = pool_df.sample(n=min(batch_size, len(pool_df)), random_state=self.random_state).copy()
            out["uncertainty"] = np.nan
            out["margin"] = np.nan
            _log.info("select_batch picked rows=%s", len(out))
            return out

        le = self._last_label_encoder
        if le is None:
            raise RuntimeError("Label encoder is not available; call fit() first")
        scored = self._score_pool(pipe, pool_df, le)
        ranked = self._rank_scored(scored, strategy)
        out = self._select_diverse_subset(ranked, pipe, batch_size)
        _log.info("select_batch picked rows=%s", len(out))
        return out

    def export_candidates(
        self,
        df: pd.DataFrame,
        *,
        output_csv: str | Path,
        output_labelstudio: str | Path | None = None,
    ) -> None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        _log.info("export_candidates csv=%s rows=%s", output_csv, len(df))
        if output_labelstudio:
            import json

            p = Path(output_labelstudio)
            p.parent.mkdir(parents=True, exist_ok=True)
            tasks = [
                {
                    "data": {
                        "text": str(r.get("text", "")),
                        "uid": str(r.get("uid", "")),
                        "predicted_label": str(r.get("predicted_label", "")),
                        "uncertainty": None if pd.isna(r.get("uncertainty")) else float(r.get("uncertainty", 0.0)),
                        "margin": None if pd.isna(r.get("margin")) else float(r.get("margin", 0.0)),
                    }
                }
                for _, r in df.iterrows()
            ]
            p.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
            _log.info("export_candidates labelstudio=%s tasks=%s", p, len(tasks))


__all__ = ["ActiveLearningAgent"]
