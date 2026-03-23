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
from sklearn.metrics import f1_score
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
            pipe = self._make_pipeline()
            y = le.transform(labeled["label"].astype(str))
            if len(np.unique(y)) < 2:
                _log.warning("iteration=%s skipped: need at least 2 label classes", it)
                history.append(
                    {
                        "iteration": it,
                        "labeled_size": len(labeled),
                        "f1_macro": float("nan"),
                        "note": "need at least 2 label classes",
                    }
                )
                break
            pipe.fit(labeled["text"].astype(str), y)
            y_test = le.transform(test_df["label"].astype(str))
            pred = pipe.predict(test_df["text"].astype(str))
            metric = f1_score(y_test, pred, average="macro")
            history.append(
                {
                    "iteration": it,
                    "labeled_size": len(labeled),
                    "f1_macro": float(metric),
                }
            )
            _log.info(
                "iteration=%s labeled_size=%s f1_macro=%.4f pool_remaining=%s",
                it,
                len(labeled),
                float(metric),
                len(pool),
            )

            if it >= n_iterations - 1 or len(pool) <= 0:
                break

            if strategy == "random":
                idx = pool.sample(n=min(batch_size, len(pool)), random_state=rng).index
            elif strategy == "entropy":
                proba = pipe.predict_proba(pool["text"].astype(str))
                entropy = -(proba * np.log(proba + 1e-12)).sum(axis=1)
                order = np.argsort(-entropy)
                take = order[: min(batch_size, len(pool))]
                idx = pool.iloc[take].index
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            batch_n = len(idx)
            labeled = pd.concat([labeled, pool.loc[idx]], axis=0)
            pool = pool.drop(index=idx)
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
        y = le.fit_transform(labeled_df["label"].astype(str))
        pipe = self._make_pipeline()
        pipe.fit(labeled_df["text"].astype(str), y)

        if strategy == "random":
            out = pool_df.sample(n=min(batch_size, len(pool_df)), random_state=self.random_state)
            _log.info("select_batch picked rows=%s", len(out))
            return out

        proba = pipe.predict_proba(pool_df["text"].astype(str))
        entropy = -(proba * np.log(proba + 1e-12)).sum(axis=1)
        order = np.argsort(-entropy)
        out = pool_df.iloc[order[: min(batch_size, len(pool_df))]]
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
                {"data": {"text": str(r.get("text", "")), "uid": str(r.get("uid", ""))}}
                for _, r in df.iterrows()
            ]
            p.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
            _log.info("export_candidates labelstudio=%s tasks=%s", p, len(tasks))


__all__ = ["ActiveLearningAgent"]
