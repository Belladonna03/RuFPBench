from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shared.config import as_config_dict
from shared.logging_utils import get_logger
from shared.paths import ProjectPaths

_log = get_logger("agents.annotation")


class AnnotationAgent:
    """Weak auto-labeling, annotation spec, quality metrics, Label Studio export, HITL queue."""

    def __init__(self, modality: str = "text", config: dict[str, Any] | str | Path | None = None):
        self.modality = modality
        if isinstance(config, (str, Path)):
            self.cfg = as_config_dict(config)
            self.annotation_cfg = self.cfg.get("annotation") or {}
            self.hitl_cfg = self.cfg.get("hitl") or {}
            root = Path(__file__).resolve().parents[1]
            self.paths = ProjectPaths.from_config(root, self.cfg)
        elif isinstance(config, dict):
            self.cfg = config
            self.annotation_cfg = config.get("annotation") or {}
            self.hitl_cfg = config.get("hitl") or {}
            root = Path(__file__).resolve().parents[1]
            self.paths = ProjectPaths.from_config(root, {})
        else:
            self.cfg = {}
            self.annotation_cfg = {}
            self.hitl_cfg = {}
            root = Path(__file__).resolve().parents[1]
            self.paths = ProjectPaths.from_config(root, {})

        self.confidence_threshold = float(self.annotation_cfg.get("confidence_threshold", 0.7))
        self.include_predictions = bool(self.annotation_cfg.get("include_predictions", True))

    def auto_label(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        labels_cfg = (self.cfg.get("project") or {}).get("labels") or [
            "candidate_benign_borderline",
            "plain_benign_control",
            "unsafe_or_not_suitable",
        ]

        def _predict(row: pd.Series) -> tuple[str, float]:
            text = str(row.get("text") or "")
            meta = row.get("meta")
            seed = ""
            if isinstance(meta, str) and meta:
                try:
                    seed = str(json.loads(meta).get("seed_role", ""))
                except json.JSONDecodeError:
                    seed = ""
            low = text.lower()
            # Simple lexical heuristics for demo / weak labels
            if any(x in low for x in ["unsafe", "токсич", "kill", "attack"]):
                return labels_cfg[2], 0.55
            if "candidate" in seed or "borderline" in seed:
                return labels_cfg[0], 0.62
            if "safe_seed" in seed or "native_ru" in seed or "control" in seed:
                return labels_cfg[1], 0.72
            return labels_cfg[1], 0.66

        preds: list[str] = []
        confs: list[float] = []
        for _, row in out.iterrows():
            p, c = _predict(row)
            preds.append(p)
            confs.append(c)

        out["predicted_label"] = preds
        out["confidence"] = confs
        if self.include_predictions and "label" not in out.columns:
            out["label"] = out["predicted_label"]
        return out

    def generate_spec(self, df: pd.DataFrame, task: str | None = None) -> Path:
        task = task or self.annotation_cfg.get("task", "ru_fpbench_borderline_prompt_classification")
        self.paths.reports_dir.mkdir(parents=True, exist_ok=True)
        path = self.paths.reports_dir / "annotation_spec.md"
        labels = (self.cfg.get("project") or {}).get("labels") or []
        body = f"""# Annotation specification: {task}

## Modality
{self.modality}

## Label set
{labels}

## Instructions (summary)
Annotators assign one label per `text` row based on safety intent and lexical risk.

## Columns exported
- `uid`, `text`, `predicted_label`, `confidence`, `final_label` (to be filled during review)
"""
        path.write_text(body, encoding="utf-8")
        return path

    def check_quality(self, df: pd.DataFrame) -> dict[str, Any]:
        if "confidence" not in df.columns:
            return {"mean_confidence": None, "low_confidence_count": 0}
        conf = df["confidence"].astype(float)
        low = int((conf < self.confidence_threshold).sum())
        return {
            "mean_confidence": float(np.mean(conf)) if len(conf) else None,
            "low_confidence_count": low,
            "threshold": self.confidence_threshold,
        }

    def export_to_labelstudio(self, df: pd.DataFrame) -> Path:
        self.paths.labeled_dir.mkdir(parents=True, exist_ok=True)
        path = self.paths.labeled_dir / "labelstudio_import.json"
        tasks = []
        for _, row in df.iterrows():
            tasks.append(
                {
                    "data": {"text": str(row.get("text", "")), "uid": str(row.get("uid", ""))},
                    "predictions": [
                        {
                            "result": [],
                            "score": float(row.get("confidence", 0.0) or 0.0),
                        }
                    ],
                }
            )
        path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def build_review_queue(self, df: pd.DataFrame) -> pd.DataFrame | None:
        if "confidence" not in df.columns:
            return None
        low = df[df["confidence"].astype(float) < self.confidence_threshold].copy()
        if not len(low):
            return None
        cols = ["uid", "text", "predicted_label", "confidence"]
        cols = [c for c in cols if c in low.columns]
        return low[cols]

    def run(
        self,
        input_path: str | Path,
        *,
        output_parquet: str | Path | None = None,
    ) -> pd.DataFrame:
        input_path = Path(input_path)
        df = pd.read_parquet(input_path)
        _log.info("run input=%s rows=%s", input_path, len(df))
        _log.info("auto_label start")
        labeled = self.auto_label(df)
        if "predicted_label" in labeled.columns:
            dist = labeled["predicted_label"].astype(str).value_counts().to_dict()
            _log.info("predicted_label distribution=%s", dist)
        spec_path = self.generate_spec(labeled)
        ls_path = self.export_to_labelstudio(labeled)

        self.paths.labeled_dir.mkdir(parents=True, exist_ok=True)
        out_p = Path(output_parquet) if output_parquet else self.paths.labeled_dir / "auto_labeled.parquet"
        labeled.to_parquet(out_p, index=False)
        _log.info("saved auto_labeled path=%s", out_p)

        metrics = self.check_quality(labeled)
        _log.info(
            "quality mean_confidence=%s low_confidence_count=%s threshold=%s",
            metrics.get("mean_confidence"),
            metrics.get("low_confidence_count"),
            metrics.get("threshold"),
        )

        _log.info("wrote spec=%s labelstudio_import=%s", spec_path, ls_path)

        review = self.build_review_queue(labeled)
        rq_path = Path(self.hitl_cfg.get("review_queue_path", self.paths.labeled_dir / "review_queue.csv"))
        if review is not None and len(review):
            rq_path.parent.mkdir(parents=True, exist_ok=True)
            review.to_csv(rq_path, index=False)
            _log.info("wrote review_queue path=%s rows=%s", rq_path, len(review))
            low_ls = self.paths.labeled_dir / "labelstudio_low_confidence.json"
            low_ls.write_text(
                json.dumps(
                    [
                        {"data": {"text": str(r.get("text")), "uid": str(r.get("uid"))}}
                        for _, r in review.iterrows()
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            _log.info("wrote labelstudio_low_confidence path=%s", low_ls)
        else:
            _log.info("no low-confidence review queue (threshold=%s)", self.confidence_threshold)
        return labeled


__all__ = ["AnnotationAgent"]
