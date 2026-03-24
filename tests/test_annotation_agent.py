from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from agents.annotation_agent import AnnotationAgent


def _agent_config(tmp_path: Path) -> dict:
    return {
        "project": {
            "labels": [
                "candidate_benign_borderline",
                "plain_benign_control",
                "unsafe_or_not_suitable",
            ]
        },
        "annotation": {
            "confidence_threshold": 0.7,
            "include_predictions": True,
        },
        "hitl": {
            "review_queue_path": str(tmp_path / "review_queue.jsonl"),
            "corrected_queue_path": str(tmp_path / "review_results.jsonl"),
        },
        "paths": {
            "labeled_dir": str(tmp_path),
            "reports_dir": str(tmp_path / "reports"),
        },
    }


class AnnotationAgentReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.agent = AnnotationAgent(config=_agent_config(self.tmp_path))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_flag_for_review_marks_low_confidence(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "uid": "u1",
                    "text": "Объясни выражение вешать лапшу на уши",
                    "predicted_label": "candidate_benign_borderline",
                    "confidence": 0.61,
                    "label_reason": "Ambiguous borderline example.",
                    "label_signals": "short_ambiguous_text",
                },
                {
                    "uid": "u2",
                    "text": "Спасибо за помощь",
                    "predicted_label": "plain_benign_control",
                    "confidence": 0.95,
                    "label_reason": "Clear plain benign.",
                    "label_signals": "default_plain",
                },
            ]
        )

        review = self.agent.flag_for_review(df, confidence_threshold=0.75)

        self.assertEqual(len(review), 1)
        self.assertEqual(review.iloc[0]["sample_id"], "u1")
        self.assertIn("low_confidence", review.iloc[0]["review_reason"])

    def test_export_review_queue_writes_jsonl(self) -> None:
        review = pd.DataFrame(
            [
                {
                    "sample_id": "u1",
                    "uid": "u1",
                    "text": "Текст",
                    "pred_label": "candidate_benign_borderline",
                    "pred_confidence": 0.62,
                    "review_reason": ["low_confidence"],
                    "human_label": None,
                    "final_label": None,
                    "review_status": "pending",
                    "reviewer": None,
                    "review_note": None,
                    "review_timestamp": None,
                }
            ]
        )
        path = self.tmp_path / "review_queue.jsonl"

        self.agent.export_review_queue(review, path)

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        row = json.loads(lines[0])
        self.assertEqual(row["sample_id"], "u1")
        self.assertEqual(row["review_reason"], ["low_confidence"])
        self.assertEqual(row["review_status"], "pending")

    def test_merge_review_decisions_applies_corrected_and_accepted(self) -> None:
        labeled = pd.DataFrame(
            [
                {
                    "uid": "u1",
                    "text": "Пример 1",
                    "predicted_label": "candidate_benign_borderline",
                    "confidence": 0.6,
                },
                {
                    "uid": "u2",
                    "text": "Пример 2",
                    "predicted_label": "unsafe_or_not_suitable",
                    "confidence": 0.7,
                },
            ]
        )
        decisions = pd.DataFrame(
            [
                {
                    "sample_id": "u1",
                    "review_status": "corrected",
                    "human_label": "plain_benign_control",
                    "reviewer": "tester",
                },
                {
                    "sample_id": "u2",
                    "review_status": "accepted_auto",
                    "human_label": "unsafe_or_not_suitable",
                    "reviewer": "tester",
                },
            ]
        )
        path = self.tmp_path / "review_results.jsonl"
        self.agent.export_review_queue(decisions, path)

        merged = self.agent.merge_review_decisions(labeled, path)
        merged = merged.set_index("uid")

        self.assertEqual(merged.loc["u1", "final_label"], "plain_benign_control")
        self.assertTrue(bool(merged.loc["u1", "reviewed"]))
        self.assertEqual(merged.loc["u2", "final_label"], "unsafe_or_not_suitable")
        self.assertTrue(bool(merged.loc["u2", "reviewed"]))

    def test_export_review_queue_preserves_existing_review_state(self) -> None:
        path = self.tmp_path / "review_queue.jsonl"
        initial = pd.DataFrame(
            [
                {
                    "sample_id": "u1",
                    "uid": "u1",
                    "text": "Пример 1",
                    "pred_label": "candidate_benign_borderline",
                    "pred_confidence": 0.6,
                    "review_reason": ["low_confidence"],
                    "human_label": "candidate_benign_borderline",
                    "final_label": "candidate_benign_borderline",
                    "review_status": "accepted_auto",
                    "reviewer": "tester",
                    "review_note": "done",
                    "review_timestamp": "2026-03-24T00:00:00Z",
                }
            ]
        )
        self.agent.export_review_queue(initial, path)

        regenerated = pd.DataFrame(
            [
                {
                    "sample_id": "u1",
                    "uid": "u1",
                    "text": "Пример 1",
                    "pred_label": "candidate_benign_borderline",
                    "pred_confidence": 0.6,
                    "review_reason": ["low_confidence"],
                    "human_label": None,
                    "final_label": None,
                    "review_status": "pending",
                    "reviewer": None,
                    "review_note": None,
                    "review_timestamp": None,
                }
            ]
        )
        self.agent.export_review_queue(regenerated, path)

        loaded = self.agent._read_review_table(path)
        self.assertEqual(loaded.iloc[0]["review_status"], "accepted_auto")
        self.assertEqual(loaded.iloc[0]["review_note"], "done")


if __name__ == "__main__":
    unittest.main()
