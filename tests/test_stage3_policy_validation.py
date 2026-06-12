"""Stage 3 policy: pydantic validation and clear errors."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rufp_stage3.policy.loader import Stage3ConfigError, load_stage3_policy
from rufp_stage3.policy.schema import ExportFormatsConfig, FilenameMap, InputPaths, Stage3Policy


def test_split_fractions_must_not_exceed_one(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        """
version: 1
inputs: { stage2_dir: x, stage25_dir: null }
split:
  dev_fraction: 0.8
  test_fraction: 0.4
""",
        encoding="utf-8",
    )
    with pytest.raises(Stage3ConfigError) as ei:
        load_stage3_policy(p)
    assert "split" in str(ei.value).lower() or "1.0" in str(ei.value)


def test_export_formats_need_at_least_one_tabular():
    with pytest.raises(ValidationError):
        Stage3Policy(
            inputs=InputPaths(stage2_dir="a", stage25_dir=None),
            filenames=FilenameMap(),
            export_formats=ExportFormatsConfig(write_jsonl=False, write_csv=False),
        )


def test_target_slice_requires_filter():
    from rufp_stage3.policy.schema import TargetSliceDefinition

    with pytest.raises(ValidationError):
        TargetSliceDefinition(name="empty")
