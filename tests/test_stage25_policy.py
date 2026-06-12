"""Stage 2.5 policy YAML validation."""

from pathlib import Path

import pytest

from rufp_stage25.policy.loader import Stage25ConfigError, load_stage25_policy
from rufp_stage25.policy.schema import Stage25Policy


def test_load_default_repo_config():
    repo = Path(__file__).resolve().parents[1]
    p = load_stage25_policy(repo / "configs" / "stage25.yaml")
    assert isinstance(p, Stage25Policy)
    assert p.revalidation.run_probes is True


def test_invalid_strategy_raises(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
version: 1
repair_strategies:
  allowed: [not_a_real_strategy]
""",
        encoding="utf-8",
    )
    with pytest.raises(Stage25ConfigError) as ei:
        load_stage25_policy(bad)
    assert "unknown" in str(ei.value).lower() or "repair_strategies" in str(ei.value)


def test_probe_promotion_inconsistent_raises(tmp_path: Path):
    bad = tmp_path / "bad2.yaml"
    bad.write_text(
        """
version: 1
repair_strategies:
  allowed: [strengthen_borderline_surface]
revalidation:
  run_probes: false
aggregator:
  probe_required_for_promotion: true
""",
        encoding="utf-8",
    )
    with pytest.raises(Stage25ConfigError):
        load_stage25_policy(bad)
