"""Smoke test for Stage 3 pipeline (dry-run)."""

import asyncio
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "configs" / "stage3.yaml"


@pytest.mark.skipif(not CFG.is_file(), reason="configs/stage3.yaml missing")
def test_stage3_pipeline_dry_run(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    from rufp_stage3.pipeline.orchestrator import run_stage3_pipeline

    out = asyncio.run(
        run_stage3_pipeline(
            "s3_smoke_test",
            config_path=str(CFG),
            dry_run=True,
            resume=False,
        )
    )
    assert out.pool_size >= 0
    base = REPO / "artifacts" / "stage3" / "s3_smoke_test"
    assert (base / "stage3_summary.json").is_file()
    assert (base / "stage3_candidate_pool.jsonl").is_file()


def test_stage3_invalid_config(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: 1\ninputs: {}\n", encoding="utf-8")
    from rufp_stage3.policy.loader import Stage3ConfigError, load_stage3_policy

    with pytest.raises(Stage3ConfigError):
        load_stage3_policy(bad)
