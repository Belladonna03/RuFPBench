from pathlib import Path

from rufpbench.config import load_config
from rufpbench.runner import run_pipeline


def test_mock_pipeline(tmp_path: Path):
    cfg = load_config(Path(__file__).parents[1] / "configs" / "default.yaml")
    cfg.run.target_raw_prompts = 12
    cfg.run.min_borderline_false_refusals = 2
    cfg.run.max_rounds = 2
    cfg.run.raw_batch_size = 12
    cfg.run.seed_batch_size = 3
    cfg.run.max_extra_raw_prompts = 12
    cfg.models.target_models = ["oss", "GigaChat-2-Max", "GigaChat-3-Ultra"]
    cfg.models.safety_judge_models = ["oss", "GigaChat-2-Max"]
    stats = run_pipeline(cfg, run_dir=tmp_path / "run", mock=True)
    assert stats["final_records"] > 0
    assert (tmp_path / "run" / "final" / "all_labeled.jsonl").exists()
    assert (tmp_path / "run" / "final" / "safe_refused_borderline.jsonl").exists()
