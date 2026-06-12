from pathlib import Path

from rufpbench.config import load_config
from rufpbench.llm import LLMOptions, LLMRouter
from rufpbench.runner import run_pipeline


ROOT = Path(__file__).parents[1]


def test_proxyapi_cheap_config_uses_main_cascade_mode_and_proxy_steps():
    cfg = load_config(ROOT / "configs" / "proxyapi_cheap.yaml")

    assert cfg.run.mode == "cascade_mining"
    assert cfg.run.target_raw_prompts >= 800
    assert cfg.llm.pipeline["generate_candidates_job"]["provider"] == "proxyapi"
    assert cfg.llm.pipeline["target_response_scout"]["max_tokens"] <= 192
    assert cfg.llm.pipeline["target_response_final"]["max_tokens"] <= 256
    assert "openai/gpt-5.4-nano" in cfg.llm.pipeline["target_response_scout"]["models"]


def test_proxyapi_prefixed_models_route_to_proxyapi_provider():
    cfg = load_config(ROOT / "configs" / "proxyapi_cheap.yaml")
    router = LLMRouter(cfg, mock=True)

    for model in ["openai/gpt-5.4-nano", "openrouter/openai/gpt-oss-20b"]:
        provider, options = router.resolve("target_response_scout", LLMOptions(model=model))
        assert provider == "proxyapi"
        assert options.model == model
        assert options.max_tokens == 192
        assert options.extra["_max_tokens_cap"] == 192
        assert options.extra["_min_max_tokens"] == 0


def test_proxyapi_cheap_config_runs_through_main_pipeline_in_mock_mode(tmp_path: Path):
    cfg = load_config(ROOT / "configs" / "proxyapi_cheap.yaml")
    cfg.run.target_raw_prompts = 18
    cfg.run.min_borderline_false_refusals = 1
    cfg.run.max_rounds = 1
    cfg.run.raw_batch_size = 18
    cfg.run.topup_raw_batch_size = 18
    cfg.run.max_extra_raw_prompts = 0
    cfg.run.jobs_output_count = 3
    cfg.run.seed_batch_size = 3
    cfg.run.max_workers = 2
    cfg.qc.category_soft_cap_fraction = 1.0

    stats = run_pipeline(cfg, run_dir=tmp_path / "proxy_main", mock=True)

    assert stats["mode"] == "cascade_mining"
    assert stats["raw_generated"] > 0
    assert stats["scout_target_calls"] >= 0
    assert (tmp_path / "proxy_main" / "config.effective.json").exists()
