from collections import Counter
from pathlib import Path

from rufpbench.config import load_config
from rufpbench.sampler import GenerationJobSampler
from rufpbench.seed_bank import load_seed_bank
from rufpbench.tactics import load_compatibility, load_tactics


def test_sampler_guarantees_control_floor_in_new_generation_batch():
    cfg = load_config(Path(__file__).parents[1] / "configs" / "default.yaml")
    cfg.run.jobs_output_count = 2
    cfg.run.safe_control_ratio = 0.10
    cfg.run.unsafe_control_ratio = 0.15
    sampler = GenerationJobSampler(
        cfg=cfg,
        seed_bank=load_seed_bank(cfg),
        tactics=load_tactics(cfg),
        compatibility=load_compatibility(cfg),
    )

    jobs = sampler.sample_jobs(
        round_id=1,
        target_candidates=20,
        pending_jobs=[],
        accepted_counts=Counter(),
        recipe_stats={},
    )
    counts = Counter()
    for job in jobs:
        counts[job.target_distribution] += job.output_count

    assert counts["benign_control"] >= 2
    assert counts["unsafe_control"] >= 3
