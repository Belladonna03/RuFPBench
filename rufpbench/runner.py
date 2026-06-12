from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import AppConfig
from .evolution import EvolutionaryPipeline


# Backward-compatible public API. `evolutionary` remains available, while
# `cascade_mining` is the new cost-optimized mode for large RuFP funnels.
def run_pipeline(cfg: AppConfig, run_dir: Path, mock: bool = False) -> dict[str, Any]:
    mode = str(cfg.run.mode or "evolutionary").strip().lower().replace("-", "_")
    if mode in {"quality", "quality_mining", "cascade_mining_quality", "cascade_quality", "cascade_mining_quality"}:
        from .quality import QualityMiningPipeline

        return QualityMiningPipeline(cfg=cfg, run_dir=run_dir, mock=mock).run()
    if mode in {"cascade", "cascade_mining", "cascade-mining"}:
        from .cascade import CascadeMiningPipeline

        return CascadeMiningPipeline(cfg=cfg, run_dir=run_dir, mock=mock).run()
    if mode in {"evolutionary", "legacy"}:
        return EvolutionaryPipeline(cfg=cfg, run_dir=run_dir, mock=mock).run()
    raise ValueError(f"Unknown RuFPBench run.mode={cfg.run.mode!r}. Use 'cascade_mining_quality', 'cascade_mining' or 'evolutionary'.")
