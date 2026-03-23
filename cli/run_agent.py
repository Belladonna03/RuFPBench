from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root must be on sys.path before importing local packages (agents, pipeline, shared).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.active_learning_agent import ActiveLearningAgent
from agents.annotation_agent import AnnotationAgent
from agents.data_collection_agent import DataCollectionAgent
from agents.data_quality_agent import DataQualityAgent
from agents.rewrite_agent import BorderlineRewriteAgent

from pipeline.io import read_table, write_parquet
from shared.config import load_config
from shared.logging_utils import configure_logging, get_logger

ROOT = _ROOT

_log = get_logger("agent_runner")

AGENT_ALIASES = {
    "collection": "data_collection",
    "data_collection": "data_collection",
    "rewrite": "rewrite",
    "borderline_rewrite": "rewrite",
    "quality": "quality",
    "data_quality": "quality",
    "annotation": "annotation",
    "al": "active_learning",
    "active_learning": "active_learning",
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run a single RuFPBench agent step (debug).")
    p.add_argument("--agent", required=True, help="Agent name (collection, rewrite, quality, annotation, al)")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--input", default=None, help="Input table (.parquet or .csv)")
    p.add_argument("--output", default=None, help="Output path (default depends on agent)")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = p.parse_args(argv)
    configure_logging(args.log_level)

    name = AGENT_ALIASES.get(args.agent.lower().strip(), args.agent.lower().strip())
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path) if cfg_path.exists() else {}

    _log.info("agent=%s config=%s", name, cfg_path)

    if name == "data_collection":
        _log.info("starting data_collection (defaults: raw output under config collection.output_dir)")
        out = DataCollectionAgent(cfg_path).run()
        _log.info("agent=data_collection finished rows=%s", len(out))
        return 0

    if name == "rewrite":
        inp = Path(args.input or ROOT / "data/raw/merged_raw.parquet")
        out = Path(args.output or ROOT / "data/interim/rewrite.parquet")
        _log.info("agent=rewrite input=%s output=%s", inp, out)
        BorderlineRewriteAgent(cfg_path if cfg_path.exists() else cfg).run(inp, out)
        rw_rows = len(read_table(out)) if out.exists() else 0
        _log.info("agent=rewrite finished output=%s rows=%s", out, rw_rows)
        return 0

    if name == "quality":
        inp = Path(args.input or ROOT / "data/interim/rewrite.parquet")
        out = Path(args.output or ROOT / "data/interim/clean.parquet")
        _log.info("agent=quality input=%s output=%s", inp, out)
        qcfg = dict(cfg.get("quality") or {})
        dq = DataQualityAgent(qcfg)
        df = read_table(inp)
        dq.detect_issues(df)
        strat = qcfg.get("strategy") or {
            "missing": "fill",
            "duplicates": "drop",
            "outliers": "drop_iqr",
        }
        clean = dq.fix(df, strategy=strat)
        write_parquet(clean, out)
        _log.info("agent=quality finished output=%s rows=%s", out, len(clean))
        return 0

    if name == "annotation":
        inp = Path(args.input or ROOT / "data/interim/clean.parquet")
        out = args.output
        _log.info("agent=annotation input=%s", inp)
        ann = AnnotationAgent(modality="text", config=cfg_path if cfg_path.exists() else cfg)
        ann.run(inp, output_parquet=out)
        hitl = cfg.get("hitl") or {}
        rq_path = Path(hitl.get("review_queue_path", ROOT / "data/labeled/review_queue.csv"))
        rq_rows = len(read_table(rq_path)) if rq_path.exists() else 0
        _log.info(
            "agent=annotation finished auto_labeled=%s review_queue=%s review_queue_rows=%s",
            out or (ROOT / "data/labeled/auto_labeled.parquet"),
            rq_path,
            rq_rows,
        )
        return 0

    if name == "active_learning":
        inp = Path(args.input or ROOT / "data/labeled/final_dataset.parquet")
        _log.info("agent=active_learning input=%s", inp)
        df = read_table(inp)
        agent = ActiveLearningAgent(config=cfg_path if cfg_path.exists() else cfg)
        if "final_label" in df.columns:
            df = df.rename(columns={"final_label": "label"})
        from sklearn.model_selection import train_test_split

        al_cfg = cfg.get("active_learning") or {}
        test_size = float(al_cfg.get("test_size", 0.2))
        rs = int(al_cfg.get("random_state", 42))
        train_pool, test_df = train_test_split(df, test_size=test_size, random_state=rs)
        start_n = min(int(al_cfg.get("start_n", 50)), max(1, len(train_pool) - 1))
        labeled_df, pool_df = train_test_split(train_pool, train_size=start_n, random_state=rs)
        strategies = al_cfg.get("strategies") or ["entropy", "random"]
        histories = {}
        for strat in strategies:
            histories[str(strat)] = agent.run_cycle(
                labeled_df=labeled_df.copy(),
                pool_df=pool_df.copy(),
                test_df=test_df.copy(),
                strategy=str(strat),
                n_iterations=int(al_cfg.get("n_iterations", 5)),
                batch_size=int(al_cfg.get("batch_size", 20)),
            )
        metric = str(al_cfg.get("metric", "f1"))
        outp = Path(args.output or ROOT / "reports/learning_curve.png")
        agent.report(
            histories,
            metric=metric,
            output_path=outp,
            title="Active learning (debug run)",
        )
        _log.info("agent=active_learning finished learning_curve=%s", outp)
        return 0

    _log.error("unknown agent name=%s", args.agent)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
