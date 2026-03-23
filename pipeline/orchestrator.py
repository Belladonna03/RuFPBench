from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from agents.active_learning_agent import ActiveLearningAgent
from agents.annotation_agent import AnnotationAgent
from agents.data_collection_agent import DataCollectionAgent
from agents.data_quality_agent import DataQualityAgent
from agents.rewrite_agent import BorderlineRewriteAgent

from pipeline.hitl import merge_hitl_labels, should_stop_for_hitl
from pipeline.io import copy_parquet, read_table, write_parquet
from pipeline.reporting import write_final_report
from pipeline.training import run_training_from_config
from shared.config import load_config
from shared.logging_utils import configure_logging, get_logger
from shared.paths import ProjectPaths

_log = get_logger("pipeline")


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_pipeline(config_path: str | Path) -> int:
    root = _root()
    cfg = load_config(config_path)
    paths = ProjectPaths.from_config(root, cfg)
    paths.ensure()

    raw_merged = paths.raw_dir / "merged_raw.parquet"
    interim_rewrite = paths.interim_dir / "rewrite.parquet"
    interim_clean = paths.interim_dir / "clean.parquet"
    labeled_auto = paths.labeled_dir / "auto_labeled.parquet"
    final_ds = Path((cfg.get("reporting") or {}).get("final_dataset_path", paths.labeled_dir / "final_dataset.parquet"))

    artifacts: dict[str, Any] = {}

    _log.info("started config=%s", config_path)

    # 1) Collection
    _log.info("step=collection started")
    collector = DataCollectionAgent(config_path)
    collector.run()
    mr_rows = len(read_table(raw_merged)) if raw_merged.exists() else 0
    _log.info("step=collection finished rows=%s output=%s", mr_rows, raw_merged)
    artifacts["merged_raw"] = str(raw_merged)

    # 2) Rewrite (optional)
    rewrite_cfg = cfg.get("rewrite") or {}
    if bool(rewrite_cfg.get("enabled", False)):
        _log.info("step=rewrite started")
        rewriter = BorderlineRewriteAgent(cfg)
        rewriter.run(raw_merged, interim_rewrite)
        rw_rows = len(read_table(interim_rewrite)) if interim_rewrite.exists() else 0
        _log.info("step=rewrite finished rows=%s output=%s", rw_rows, interim_rewrite)
    else:
        _log.warning("step=rewrite skipped reason=rewrite.disabled (copying merged raw to interim)")
        paths.interim_dir.mkdir(parents=True, exist_ok=True)
        copy_parquet(raw_merged, interim_rewrite)

    # 3) Quality
    _log.info("step=quality started input=%s", interim_rewrite)
    qcfg = dict(cfg.get("quality") or {})
    dq = DataQualityAgent(qcfg)
    df_rw = read_table(interim_rewrite)
    issues = dq.detect_issues(df_rw)
    (paths.reports_dir / "quality_prescan.json").write_text(
        json.dumps(issues, indent=2, default=str),
        encoding="utf-8",
    )
    strat = qcfg.get("strategy") or {
        "missing": "fill",
        "duplicates": "drop",
        "outliers": "drop_iqr",
    }
    clean = dq.fix(df_rw, strategy=strat)
    write_parquet(clean, interim_clean)
    _log.info(
        "step=quality finished rows=%s output=%s report=%s",
        len(clean),
        interim_clean,
        paths.reports_dir / "quality_prescan.json",
    )
    artifacts["clean"] = str(interim_clean)

    # 4) Annotation
    _log.info("step=annotation started input=%s", interim_clean)
    ann = AnnotationAgent(modality="text", config=config_path)
    labeled = ann.run(interim_clean, output_parquet=labeled_auto)
    rq_path = Path(cfg.get("hitl") or {}).get("review_queue_path", paths.labeled_dir / "review_queue.csv")
    rq_rows = 0
    if rq_path.exists():
        try:
            rq_rows = len(read_table(rq_path))
        except Exception:
            rq_rows = -1
    _log.info(
        "step=annotation finished auto_labeled=%s review_queue=%s review_queue_rows=%s",
        labeled_auto,
        rq_path,
        rq_rows,
    )
    artifacts["auto_labeled"] = str(labeled_auto)

    # 5) HITL gate
    stop, reason = should_stop_for_hitl(cfg, root)
    if stop:
        corrected = Path((cfg.get("hitl") or {}).get("corrected_queue_path", root / "data/labeled/review_queue_corrected.csv"))
        _log.warning(
            "stopped at HITL gate: %s (create or complete %s then re-run)",
            reason.strip(),
            corrected,
        )
        rq = (cfg.get("hitl") or {}).get("review_queue_path", "data/labeled/review_queue.csv")
        root_rq = Path(rq)
        if root_rq.exists():
            shutil.copy2(root_rq, root / "review_queue.csv")
        return 2

    # 6) Merge labels → final dataset
    _log.info("step=merge_labels started")
    merged = merge_hitl_labels(labeled, cfg, root)
    write_parquet(merged, final_ds)
    csv_path = (cfg.get("reporting") or {}).get("final_dataset_csv")
    if csv_path:
        p = Path(csv_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(p, index=False)
        _log.info("wrote final_dataset_csv=%s", p)
    _log.info("step=merge_labels finished rows=%s output=%s", len(merged), final_ds)
    artifacts["final_dataset"] = str(final_ds)

    # 7) AL selection batch (optional)
    al_sel = cfg.get("al_select") or {}
    if al_sel.get("enabled"):
        _log.info("step=al_select started")
        agent = ActiveLearningAgent(config=cfg)
        pool_full = merged.copy()
        start_n = min(int(cfg.get("active_learning", {}).get("start_n", 50)), len(pool_full))
        labeled_seed = pool_full.sample(n=start_n, random_state=int(cfg.get("active_learning", {}).get("random_state", 42)))
        remaining = pool_full.drop(index=labeled_seed.index)
        if "final_label" in labeled_seed.columns:
            labeled_seed = labeled_seed.rename(columns={"final_label": "label"})
        elif "label" not in labeled_seed.columns:
            raise ValueError("al_select requires final_label or label on merged dataset")
        batch = agent.select_batch(
            pool_df=remaining,
            labeled_df=labeled_seed,
            strategy=str(al_sel.get("strategy", "entropy")),
            batch_size=int(al_sel.get("batch_size", 100)),
        )
        out_csv = Path(al_sel.get("output_csv", paths.labeled_dir / "al_candidates.csv"))
        out_ls = al_sel.get("output_labelstudio")
        agent.export_candidates(
            batch,
            output_csv=out_csv,
            output_labelstudio=Path(out_ls) if out_ls else None,
        )
        _log.info(
            "step=al_select finished candidates=%s rows=%s labelstudio=%s",
            out_csv,
            len(batch),
            out_ls or "(none)",
        )
        artifacts["al_candidates"] = str(out_csv)
    else:
        _log.warning("step=al_select skipped reason=al_select.disabled")

    # 8) Active learning curves (optional)
    al_cfg = cfg.get("active_learning") or {}
    if al_cfg.get("enabled"):
        _log.info("step=active_learning_curves started")
        al_agent = ActiveLearningAgent(config=cfg)
        df = read_table(final_ds)
        if "final_label" in df.columns:
            df = df.rename(columns={"final_label": "label"})
        elif "label" not in df.columns:
            _log.warning("step=active_learning_curves skipped reason=no_label_column")
        else:
            test_size = float(al_cfg.get("test_size", 0.2))
            rs = int(al_cfg.get("random_state", 42))
            from sklearn.model_selection import train_test_split

            train_pool, test_df = train_test_split(df, test_size=test_size, random_state=rs)
            if len(train_pool) < 3:
                _log.warning("step=active_learning_curves skipped reason=train_pool_too_small n=%s", len(train_pool))
            else:
                start_n = min(int(al_cfg.get("start_n", 50)), max(1, len(train_pool) - 1))
                labeled_df, pool_df = train_test_split(
                    train_pool,
                    train_size=start_n,
                    random_state=rs,
                )
                strategies = al_cfg.get("strategies") or ["entropy", "random"]
                histories = {}
                for strat in strategies:
                    histories[str(strat)] = al_agent.run_cycle(
                        labeled_df=labeled_df.copy(),
                        pool_df=pool_df.copy(),
                        test_df=test_df.copy(),
                        strategy=str(strat),
                        n_iterations=int(al_cfg.get("n_iterations", 5)),
                        batch_size=int(al_cfg.get("batch_size", 20)),
                    )
                metric = str(al_cfg.get("metric", "f1"))
                curve_path = paths.reports_dir / "learning_curve.png"
                al_agent.report(
                    histories,
                    metric=metric,
                    output_path=curve_path,
                    title="Active learning",
                )
                _log.info("step=active_learning_curves finished output=%s", curve_path)
                artifacts["learning_curve"] = str(curve_path)
    else:
        _log.warning("step=active_learning_curves skipped reason=active_learning.disabled")

    # 9) Training
    train_out = run_training_from_config(merged, cfg, paths.reports_dir)
    if train_out:
        _log.info("step=training finished model=%s", train_out)
        artifacts["model"] = str(train_out)

    # 10) Report
    write_final_report(cfg=cfg, reports_dir=paths.reports_dir, artifacts=artifacts)

    # Data card (short)
    dc = paths.labeled_dir / "DATA_CARD.md"
    dc.write_text(
        "# Dataset card\n\nGenerated by RuFPBench pipeline. See `reports/final_report.md`.\n",
        encoding="utf-8",
    )

    _log.info("finished ok artifacts=%s", ", ".join(f"{k}={v}" for k, v in artifacts.items()))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="RuFPBench end-to-end pipeline")
    p.add_argument("--config", required=True, help="Path to config.yaml")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = p.parse_args(argv)
    configure_logging(args.log_level)
    return run_pipeline(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
