from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from joblib import dump as joblib_dump

from agents.data_collection_agent import DataCollectionAgent
from agents.post_collection_appendix import RuFPBenchAppendixRewriter
from agents.data_quality_agent import DataQualityAgent
from agents.annotation_agent import AnnotationAgent
from agents.al_agent import ActiveLearningAgent


# ----------------------------
# IO helpers
# ----------------------------

def read_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def to_md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for r in rows:
        vals = []
        for c in columns:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


# ----------------------------
# HITL helpers
# ----------------------------

def apply_human_corrections(
    df_labeled: pd.DataFrame,
    corrected_path: Path,
    uid_col: str = "uid",
    label_col: str = "label",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Keep auto labels in `label` and add `human_label` (from corrected file).
    Also compute `label_final` = human_label if provided else label.

    corrected file expected columns:
      - uid
      - label  (corrected label)
    """
    corrected = pd.read_csv(corrected_path)

    if uid_col not in corrected.columns:
        raise ValueError(f"Corrected file must contain column '{uid_col}'. Columns: {list(corrected.columns)}")
    if label_col not in corrected.columns:
        # allow human_label as alias
        if "human_label" in corrected.columns:
            corrected[label_col] = corrected["human_label"]
        else:
            raise ValueError(
                f"Corrected file must contain '{label_col}' (or 'human_label'). Columns: {list(corrected.columns)}"
            )

    corrected = corrected[[uid_col, label_col]].dropna()
    corrected[uid_col] = corrected[uid_col].astype(str)
    corrected[label_col] = corrected[label_col].astype(str)

    mapping = dict(zip(corrected[uid_col].tolist(), corrected[label_col].tolist()))

    out = df_labeled.copy()
    out[uid_col] = out[uid_col].astype(str)

    # keep auto labels
    out["label_auto"] = out[label_col].astype(str)

    # build human_label
    out["human_label"] = out[uid_col].map(mapping)
    n_human = int(out["human_label"].notna().sum())

    # compute changes among corrected
    changed = 0
    if n_human:
        changed = int((out.loc[out["human_label"].notna(), "human_label"] != out.loc[out["human_label"].notna(), "label_auto"]).sum())

    # final label
    out["label_final"] = out["human_label"].where(out["human_label"].notna(), out["label_auto"])

    stats = {
        "corrected_path": corrected_path.as_posix(),
        "n_in_corrected_file": int(len(corrected)),
        "n_matched_in_df": n_human,
        "n_changed_labels": changed,
        "change_rate_among_corrected": float(changed / n_human) if n_human else 0.0,
    }
    return out, stats


# ----------------------------
# Training helpers
# ----------------------------

def train_final_model(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    label_col: str = "label_final",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train TF-IDF + LogisticRegression baseline (same as ActiveLearningAgent).
    Returns metrics and saves a sklearn pipeline outside of this function.
    """
    df_ = df.copy()
    df_[text_col] = df_[text_col].astype("string").fillna("").str.strip()
    df_ = df_[df_[text_col] != ""].copy()

    df_[label_col] = df_[label_col].astype("string").fillna("unknown")

    train_df, test_df = train_test_split(
        df_,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=df_[label_col] if df_[label_col].nunique() > 1 else None,
    )

    agent = ActiveLearningAgent(model="logreg", config={"text_col": text_col, "label_col": label_col, "random_state": random_state})
    agent.fit(train_df)

    X_test = test_df[text_col].astype("string").fillna("").tolist()
    y_true = test_df[label_col].astype("string").fillna("unknown").tolist()
    y_pred = agent.pipeline.predict(X_test)  # type: ignore

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro"))

    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "accuracy": acc,
        "f1_macro": f1,
        "classification_report": rep,
        "model": agent.pipeline,
    }


# ----------------------------
# Report writers
# ----------------------------

def render_quality_report(report: dict, comparison: dict) -> str:
    missing = report.get("missing", {})
    duplicates = report.get("duplicates")
    outliers = report.get("outliers", [])
    imbalance = report.get("imbalance", {})
    meta = report.get("meta", {})

    lines = []
    lines.append("# quality_report\n\n")
    lines.append(f"- generated_at: {meta.get('generated_at')}\n")
    lines.append(f"- n_rows: {meta.get('n_rows')}\n\n")

    req = missing.get("_required_rows", {})
    lines.append("## Missing values\n\n")
    lines.append(f"- missing required rows: {req.get('count', 0)} ({req.get('rate', 0):.4f})\n")
    # show top missing columns
    cols = [(k, v.get("count", 0), v.get("rate", 0)) for k, v in missing.items() if k != "_required_rows"]
    cols = sorted(cols, key=lambda x: x[1], reverse=True)[:10]
    if cols:
        rows = [{"column": c, "count": n, "rate": r} for c, n, r in cols]
        lines.append("\n" + to_md_table(rows, ["column", "count", "rate"]) + "\n\n")
    else:
        lines.append("- no missing columns detected (beyond optional).\n\n")

    lines.append("## Duplicates\n\n")
    lines.append(f"- duplicates count: {duplicates}\n\n")

    lines.append("## Outliers (IQR on numeric features incl. text length)\n\n")
    if outliers:
        rows = []
        for o in outliers:
            rows.append({
                "feature": o.get("feature"),
                "n_outliers": o.get("n_outliers"),
                "rate": o.get("outlier_rate"),
                "lower": o.get("lower"),
                "upper": o.get("upper"),
            })
        lines.append(to_md_table(rows, ["feature", "n_outliers", "rate", "lower", "upper"]) + "\n\n")
    else:
        lines.append("- no outliers detected.\n\n")

    lines.append("## Class imbalance\n\n")
    if isinstance(imbalance, dict) and imbalance.get("counts"):
        rows = [{"label": k, "count": v, "rate": imbalance.get("rates", {}).get(k, 0)} for k, v in imbalance["counts"].items()]
        lines.append(to_md_table(rows, ["label", "count", "rate"]) + "\n\n")
        lines.append(f"- majority/minority ratio: {imbalance.get('majority_to_minority_ratio')}\n\n")
    else:
        lines.append("- imbalance info not available.\n\n")

    lines.append("## Before vs After (comparison)\n\n")
    comp_rows = comparison.get("table", [])
    if comp_rows:
        lines.append(to_md_table(comp_rows, ["metric", "before", "after", "delta"]) + "\n")
    else:
        lines.append("- no comparison table.\n")

    return "".join(lines)


def render_annotation_report(metrics: dict, spec_path: str, ls_path: str, hitl_stats: dict) -> str:
    lines = []
    lines.append("# annotation_report\n\n")
    lines.append(f"- spec: `{spec_path}`\n")
    lines.append(f"- labelstudio import: `{ls_path}`\n\n")

    lines.append("## Auto-label quality\n\n")
    rows = [{"metric": k, "value": v} for k, v in metrics.items() if k not in {"label_dist", "label_rates"}]
    # show some key metrics first
    key_order = ["confidence_mean", "low_confidence_rate", "kappa", "agreement"]
    rows_sorted = sorted(rows, key=lambda r: key_order.index(r["metric"]) if r["metric"] in key_order else 999)
    lines.append(to_md_table(rows_sorted, ["metric", "value"]) + "\n\n")

    lines.append("## Label distribution\n\n")
    dist = metrics.get("label_dist", {})
    rates = metrics.get("label_rates", {})
    rows = [{"label": k, "count": v, "rate": rates.get(k, 0)} for k, v in dist.items()]
    lines.append(to_md_table(rows, ["label", "count", "rate"]) + "\n\n")

    lines.append("## HITL summary\n\n")
    if hitl_stats:
        rows = [{"metric": k, "value": v} for k, v in hitl_stats.items()]
        lines.append(to_md_table(rows, ["metric", "value"]) + "\n")
    else:
        lines.append("- no human corrections applied.\n")

    return "".join(lines)


def render_al_report(al_summary: dict) -> str:
    lines = []
    lines.append("# al_report\n\n")
    lines.append("## Setup\n\n")
    setup = al_summary.get("setup", {})
    rows = [{"param": k, "value": v} for k, v in setup.items()]
    lines.append(to_md_table(rows, ["param", "value"]) + "\n\n")

    lines.append("## Results\n\n")
    results = al_summary.get("results", {})
    for strat, info in results.items():
        lines.append(f"### {strat}\n\n")
        rows = [{"iteration": r["iteration"], "n_labeled": r["n_labeled"], "accuracy": r["accuracy"], "f1": r["f1"]} for r in info.get("history", [])]
        if rows:
            lines.append(to_md_table(rows, ["iteration", "n_labeled", "accuracy", "f1"]) + "\n\n")

    lines.append("## Label savings\n\n")
    savings = al_summary.get("label_savings", {})
    if savings:
        rows = [{"metric": k, "value": v} for k, v in savings.items()]
        lines.append(to_md_table(rows, ["metric", "value"]) + "\n")
    else:
        lines.append("- not computed.\n")

    lines.append("\n## Artifacts\n\n")
    lines.append(f"- learning curve: `{al_summary.get('curve_path')}`\n\n")
    return "".join(lines)


def render_train_report(train_metrics: dict, model_path: str) -> str:
    lines = []
    lines.append("# train_report\n\n")
    lines.append(f"- model_path: `{model_path}`\n\n")
    lines.append("## Metrics\n\n")
    rows = [
        {"metric": "n_train", "value": train_metrics.get("n_train")},
        {"metric": "n_test", "value": train_metrics.get("n_test")},
        {"metric": "accuracy", "value": train_metrics.get("accuracy")},
        {"metric": "f1_macro", "value": train_metrics.get("f1_macro")},
    ]
    lines.append(to_md_table(rows, ["metric", "value"]) + "\n\n")
    lines.append("## Per-class report\n\n")
    rep = train_metrics.get("classification_report", {})
    # show only per-class rows (exclude averages)
    class_rows = []
    for k, v in rep.items():
        if isinstance(v, dict) and k not in {"accuracy", "macro avg", "weighted avg"}:
            class_rows.append({"label": k, "precision": v.get("precision"), "recall": v.get("recall"), "f1": v.get("f1-score"), "support": v.get("support")})
    if class_rows:
        lines.append(to_md_table(class_rows, ["label", "precision", "recall", "f1", "support"]) + "\n")
    return "".join(lines)


def render_data_card(df: pd.DataFrame) -> str:
    # basic counts
    n = len(df)
    labels = df.get("label_final") if "label_final" in df.columns else df.get("label")
    dist = labels.value_counts().to_dict() if labels is not None else {}
    # sources
    src = df.get("source")
    src_dist = src.value_counts().to_dict() if src is not None else {}

    lines = []
    lines.append("# Data Card — RuFPBench-MVP (Course Project)\n\n")
    lines.append("## Summary\n\n")
    lines.append("This dataset was produced by an end-to-end pipeline (collection → cleaning → auto-labeling → HITL corrections → AL analysis).\n\n")
    lines.append("## Data\n\n")
    lines.append(f"- rows: {n}\n")
    lines.append(f"- modality: text\n")
    lines.append("- label schema: `candidate_benign_borderline`, `plain_benign_control`, `unsafe_or_not_suitable`\n\n")

    lines.append("## Label distribution\n\n")
    rows = [{"label": k, "count": v} for k, v in dist.items()]
    if rows:
        lines.append(to_md_table(rows, ["label", "count"]) + "\n\n")

    lines.append("## Sources\n\n")
    rows = [{"source": k, "count": v} for k, v in src_dist.items()]
    if rows:
        lines.append(to_md_table(rows, ["source", "count"]) + "\n\n")

    lines.append("## Annotation\n\n")
    lines.append("- `label_auto`: automatic label from AnnotationAgent\n")
    lines.append("- `human_label`: human corrections for low-confidence subset (if provided)\n")
    lines.append("- `label_final`: final label used for training (human label if available, else auto)\n\n")

    lines.append("## Intended use\n\n")
    lines.append("Educational project / ML portfolio: studying borderline benign prompts that may trigger over-refusal in safety systems.\n\n")

    lines.append("## Limitations\n\n")
    lines.append("- This is an MVP; labels represent *candidates* and depend on heuristics + limited human review.\n")
    lines.append("- The dataset may contain lexical triggers; do not use as a harmful instruction corpus.\n\n")

    lines.append("## Ethical considerations\n\n")
    lines.append("- Donor toxicity sources are transformed/filtered; still, treat data carefully.\n")
    lines.append("- Avoid deploying any model trained on this dataset as a safety classifier without further evaluation.\n")
    return "".join(lines)


def render_final_report(summary: dict) -> str:
    lines = []
    lines.append("# Final Report — RuFPBench-MVP Data Pipeline\n\n")

    lines.append("## 1. Task and dataset\n\n")
    ds = summary.get("dataset", {})
    rows = [{"metric": k, "value": v} for k, v in ds.items()]
    lines.append(to_md_table(rows, ["metric", "value"]) + "\n\n")

    lines.append("## 2. What each agent did\n\n")
    lines.append("- **DataCollectionAgent**: collected seed corpora from 2+ sources and mapped them to a unified schema.\n")
    lines.append("- **(Optional) Appendix step**: rewrote unsafe donors into safe-but-borderline candidates (configurable).\n")
    lines.append("- **DataQualityAgent**: detected issues (missing/duplicates/outliers/imbalance) and applied cleaning strategy.\n")
    lines.append("- **AnnotationAgent**: auto-labeled into 3 classes, generated annotation spec, and exported to Label Studio.\n")
    lines.append("- **ActiveLearningAgent**: compared entropy vs random selection and produced learning curves.\n\n")

    lines.append("## 3. HITL point\n\n")
    hitl = summary.get("hitl", {})
    if hitl:
        rows = [{"metric": k, "value": v} for k, v in hitl.items()]
        lines.append(to_md_table(rows, ["metric", "value"]) + "\n\n")
    else:
        lines.append("- HITL not applied (no corrected file provided).\n\n")

    lines.append("## 4. Metrics\n\n")
    m = summary.get("metrics", {})
    for name, blob in m.items():
        lines.append(f"### {name}\n\n")
        if isinstance(blob, dict):
            rows = [{"metric": k, "value": v} for k, v in blob.items() if k not in {"classification_report"}]
            if rows:
                lines.append(to_md_table(rows, ["metric", "value"]) + "\n\n")
        else:
            lines.append(f"{blob}\n\n")

    lines.append("## 5. Retrospective\n\n")
    lines.append("- What worked: unified schema, modular appendix, and HITL queue made the pipeline reproducible.\n")
    lines.append("- What didn't: direct translation of English benign seeds does not always preserve Russian ambiguity.\n")
    lines.append("- Next steps: richer Russian-native seeds; better rewrite strategies; larger human-labeled slice; multi-model refusal evaluation.\n")
    return "".join(lines)


# ----------------------------
# Pipeline
# ----------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--human-mode", default=None, choices=["stop_if_missing", "accept_auto"], help="Override hitl.human_mode")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # dirs
    data_raw_dir = Path(cfg.get("output_dir", "data/raw"))
    data_interim_dir = Path("data/interim")
    data_labeled_dir = Path("data/labeled")
    models_dir = Path("models")
    reports_dir = Path("reports")
    for d in [data_raw_dir, data_interim_dir, data_labeled_dir, models_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {"dataset": {}, "hitl": {}, "metrics": {}}

    # Step 1: Collection
    collector = DataCollectionAgent(config=str(cfg_path))
    df_raw = collector.run()
    raw_path = data_raw_dir / "merged_raw.parquet"
    if raw_path.exists():
        df_raw = read_df(raw_path)

    # Step 1.5: Optional appendix rewrite
    appendix_cfg = cfg.get("appendix_rewrite", {}) or {}
    if bool(appendix_cfg.get("enabled", False)):
        rewriter = RuFPBenchAppendixRewriter(str(cfg_path))
        df_after_appendix = rewriter.run(df_raw)
        appendix_path = data_raw_dir / "merged_with_appendix.parquet"
        df_after_appendix.to_parquet(appendix_path, index=False)
        df_work = df_after_appendix
        summary["dataset"]["appendix_enabled"] = True
        summary["dataset"]["appendix_rows_added"] = int(len(df_after_appendix) - len(df_raw))
    else:
        df_work = df_raw
        summary["dataset"]["appendix_enabled"] = False

    # Step 2: Quality
    q_cfg = cfg.get("quality", {}) or {}
    duplicate_subset = q_cfg.get("duplicate_subset") or ["text"]
    outlier_k = float(q_cfg.get("outlier_k", 1.5))

    q_agent = DataQualityAgent(duplicate_subset=list(duplicate_subset), outlier_k=outlier_k)
    quality_report = q_agent.detect_issues(df_work)
    df_clean = q_agent.fix(df_work, strategy=(q_cfg.get("strategy") or {"missing": "fill", "duplicates": "drop", "outliers": "drop_iqr"}))
    comparison = q_agent.compare(df_work, df_clean)

    df_clean.to_parquet(data_interim_dir / "clean.parquet", index=False)
    write_json(reports_dir / "quality_report.json", quality_report)
    write_json(reports_dir / "quality_comparison.json", comparison)
    write_text(reports_dir / "quality_report.md", render_quality_report(quality_report, comparison))

    # Step 3: Annotation
    a_cfg = cfg.get("annotation", {}) or {}
    ann = AnnotationAgent(modality="text", config={
        "confidence_threshold": float(a_cfg.get("confidence_threshold", 0.7)),
        "include_predictions": bool(a_cfg.get("include_predictions", True)),
    })
    df_auto = ann.auto_label(df_clean)
    df_auto.to_parquet(data_interim_dir / "auto_labeled.parquet", index=False)

    spec_path = ann.generate_spec(df_auto, task=str(a_cfg.get("task", "text_classification")))
    ls_path = ann.export_to_labelstudio(df_auto)

    # HITL: apply corrections if present
    hitl_cfg = cfg.get("hitl", {}) or {}
    human_mode = args.human_mode or hitl_cfg.get("human_mode", "stop_if_missing")
    review_queue_path = Path(hitl_cfg.get("review_queue_path", "data/labeled/review_queue.csv"))
    corrected_path = Path(hitl_cfg.get("corrected_queue_path", "data/labeled/review_queue_corrected.csv"))

    # Also maintain root-level copies as per course structure
    if review_queue_path.exists():
        Path("review_queue.csv").write_text(review_queue_path.read_text(encoding="utf-8"), encoding="utf-8")

    df_reviewed = df_auto.copy()
    hitl_stats: Dict[str, Any] = {}
    if corrected_path.exists():
        df_reviewed, hitl_stats = apply_human_corrections(df_auto, corrected_path=corrected_path)
        summary["hitl"] = hitl_stats
    else:
        # If no corrected file, optionally stop the pipeline here
        if human_mode == "stop_if_missing":
            # still write annotation report (without hitl corrections) and instructions
            metrics_pre = ann.check_quality(df_auto)
            write_json(reports_dir / "annotation_metrics.json", metrics_pre)
            write_text(
                reports_dir / "annotation_report.md",
                render_annotation_report(metrics_pre, spec_path, ls_path, hitl_stats={})
            )
            msg = (
                "\n[HITL] review_queue.csv has been generated.\n"
                f"Please edit it and save as {corrected_path.as_posix()} to continue.\n"
                "Then re-run: python run_pipeline.py\n"
            )
            print(msg)
            return 2

    # metrics after HITL (kappa if human_label exists)
    metrics = ann.check_quality(df_reviewed)
    write_json(reports_dir / "annotation_metrics.json", metrics)
    write_text(reports_dir / "annotation_report.md", render_annotation_report(metrics, spec_path, ls_path, hitl_stats))

    # Prepare final label for later steps
    if "label_final" not in df_reviewed.columns:
        df_reviewed["label_final"] = df_reviewed["label"].astype("string")

    

    # Step 3.5: AL-based selection for *additional* manual labeling (optional)
    al_sel_cfg = cfg.get("al_select", {}) or {}
    if bool(al_sel_cfg.get("enabled", True)):
        sel_strategy = str(al_sel_cfg.get("strategy", "entropy"))
        sel_batch = int(al_sel_cfg.get("batch_size", 100))
        out_csv = Path(al_sel_cfg.get("output_csv", "data/labeled/al_candidates.csv"))
        out_ls = Path(al_sel_cfg.get("output_labelstudio", "data/labeled/al_candidates_labelstudio.json"))

        # Decide what is "already human-labeled" vs pool.
        if "human_label" in df_reviewed.columns and df_reviewed["human_label"].notna().any():
            fit_df = df_reviewed[df_reviewed["human_label"].notna()].copy()
            pool_df = df_reviewed[df_reviewed["human_label"].isna()].copy()
        else:
            # Fallback: treat high-confidence auto labels as labeled, low-confidence as pool.
            thr = float(a_cfg.get("confidence_threshold", 0.7))
            conf = pd.to_numeric(df_reviewed.get("confidence"), errors="coerce")
            fit_df = df_reviewed[conf >= thr].copy()
            pool_df = df_reviewed[conf < thr].copy()

        # Only run selection if we have enough fit data and a non-empty pool
        if len(fit_df) >= 10 and len(pool_df) > 0:
            selector = ActiveLearningAgent(model="logreg", config={"text_col": "text", "label_col": "label_final", "random_state": int((cfg.get("active_learning", {}) or {}).get("random_state", 42))})
            selector.fit(fit_df)

            chosen_idx = selector.query(pool_df, strategy=sel_strategy, batch_size=sel_batch)
            selected = pool_df.loc[chosen_idx].copy()

            out_csv.parent.mkdir(parents=True, exist_ok=True)
            selected.to_csv(out_csv, index=False)

            # Label Studio import for selected batch (with auto-predictions as pre-annotations)
            out_ls.parent.mkdir(parents=True, exist_ok=True)
            ann.export_to_labelstudio(selected, output_path=out_ls.as_posix())

            write_text(
                reports_dir / "al_select_report.md",
                "# al_select_report\n\n"
                f"- strategy: {sel_strategy}\n"
                f"- batch_size: {sel_batch}\n"
                f"- fit_rows: {len(fit_df)}\n"
                f"- pool_rows: {len(pool_df)}\n"
                f"- selected_rows: {len(selected)}\n\n"
                f"Artifacts:\n- `{out_csv.as_posix()}`\n- `{out_ls.as_posix()}`\n"
            )
        else:
            write_text(
                reports_dir / "al_select_report.md",
                "# al_select_report\n\n"
                "Selection skipped: not enough labeled data for fitting or pool is empty.\n"
            )

# Step 4: Active Learning (compare strategies)
    al_cfg = cfg.get("active_learning", {}) or {}
    if bool(al_cfg.get("enabled", True)):
        start_n = int(al_cfg.get("start_n", 50))
        n_iterations = int(al_cfg.get("n_iterations", 5))
        batch_size = int(al_cfg.get("batch_size", 20))
        strategies = al_cfg.get("strategies") or ["entropy", "random"]
        metric = str(al_cfg.get("metric", "f1"))
        test_size = float(al_cfg.get("test_size", 0.2))
        rs = int((cfg.get("active_learning", {}) or {}).get("random_state", 42))

        # Use oracle labels from label_final for simulation
        df_al = df_reviewed.copy()
        df_al["text"] = df_al["text"].astype("string").fillna("").str.strip()
        df_al = df_al[df_al["text"] != ""].copy()
        df_al["al_label"] = df_al["label_final"].astype("string").fillna("unknown")

        # split once; keep same test across strategies
        train_pool, test_df = train_test_split(
            df_al,
            test_size=test_size,
            random_state=rs,
            stratify=df_al["al_label"] if df_al["al_label"].nunique() > 1 else None,
        )

        # start labeled subset
        if len(train_pool) <= start_n:
            start_n = max(5, int(len(train_pool) * 0.2))
        labeled_df, pool_df = train_test_split(
            train_pool,
            train_size=start_n,
            random_state=rs,
            stratify=train_pool["al_label"] if train_pool["al_label"].nunique() > 1 else None,
        )

        agent_al = ActiveLearningAgent(model="logreg", config={"text_col": "text", "label_col": "al_label", "random_state": rs})

        histories = {}
        for s in strategies:
            histories[s] = agent_al.run_cycle(
                labeled_df=labeled_df,
                pool_df=pool_df,
                test_df=test_df,
                strategy=str(s),
                n_iterations=n_iterations,
                batch_size=batch_size,
            )

        curve_path = agent_al.report(histories, metric=metric, output_path="reports/learning_curve.png",
                                     title=f"Active Learning: {' vs '.join(strategies)} ({metric})")

        # compute label savings: how many labels entropy needs to reach random final metric
        savings = {}
        if "random" in histories and "entropy" in histories:
            target = histories["random"][-1].get(metric)
            best_n = None
            for r in histories["entropy"]:
                if r.get(metric, 0) >= target:
                    best_n = r["n_labeled"]
                    break
            if best_n is not None:
                savings = {
                    "metric": metric,
                    "random_final_value": float(target),
                    "random_final_n_labeled": int(histories["random"][-1]["n_labeled"]),
                    "entropy_reaches_at_n_labeled": int(best_n),
                    "label_savings": int(histories["random"][-1]["n_labeled"] - best_n),
                }

        al_summary = {
            "setup": {
                "start_n": start_n,
                "n_iterations": n_iterations,
                "batch_size": batch_size,
                "strategies": strategies,
                "metric": metric,
                "test_size": test_size,
            },
            "results": {k: {"history": v} for k, v in histories.items()},
            "label_savings": savings,
            "curve_path": curve_path,
        }
        write_json(reports_dir / "al_history.json", al_summary)
        write_text(reports_dir / "al_report.md", render_al_report(al_summary))
        summary["metrics"]["active_learning"] = {
            "curve_path": curve_path,
            "label_savings": savings,
        }

    # Step 5: Training
    t_cfg = cfg.get("training", {}) or {}
    if bool(t_cfg.get("enabled", True)):
        test_size = float(t_cfg.get("test_size", 0.2))
        rs = int(t_cfg.get("random_state", 42))
        train_metrics = train_final_model(df_reviewed, label_col="label_final", test_size=test_size, random_state=rs)
        model = train_metrics.pop("model")
        model_path = Path(t_cfg.get("output_model_path", "models/baseline_logreg.pkl"))
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib_dump(model, model_path.as_posix())

        write_json(reports_dir / "train_metrics.json", train_metrics)
        write_text(reports_dir / "train_report.md", render_train_report(train_metrics, model_path.as_posix()))
        summary["metrics"]["training"] = {
            "accuracy": train_metrics.get("accuracy"),
            "f1_macro": train_metrics.get("f1_macro"),
            "model_path": model_path.as_posix(),
        }

    # Step 6: Final dataset + reports
    rep_cfg = cfg.get("reporting", {}) or {}
    final_path = Path(rep_cfg.get("final_dataset_path", "data/labeled/final_dataset.parquet"))
    final_csv = Path(rep_cfg.get("final_dataset_csv", "data/labeled/final_dataset.csv"))

    df_reviewed.to_parquet(final_path, index=False)
    df_reviewed.to_csv(final_csv, index=False)

    write_text(data_labeled_dir / "DATA_CARD.md", render_data_card(df_reviewed))

    summary["dataset"].update({
        "n_rows_final": int(len(df_reviewed)),
        "final_dataset_path": final_path.as_posix(),
        "data_card": (data_labeled_dir / "DATA_CARD.md").as_posix(),
    })

    # Final report
    write_text(reports_dir / "final_report.md", render_final_report(summary))
    write_json(reports_dir / "pipeline_summary.json", summary)

    print("\nPipeline completed successfully.")
    print(f"- Final dataset: {final_path}")
    print(f"- Reports: {reports_dir}/final_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
