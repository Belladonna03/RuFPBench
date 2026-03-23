from __future__ import annotations

"""Code-only EDA for merged collection dataframe (text-first)."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd

from shared.logging_utils import get_logger
from shared.ru_eda_text import extract_top_ru_content_words, plot_top_content_words
from shared.utils import loads_meta

_log = get_logger("agents.collection_eda")


def _extract_seed_roles(series: pd.Series) -> pd.Series:
    roles: list[Any] = []
    for v in series:
        if pd.isna(v) or v is None:
            roles.append(pd.NA)
            continue
        if isinstance(v, str):
            m = loads_meta(v)
            sr = m.get("seed_role")
            roles.append(str(sr) if isinstance(sr, str) and sr.strip() else pd.NA)
        else:
            roles.append(pd.NA)
    return pd.Series(roles, index=series.index, dtype="object")


def eda(
    df: pd.DataFrame,
    *,
    top_content_words_k: int = 20,
    content_word_min_len: int = 3,
    include_english_stopwords: bool = True,
) -> dict[str, Any]:
    """Compute text-only EDA statistics (no LLM)."""
    out: dict[str, Any] = {"n_rows": int(len(df))}
    if df.empty:
        out["empty"] = True
        return out

    work = df.copy()
    for col in ("text", "label", "source"):
        if col not in work.columns:
            work[col] = None

    work["text"] = work["text"].astype(str)
    work["label"] = work["label"].astype(str)
    work["source"] = work["source"].astype(str)

    out["class_distribution"] = work["label"].value_counts(dropna=False).to_dict()
    out["source_distribution"] = work["source"].value_counts(dropna=False).to_dict()

    chars = work["text"].str.len()
    words = work["text"].str.split().apply(len)
    out["text_length_chars"] = {
        "min": float(chars.min()),
        "max": float(chars.max()),
        "mean": float(chars.mean()),
        "median": float(chars.median()),
        "std": float(chars.std(ddof=0)) if len(chars) > 1 else 0.0,
    }
    out["text_length_words"] = {
        "min": float(words.min()),
        "max": float(words.max()),
        "mean": float(words.mean()),
        "median": float(words.median()),
        "std": float(words.std(ddof=0)) if len(words) > 1 else 0.0,
    }

    top_rows, tw_meta = extract_top_ru_content_words(
        work["text"],
        top_k=top_content_words_k,
        min_token_len=content_word_min_len,
        include_english_stopwords=include_english_stopwords,
    )
    out["top_content_words"] = top_rows
    out["top20_words"] = top_rows  # backward-compatible name for downstream / LLM JSON
    out["top_words_meta"] = tw_meta

    ct = pd.crosstab(work["source"], work["label"], dropna=False)
    out["source_label_crosstab"] = ct.to_dict()

    if "meta" in work.columns:
        sr = _extract_seed_roles(work["meta"])
        roles_clean = [str(x) for x in sr.tolist() if pd.notna(x) and str(x).strip() and str(x) != "<NA>"]
        out["seed_role_distribution"] = dict(Counter(roles_clean)) if roles_clean else {}
    else:
        out["seed_role_distribution"] = {}

    return out


def _write_eda_summary_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Collection EDA (code-generated)",
        "",
        f"- **Rows:** {report.get('n_rows', 0)}",
        "",
        "## Class distribution",
        "",
    ]
    cd = report.get("class_distribution") or {}
    for k, v in sorted(cd.items(), key=lambda x: (-x[1], str(x[0]))):
        lines.append(f"- `{k}`: {v}")
    lines.extend(["", "## Source distribution", ""])
    sd = report.get("source_distribution") or {}
    for k, v in sorted(sd.items(), key=lambda x: (-x[1], str(x[0]))):
        lines.append(f"- `{k}`: {v}")
    lines.extend(["", "## Text length (chars)", ""])
    tl = report.get("text_length_chars") or {}
    for k in ("min", "max", "mean", "median", "std"):
        if k in tl:
            lines.append(f"- **{k}:** {tl[k]:.4g}")
    lines.extend(["", "## Text length (words)", ""])
    tlw = report.get("text_length_words") or {}
    for k in ("min", "max", "mean", "median", "std"):
        if k in tlw:
            lines.append(f"- **{k}:** {tlw[k]:.4g}")
    lines.extend(["", "## Top content words", ""])
    topw = report.get("top_content_words") or report.get("top20_words") or []
    tm = report.get("top_words_meta") or {}
    if not topw:
        lines.append("(no tokens after cleaning / stopwords; check texts or dependencies)")
    else:
        lines.append(
            f"- **Method:** NLTK stopwords ({tm.get('stopwords', '?')}), "
            f"lemmatized={tm.get('lemmatized', '?')}, lemma_backend={tm.get('lemma_backend', '?')}, "
            f"min_len={tm.get('min_token_len', '?')}"
        )
        for row in topw[:30]:
            lines.append(f"- `{row.get('word')}`: {row.get('count')}")
        if len(topw) > 30:
            lines.append(f"- … ({len(topw)} total in table)")
    lines.extend(["", "## Seed role (from meta)", ""])
    srd = report.get("seed_role_distribution") or {}
    if not srd:
        lines.append("(no seed_role in meta or all empty)")
    else:
        for k, v in sorted(srd.items(), key=lambda x: (-x[1], str(x[0]))):
            lines.append(f"- `{k}`: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_eda(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    top_content_words_k: int = 20,
    content_word_min_len: int = 3,
    include_english_stopwords: bool = True,
) -> dict[str, Any]:
    """Compute EDA, save CSV/JSON/PNG and eda_summary.md. Returns the same dict as `eda()`."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report = eda(
        df,
        top_content_words_k=top_content_words_k,
        content_word_min_len=content_word_min_len,
        include_english_stopwords=include_english_stopwords,
    )

    if df.empty or report.get("empty"):
        (out_path / "text_length_summary.json").write_text(
            json.dumps({"note": "empty dataframe"}, indent=2), encoding="utf-8"
        )
        _write_eda_summary_md(report, out_path / "eda_summary.md")
        return report

    work = df.copy()
    for col in ("text", "label", "source"):
        if col not in work.columns:
            work[col] = None
    work["text"] = work["text"].astype(str)
    work["label"] = work["label"].astype(str)
    work["source"] = work["source"].astype(str)

    # CSVs
    pd.Series(report["class_distribution"], name="count").to_csv(out_path / "class_distribution.csv")
    pd.Series(report["source_distribution"], name="count").to_csv(out_path / "source_distribution.csv")
    k = top_content_words_k
    top_fname = f"top{k}_words.csv"
    pd.DataFrame(report["top_content_words"]).to_csv(out_path / top_fname, index=False)
    ct = pd.crosstab(work["source"], work["label"], dropna=False)
    ct.to_csv(out_path / "source_label_crosstab.csv")

    tjson = {
        "text_length_chars": report["text_length_chars"],
        "text_length_words": report["text_length_words"],
    }
    (out_path / "text_length_summary.json").write_text(json.dumps(tjson, indent=2), encoding="utf-8")

    chars = work["text"].str.len()
    words = work["text"].str.split().apply(len)

    # Plots
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(chars.clip(upper=chars.quantile(0.99) if len(chars) else 0), bins=50, color="steelblue", edgecolor="white")
    ax.set_title("Text length (characters)")
    ax.set_xlabel("chars")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path / "text_length_hist_chars.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    wclip = words.clip(upper=words.quantile(0.99) if len(words) else 0)
    ax.hist(wclip, bins=50, color="seagreen", edgecolor="white")
    ax.set_title("Text length (words)")
    ax.set_xlabel("words")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path / "text_length_hist_words.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    pd.Series(report["class_distribution"]).plot(kind="bar", ax=ax, color="coral")
    ax.set_title("Class distribution")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path / "class_distribution.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    pd.Series(report["source_distribution"]).plot(kind="bar", ax=ax, color="slateblue")
    ax.set_title("Source distribution")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path / "source_distribution.png", dpi=120)
    plt.close(fig)

    top_rows = report.get("top_content_words") or []
    plot_top_content_words(
        top_rows,
        out_path / f"top{k}_words.png",
        title=f"Top {k} content words",
    )

    _write_eda_summary_md(report, out_path / "eda_summary.md")
    _log.info("collection EDA artifacts saved under %s", out_path)
    return report
