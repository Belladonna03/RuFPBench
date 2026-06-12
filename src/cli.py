"""CLI: run pseudo-graph extraction pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import typer

from src.config import DEFAULT_GLINER_LABELS, DEFAULT_KEYPHRASE_EXTRA_BLOCKLIST, InputFormat, PipelineConfig
from src.io_utils import (
    ensure_output_writable,
    iter_document_inputs,
    load_documents,
    validate_input_file,
    write_debug_edges_csv,
    write_debug_nodes_csv,
    write_jsonl,
    write_jsonl_line,
)
from src import __version__
from src.pipeline import PseudoGraphPipeline, RunStats, process_one_safe

app = typer.Typer(
    add_completion=False,
    help=(
        "Pseudo-graph extraction: GLiNER entities + KeyBERT keyphrases → merged nodes → "
        "co-occurrence edges. Reads JSONL or CSV, writes one JSON object per input row."
    ),
)


def _metadata_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s and s.lower() != "nan" else None


def _normalize_input_format(value: str) -> InputFormat:
    v = value.strip().lower()
    if v not in ("auto", "jsonl", "csv"):
        raise typer.BadParameter("Expected one of: auto, jsonl, csv")
    return v  # type: ignore[return-value]


def _validate_corpus_options(
    entity_threshold: float,
    top_n_keyphrases: int,
    keyphrase_min_ngram: int,
    keyphrase_max_ngram: int,
    diversity: float,
    window_size: int,
    limit: Optional[int],
) -> None:
    if not 0.0 <= entity_threshold <= 1.0:
        raise typer.BadParameter("--entity-threshold must be between 0 and 1")
    if top_n_keyphrases < 1:
        raise typer.BadParameter("--top-n-keyphrases must be >= 1")
    if keyphrase_min_ngram < 1 or keyphrase_max_ngram < 1:
        raise typer.BadParameter("keyphrase ngram bounds must be >= 1")
    if keyphrase_min_ngram > keyphrase_max_ngram:
        raise typer.BadParameter("--keyphrase-min-ngram must be <= --keyphrase-max-ngram")
    if not 0.0 <= diversity <= 1.0:
        raise typer.BadParameter("--diversity must be between 0 and 1")
    if window_size < 1:
        raise typer.BadParameter("--window-size must be >= 1")
    if limit is not None and limit < 1:
        raise typer.BadParameter("--limit must be >= 1 when set")


@app.command("version")
def version_cmd() -> None:
    """Print package version."""
    typer.echo(__version__)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


@app.command("run")
def run_cmd(
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        exists=False,
        readable=False,
        help="Input file: newline-delimited JSON (.jsonl / .ndjson / .json) or .csv",
    ),
    output_path: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output JSONL path (one DocumentGraph JSON per line). Parent dirs are created.",
    ),
    text_column: str = typer.Option(
        "text",
        "--text-column",
        help="Column / JSON field containing document body text",
    ),
    id_column: str = typer.Option(
        "id",
        "--id-column",
        help="Column / JSON field for stable id (fallback: doc_<row_index> if missing/empty)",
    ),
    input_format: str = typer.Option(
        "auto",
        "--input-format",
        help="auto (from extension), jsonl, or csv",
    ),
    gliner_model: str = typer.Option(
        "urchade/gliner_medium-v2.1",
        "--gliner-model",
        help="HuggingFace id or path for GLiNER weights",
    ),
    keybert_model: str = typer.Option(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "--keybert-model",
        help="sentence-transformers model id for KeyBERT embeddings",
    ),
    entity_threshold: float = typer.Option(0.35, "--entity-threshold", min=0.0, max=1.0),
    top_n_keyphrases: int = typer.Option(10, "--top-n-keyphrases", min=1),
    keyphrase_min_ngram: int = typer.Option(1, "--keyphrase-min-ngram", min=1),
    keyphrase_max_ngram: int = typer.Option(3, "--keyphrase-max-ngram", min=1),
    use_mmr: bool = typer.Option(True, "--use-mmr/--no-mmr"),
    diversity: float = typer.Option(0.5, "--diversity", min=0.0, max=1.0),
    window_size: int = typer.Option(
        64,
        "--window-size",
        min=1,
        help="Max character gap for nearby_window edges (see README)",
    ),
    save_debug_csv: bool = typer.Option(
        False,
        "--save-debug-csv",
        help="Write *_debug_nodes.csv and *_debug_edges.csv next to --output (needs full RAM pass)",
    ),
    disable_soft_merge: bool = typer.Option(False, "--disable-soft-merge"),
    filter_overlaps_before_merge: bool = typer.Option(
        False,
        "--filter-overlaps-before-merge",
        help="Resolve overlapping GLiNER spans before merge/graph; raw entities unchanged in output",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="Torch device hint for models, e.g. cuda, cpu, mps",
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="DEBUG, INFO, WARNING, ERROR"),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        help="Process only the first N rows after load (debug, N >= 1)",
    ),
    skip_bad_jsonl_lines: bool = typer.Option(
        False,
        "--skip-bad-jsonl-lines",
        help="Skip malformed JSON lines instead of aborting (JSONL only; logs warnings)",
    ),
    continue_on_error: bool = typer.Option(
        False,
        "--continue-on-error",
        help="On per-document failure, write an empty graph with metadata.extra.pipeline_error",
    ),
) -> None:
    """
    Run the full pipeline on a corpus file.

    Input rows keep all fields except text/id/source/category in output ``metadata.extra``.
    """
    _setup_logging(log_level)
    log = logging.getLogger("src.cli")

    try:
        fmt = _normalize_input_format(input_format)
        _validate_corpus_options(
            entity_threshold,
            top_n_keyphrases,
            keyphrase_min_ngram,
            keyphrase_max_ngram,
            diversity,
            window_size,
            limit,
        )
    except typer.BadParameter:
        raise

    in_path = input_path.expanduser().resolve()
    out_path = output_path.expanduser().resolve()

    try:
        validate_input_file(in_path)
        ensure_output_writable(out_path)
    except ValueError as e:
        typer.echo(f"Path error: {e}", err=True)
        raise typer.Exit(code=1) from e

    cfg = PipelineConfig(
        gliner_model=gliner_model,
        keybert_model=keybert_model,
        entity_threshold=entity_threshold,
        entity_labels=list(DEFAULT_GLINER_LABELS),
        keyphrase_ngram_range=(keyphrase_min_ngram, keyphrase_max_ngram),
        top_n_keyphrases=top_n_keyphrases,
        use_mmr=use_mmr,
        diversity=diversity,
        nearby_window_chars=window_size,
        soft_merge_by_normalized_text=not disable_soft_merge,
        text_column=text_column,
        id_column=id_column,
        input_format=fmt,
        device=device,
        keyphrase_stop_phrases=DEFAULT_KEYPHRASE_EXTRA_BLOCKLIST,
    )

    try:
        rows = load_documents(
            in_path,
            text_column=text_column,
            id_column=id_column,
            input_format=cfg.input_format,
            skip_malformed_jsonl_lines=skip_bad_jsonl_lines,
        )
    except ValueError as e:
        typer.echo(f"Failed to load input: {e}", err=True)
        raise typer.Exit(code=1) from e

    if limit is not None:
        rows = rows[:limit]

    if not rows:
        typer.echo("No rows to process (empty input). Nothing written.", err=True)
        raise typer.Exit(code=0)

    log.info("Loaded %d row(s); initializing models (first run may download weights)...", len(rows))
    try:
        pipeline = PseudoGraphPipeline(cfg)
    except Exception as e:
        typer.echo(
            "Failed to load ML models (GLiNER / sentence-transformers). "
            "Check network, disk space, CUDA drivers, and --device. "
            f"Underlying error: {e}",
            err=True,
        )
        log.exception("Model initialization failed")
        raise typer.Exit(code=1) from e

    from tqdm import tqdm

    stats = RunStats()
    on_err = "empty" if continue_on_error else "raise"
    documents = list(iter_document_inputs(rows, text_column, id_column))

    try:
        if save_debug_csv:
            graphs: list = []
            for doc_id, text, meta in tqdm(
                documents,
                total=len(documents),
                desc="Documents",
                unit="doc",
            ):
                g = process_one_safe(
                    pipeline,
                    doc_id,
                    text,
                    source=_metadata_optional_str(meta.get("source")),
                    category=_metadata_optional_str(meta.get("category")),
                    extra_meta=dict(meta.get("extra") or {}),
                    filter_overlaps_before_merge=filter_overlaps_before_merge,
                    on_error=on_err,
                    stats=stats,
                )
                graphs.append(g)
            write_jsonl(out_path, graphs)
            stem = out_path.with_suffix("")
            write_debug_nodes_csv(Path(str(stem) + "_debug_nodes.csv"), graphs)
            write_debug_edges_csv(Path(str(stem) + "_debug_edges.csv"), graphs)
        else:
            ensure_output_writable(out_path)
            with out_path.open("w", encoding="utf-8") as out_fp:
                for doc_id, text, meta in tqdm(
                    documents,
                    total=len(documents),
                    desc="Documents",
                    unit="doc",
                ):
                    g = process_one_safe(
                        pipeline,
                        doc_id,
                        text,
                        source=_metadata_optional_str(meta.get("source")),
                        category=_metadata_optional_str(meta.get("category")),
                        extra_meta=dict(meta.get("extra") or {}),
                        filter_overlaps_before_merge=filter_overlaps_before_merge,
                        on_error=on_err,
                        stats=stats,
                    )
                    write_jsonl_line(out_fp, g)
            log.info("Streamed %d document(s) to %s", len(documents), out_path)
    except Exception as e:
        typer.echo(f"Pipeline stopped: {e}", err=True)
        log.exception("Fatal error during corpus processing")
        raise typer.Exit(code=1) from e

    stats.log_summary(log)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
