"""Load JSONL/CSV and write JSONL + optional debug CSV."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Iterator, Literal, TextIO

import pandas as pd

from src.schemas import DocumentGraph

logger = logging.getLogger(__name__)

InputFormat = Literal["auto", "jsonl", "csv"]

_JSONL_BAD_LINE_LOG_CAP = 24


def detect_format(path: Path) -> Literal["jsonl", "csv"]:
    suf = path.suffix.lower()
    if suf == ".csv":
        return "csv"
    if suf in {".jsonl", ".ndjson", ".json"}:
        return "jsonl"
    return "jsonl"


def validate_input_file(path: Path) -> None:
    """Readable existing file; raises ValueError with a clear message."""
    if not path.exists():
        raise ValueError(f"Input path does not exist: {path.resolve()}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file (directories are not supported): {path.resolve()}")
    try:
        with path.open("r", encoding="utf-8") as f:
            f.read(1)
    except OSError as e:
        raise ValueError(f"Cannot read input file {path.resolve()}: {e}") from e
    except UnicodeDecodeError as e:
        raise ValueError(
            f"Input file is not valid UTF-8: {path.resolve()}. "
            f"Re-encode as UTF-8 or fix the file. ({e})"
        ) from e


def ensure_output_writable(path: Path) -> None:
    """Ensure parent directory exists or can be created."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise ValueError(f"Cannot create output directory {path.parent.resolve()}: {e}") from e


def _coerce_row_text(value: Any, *, line_no: int, path: Path, text_column: str) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        raise ValueError(
            f"{path}: line {line_no}: field {text_column!r} must be a scalar string/number, "
            f"got {type(value).__name__}. JSON objects/arrays are not valid body text."
        )
    return str(value)


def _parse_jsonl_object(line: str, line_no: int, path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"{path}: line {line_no}: invalid JSON ({e}). "
            f"Fix the line or re-run with --skip-bad-jsonl-lines to skip."
        ) from e
    if not isinstance(obj, dict):
        raise ValueError(
            f"{path}: line {line_no}: expected a JSON object per line, got {type(obj).__name__}"
        )
    return obj


def load_documents(
    path: Path,
    text_column: str = "text",
    id_column: str = "id",
    input_format: InputFormat = "auto",
    *,
    skip_malformed_jsonl_lines: bool = False,
) -> list[dict[str, Any]]:
    """
    Load rows as dicts. Each row must include ``text_column`` (value coerced to str).

    Preserves all keys for downstream ``metadata.extra`` (except reserved columns
    handled in ``iter_document_inputs``).

    :param skip_malformed_jsonl_lines: if True, skip invalid JSON lines after logging
        (capped number of warnings); only applies to JSONL.
    """
    path = path.expanduser().resolve()
    validate_input_file(path)

    fmt = detect_format(path) if input_format == "auto" else input_format
    if fmt == "csv":
        return _load_csv(path, text_column)

    return _load_jsonl(path, text_column, skip_malformed=skip_malformed_jsonl_lines)


def _load_csv(path: Path, text_column: str) -> list[dict[str, Any]]:
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        logger.warning("CSV is empty: %s — returning zero rows", path)
        return []
    except Exception as e:
        raise ValueError(f"Failed to read CSV {path}: {e}") from e

    if text_column not in df.columns:
        raise ValueError(
            f"CSV {path} has no column {text_column!r}. "
            f"Available columns: {list(df.columns)}. Use --text-column to set the text field name."
        )
    rows = df.to_dict(orient="records")
    logger.info("Loaded %d rows from %s (csv)", len(rows), path)
    return rows


def _load_jsonl(
    path: Path,
    text_column: str,
    *,
    skip_malformed: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bad_lines = 0
    bad_logged = 0

    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = _parse_jsonl_object(line, line_no, path)
            except ValueError as e:
                if not skip_malformed:
                    raise
                bad_lines += 1
                if bad_logged < _JSONL_BAD_LINE_LOG_CAP:
                    logger.warning("%s", e)
                    bad_logged += 1
                elif bad_logged == _JSONL_BAD_LINE_LOG_CAP:
                    logger.warning(
                        "%s: further malformed JSONL lines will not be logged individually "
                        "(use DEBUG for full trace if needed).",
                        path,
                    )
                    bad_logged += 1
                continue

            if text_column not in obj:
                raise ValueError(
                    f"{path}: line {line_no}: missing required field {text_column!r}. "
                    f"Keys present: {list(obj.keys())[:20]}{'…' if len(obj) > 20 else ''}"
                )
            try:
                obj = dict(obj)
                obj[text_column] = _coerce_row_text(
                    obj[text_column],
                    line_no=line_no,
                    path=path,
                    text_column=text_column,
                )
            except ValueError as e:
                if not skip_malformed:
                    raise
                bad_lines += 1
                if bad_logged < _JSONL_BAD_LINE_LOG_CAP:
                    logger.warning("%s", e)
                    bad_logged += 1
                continue

            rows.append(obj)

    if bad_lines:
        logger.warning(
            "JSONL %s: skipped %d malformed or invalid-text line(s); loaded %d good row(s)",
            path,
            bad_lines,
            len(rows),
        )
    logger.info("Loaded %d rows from %s (jsonl)", len(rows), path)
    return rows


def extra_fields_from_row(
    row: dict[str, Any],
    text_column: str,
    id_column: str,
) -> dict[str, Any]:
    """
    All row keys except text/id/source/category → copied into metadata.extra.
    Same rules for JSONL dict rows and CSV records.
    """
    skip = {text_column, id_column, "source", "category"}
    return {k: row[k] for k in row if k not in skip}


def _stable_doc_id(row: dict[str, Any], id_column: str, index: int) -> str:
    raw = row.get(id_column)
    if raw is None:
        return f"doc_{index}"
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return f"doc_{index}"
    return s


def iter_document_inputs(
    rows: list[dict[str, Any]],
    text_column: str,
    id_column: str,
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """
    Yield (doc_id, text, meta).

    meta keys:
      - source, category: optional first-class fields for DocumentMetadata
      - extra: dict of all other columns (preserves custom JSONL/CSV fields)
    """
    for i, row in enumerate(rows):
        doc_id = _stable_doc_id(row, id_column, i)
        text = row.get(text_column, "")
        if text is None:
            text = ""
        text = str(text)
        meta: dict[str, Any] = {
            "source": row.get("source"),
            "category": row.get("category"),
            "extra": extra_fields_from_row(row, text_column, id_column),
        }
        yield doc_id, text, meta


def write_jsonl_line(fp: TextIO, graph: DocumentGraph) -> None:
    fp.write(graph.model_dump_json(ensure_ascii=False))
    fp.write("\n")


def write_jsonl(path: Path, graphs: list[DocumentGraph]) -> None:
    ensure_output_writable(path)
    with path.open("w", encoding="utf-8") as f:
        for g in graphs:
            write_jsonl_line(f, g)
    logger.info("Wrote %d documents to %s", len(graphs), path)


def write_debug_nodes_csv(path: Path, graphs: list[DocumentGraph]) -> None:
    ensure_output_writable(path)
    fieldnames = [
        "doc_id",
        "node_id",
        "canonical_text",
        "normalized_text",
        "labels",
        "sources",
        "max_score",
        "mention_count",
        "raw_mentions_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as wf:
        w = csv.DictWriter(wf, fieldnames=fieldnames)
        w.writeheader()
        for g in graphs:
            for n in g.merged_nodes:
                w.writerow(
                    {
                        "doc_id": g.id,
                        "node_id": n.node_id,
                        "canonical_text": n.canonical_text,
                        "normalized_text": n.normalized_text,
                        "labels": "|".join(n.labels),
                        "sources": "|".join(n.sources),
                        "max_score": n.max_score,
                        "mention_count": n.mention_count,
                        "raw_mentions_json": json.dumps(
                            [m.model_dump() for m in n.raw_mentions],
                            ensure_ascii=False,
                        ),
                    }
                )
    logger.info("Wrote node debug CSV to %s", path)


def write_debug_edges_csv(path: Path, graphs: list[DocumentGraph]) -> None:
    ensure_output_writable(path)
    fieldnames = [
        "doc_id",
        "source_node_id",
        "target_node_id",
        "edge_type",
        "weight",
        "evidence_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as wf:
        w = csv.DictWriter(wf, fieldnames=fieldnames)
        w.writeheader()
        for g in graphs:
            for e in g.edges:
                w.writerow(
                    {
                        "doc_id": g.id,
                        "source_node_id": e.source_node_id,
                        "target_node_id": e.target_node_id,
                        "edge_type": e.edge_type,
                        "weight": e.weight,
                        "evidence_json": e.evidence.model_dump_json(),
                    }
                )
    logger.info("Wrote edge debug CSV to %s", path)
