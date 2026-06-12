"""Orchestrate normalization → GLiNER → KeyBERT → merge → graph → DocumentGraph."""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

from src.config import PipelineConfig
from src.entity_extractor import GlinerEntityExtractor, resolve_overlapping_entities
from src.graph_builder import build_cooccurrence_edges
from src.keyphrase_extractor import KeyphraseExtractor
from src.merge_dedup import merge_entities_and_keyphrases
from src.normalization import normalize_text
from src.schemas import DocumentGraph, DocumentMetadata

logger = logging.getLogger(__name__)


@dataclass
class RunStats:
    """Counters aggregated over a CLI ``run`` (or any loop over ``process_one``)."""

    documents: int = 0
    empty_text: int = 0
    no_entities: int = 0
    no_keyphrases: int = 0
    nonempty_text_zero_extractions: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    errors: int = 0
    error_samples: list[str] = field(default_factory=list)

    def record_graph(self, graph: DocumentGraph, original_text: str) -> None:
        self.documents += 1
        stripped = original_text.strip()
        if not stripped:
            self.empty_text += 1
        if not graph.entities:
            self.no_entities += 1
        if not graph.keyphrases:
            self.no_keyphrases += 1
        if stripped and not graph.entities and not graph.keyphrases:
            self.nonempty_text_zero_extractions += 1
        self.total_nodes += len(graph.merged_nodes)
        self.total_edges += len(graph.edges)

    def record_error(self, doc_id: str, exc: BaseException, max_samples: int = 5) -> None:
        self.errors += 1
        if len(self.error_samples) < max_samples:
            self.error_samples.append(f"{doc_id}: {type(exc).__name__}: {exc}")

    def log_summary(self, log: logging.Logger) -> None:
        n = self.documents or 1
        log.info(
            "Run summary: documents=%d empty_text=%d no_entities=%d no_keyphrases=%d "
            "nonempty_but_zero_extractions=%d errors=%d",
            self.documents,
            self.empty_text,
            self.no_entities,
            self.no_keyphrases,
            self.nonempty_text_zero_extractions,
            self.errors,
        )
        log.info(
            "Totals: avg_nodes_per_doc=%.2f avg_edges_per_doc=%.2f",
            self.total_nodes / n,
            self.total_edges / n,
        )
        for sample in self.error_samples:
            log.error("Error sample: %s", sample)


def empty_result_graph(
    doc_id: str,
    original_text: str,
    *,
    source: Optional[str] = None,
    category: Optional[str] = None,
    extra_meta: Optional[dict[str, Any]] = None,
    error_message: Optional[str] = None,
) -> DocumentGraph:
    """Minimal graph after a hard failure (continue-on-error mode)."""
    norm = normalize_text(original_text)
    extra: dict[str, Any] = dict(extra_meta or {})
    if error_message:
        extra["pipeline_error"] = error_message
    return DocumentGraph(
        id=doc_id,
        original_text=original_text,
        normalized_text=norm,
        entities=[],
        keyphrases=[],
        merged_nodes=[],
        edges=[],
        metadata=DocumentMetadata(
            source=source,
            category=category,
            num_entities=0,
            num_keyphrases=0,
            num_nodes=0,
            num_edges=0,
            extra=extra,
        ),
    )


class PseudoGraphPipeline:
    """Runs GLiNER + KeyBERT + merge + graph for one document string."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.gliner = GlinerEntityExtractor(
            model_name=config.gliner_model,
            labels=config.entity_labels,
            threshold=config.entity_threshold,
            device=config.device,
        )
        self.keyphrase_extractor = KeyphraseExtractor(config)

    def process_one(
        self,
        doc_id: str,
        original_text: str,
        source: Optional[str] = None,
        category: Optional[str] = None,
        extra_meta: Optional[dict[str, Any]] = None,
        filter_overlaps_before_merge: bool = False,
    ) -> DocumentGraph:
        if not original_text.strip():
            logger.debug("Skipping model calls for empty document id=%s", doc_id)

        norm_full = normalize_text(original_text)

        try:
            entities = self.gliner.extract(original_text)
        except Exception:
            logger.exception("Entity extraction failed for id=%s", doc_id)
            raise

        try:
            keyphrases = self.keyphrase_extractor.extract(original_text)
        except Exception:
            logger.exception("Keyphrase extraction failed for id=%s", doc_id)
            raise

        if not entities and not keyphrases and original_text.strip():
            logger.debug("No entities or keyphrases for id=%s (non-empty text)", doc_id)

        entities_for_merge = (
            resolve_overlapping_entities(entities, strategy="keep_highest_score")
            if filter_overlaps_before_merge
            else entities
        )

        merged_nodes = merge_entities_and_keyphrases(
            entities_for_merge,
            keyphrases,
            doc_id=doc_id,
            soft_merge_same_normalized=self.config.soft_merge_by_normalized_text,
        )

        edges = build_cooccurrence_edges(
            original_text,
            merged_nodes,
            nearby_window_chars=self.config.nearby_window_chars,
        )

        meta_extra: dict[str, Any] = dict(extra_meta or {})
        metadata = DocumentMetadata(
            source=source,
            category=category,
            num_entities=len(entities),
            num_keyphrases=len(keyphrases),
            num_nodes=len(merged_nodes),
            num_edges=len(edges),
            extra=meta_extra,
        )

        return DocumentGraph(
            id=doc_id,
            original_text=original_text,
            normalized_text=norm_full,
            entities=entities,
            keyphrases=keyphrases,
            merged_nodes=merged_nodes,
            edges=edges,
            metadata=metadata,
        )


def process_one_safe(
    pipeline: PseudoGraphPipeline,
    doc_id: str,
    original_text: str,
    *,
    source: Optional[str] = None,
    category: Optional[str] = None,
    extra_meta: Optional[dict[str, Any]] = None,
    filter_overlaps_before_merge: bool = False,
    on_error: str = "raise",
    stats: Optional[RunStats] = None,
) -> DocumentGraph:
    """
    Run ``process_one``; on failure either re-raise or return an empty graph.

    on_error: "raise" | "empty"
    """
    if on_error not in ("raise", "empty"):
        raise ValueError(f"on_error must be 'raise' or 'empty', got {on_error!r}")
    try:
        g = pipeline.process_one(
            doc_id,
            original_text,
            source=source,
            category=category,
            extra_meta=extra_meta,
            filter_overlaps_before_merge=filter_overlaps_before_merge,
        )
        if stats:
            stats.record_graph(g, original_text)
        return g
    except Exception as exc:
        if on_error == "raise":
            raise
        tb = traceback.format_exc()
        logger.error("Document id=%s failed; emitting empty graph (--continue-on-error).", doc_id)
        logger.debug("%s", tb)
        empty_g = empty_result_graph(
            doc_id,
            original_text,
            source=source,
            category=category,
            extra_meta=extra_meta,
            error_message=str(exc),
        )
        if stats:
            stats.record_error(doc_id, exc)
            stats.record_graph(empty_g, original_text)
        return empty_g
