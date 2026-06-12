"""
End-to-end pipeline smoke: full orchestration with ML mocked (no downloaded weights).

Checks JSON contract (round-trip through model_dump) and that non-empty synthetic
extractions yield merged_nodes and at least one edge (same_prompt for 2+ nodes).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.config import PipelineConfig
from src.pipeline import PseudoGraphPipeline
from src.schemas import DocumentGraph, RawEntity


def test_pipeline_e2e_synthetic_nonempty_has_nodes_and_edges() -> None:
    text = "Альфа бета"
    entities = [
        RawEntity(text="Альфа", label="thing", score=0.9, start=0, end=5),
        RawEntity(text="бета", label="thing", score=0.85, start=6, end=10),
    ]
    cfg = PipelineConfig()
    pipeline = PseudoGraphPipeline(cfg)
    pipeline.gliner.extract = MagicMock(return_value=entities)
    pipeline.keyphrase_extractor.extract = MagicMock(return_value=[])

    graph = pipeline.process_one("e2e_1", text)

    round_trip = DocumentGraph.model_validate(graph.model_dump())
    assert round_trip.id == graph.id

    assert len(graph.merged_nodes) >= 2
    assert len(graph.edges) > 0
    assert any(e.edge_type == "same_prompt" for e in graph.edges)

    assert graph.metadata.num_entities == len(graph.entities)
    assert graph.metadata.num_keyphrases == len(graph.keyphrases)
    assert graph.metadata.num_nodes == len(graph.merged_nodes)
    assert graph.metadata.num_edges == len(graph.edges)


def test_pipeline_e2e_whitespace_only_valid_empty_graph() -> None:
    cfg = PipelineConfig()
    pipeline = PseudoGraphPipeline(cfg)
    pipeline.gliner.extract = MagicMock(return_value=[])
    pipeline.keyphrase_extractor.extract = MagicMock(return_value=[])

    graph = pipeline.process_one("e2e_empty", "  \n\t  ")
    DocumentGraph.model_validate(graph.model_dump())

    assert graph.merged_nodes == []
    assert graph.edges == []
    assert graph.metadata.num_nodes == 0
    assert graph.metadata.num_edges == 0
