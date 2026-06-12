from unittest.mock import MagicMock

import pytest

from src.config import PipelineConfig
from src.pipeline import (
    PseudoGraphPipeline,
    RunStats,
    empty_result_graph,
    process_one_safe,
)
from src.schemas import RawEntity


def test_process_one_empty_text_stable_output():
    cfg = PipelineConfig()
    p = PseudoGraphPipeline(cfg)
    p.gliner.extract = MagicMock(return_value=[])
    p.keyphrase_extractor.extract = MagicMock(return_value=[])

    g = p.process_one("doc", "   \n\t  ")
    assert g.entities == []
    assert g.keyphrases == []
    assert g.merged_nodes == []
    assert g.edges == []
    assert g.metadata.num_entities == 0
    assert g.metadata.num_keyphrases == 0


def test_process_one_very_short_text():
    cfg = PipelineConfig()
    p = PseudoGraphPipeline(cfg)
    p.gliner.extract = MagicMock(return_value=[])
    p.keyphrase_extractor.extract = MagicMock(return_value=[])

    g = p.process_one("d", "ab")
    assert g.id == "d"
    assert "ab" in g.original_text or g.original_text == "ab"


def test_process_one_safe_continue_on_error():
    cfg = PipelineConfig()
    p = PseudoGraphPipeline(cfg)
    p.gliner.extract = MagicMock(side_effect=RuntimeError("model failure"))
    stats = RunStats()

    g = process_one_safe(
        p,
        "doc1",
        "hello",
        extra_meta={"k": 1},
        on_error="empty",
        stats=stats,
    )
    assert g.metadata.extra.get("pipeline_error") == "model failure"
    assert g.metadata.extra.get("k") == 1
    assert stats.errors == 1
    assert stats.documents == 1


def test_process_one_safe_reraise():
    cfg = PipelineConfig()
    p = PseudoGraphPipeline(cfg)
    p.gliner.extract = MagicMock(side_effect=RuntimeError("fail"))

    with pytest.raises(RuntimeError):
        process_one_safe(p, "x", "y", on_error="raise")


def test_process_one_safe_invalid_on_error():
    cfg = PipelineConfig()
    p = PseudoGraphPipeline(cfg)
    p.gliner.extract = MagicMock(
        return_value=[RawEntity(text="a", label="l", score=1.0, start=0, end=1)]
    )
    p.keyphrase_extractor.extract = MagicMock(return_value=[])
    with pytest.raises(ValueError, match="on_error"):
        process_one_safe(p, "id", "a", on_error="invalid")  # type: ignore[arg-type]


def test_run_stats_record_graph():
    from src.schemas import DocumentGraph, DocumentMetadata

    s = RunStats()
    g = DocumentGraph(
        id="1",
        original_text="x",
        normalized_text="x",
        metadata=DocumentMetadata(num_entities=0, num_keyphrases=0, num_nodes=0, num_edges=0),
    )
    s.record_graph(g, "   ")
    assert s.empty_text == 1
    assert s.no_entities == 1
    assert s.no_keyphrases == 1


def test_empty_result_graph_has_error_key():
    g = empty_result_graph("id1", "t", error_message="oops")
    assert g.metadata.extra["pipeline_error"] == "oops"
