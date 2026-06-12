"""Cheap checks that the public contract (imports, schemas, merge) holds."""

from src.merge_dedup import merge_entities_and_keyphrases, stable_node_id
from src.schemas import DocumentGraph, RawEntity


def test_document_graph_minimal_json_roundtrip():
    g = DocumentGraph(id="1", original_text="a", normalized_text="a")
    g2 = DocumentGraph.model_validate_json(g.model_dump_json())
    assert g2.id == "1"
    assert g2.merged_nodes == []


def test_stable_node_id_contract():
    nid = stable_node_id("doc_a", "hello", ["location", "person"])
    assert nid.startswith("n_")
    assert len(nid) == 2 + 12


def test_merge_produces_graph_ready_nodes():
    e = RawEntity(text="X", label="t", score=1.0, start=0, end=1)
    nodes = merge_entities_and_keyphrases([e], [], doc_id="d1", soft_merge_same_normalized=False)
    assert len(nodes) == 1
    assert nodes[0].node_id == stable_node_id("d1", nodes[0].canonical_text, nodes[0].labels)
