from src.merge_dedup import (
    group_by_normalized_and_label,
    merge_entities_and_keyphrases,
    soft_merge_by_normalized_text,
    stable_node_id,
)
from src.schemas import RawEntity, RawKeyphrase


def test_group_by_normalized_and_label_distinct_labels():
    e1 = RawEntity(text="Москва", label="location", score=0.9, start=0, end=6)
    e2 = RawEntity(text="москва", label="location", score=0.8, start=10, end=16)
    k1 = RawKeyphrase(text="Москва", score=0.5)
    nodes = group_by_normalized_and_label([e1, e2], [k1])
    # same norm + same entity label -> one bucket; keyphrase separate label
    assert len(nodes) == 2
    by_labels = {tuple(n.labels): n for n in nodes}
    loc = by_labels[("location",)]
    assert loc.mention_count == 2
    assert loc.max_score == 0.9


def test_soft_merge_unions_labels():
    e = RawEntity(text="Test", label="person", score=0.9, start=0, end=4)
    k = RawKeyphrase(text="test", score=0.4)
    nodes = group_by_normalized_and_label([e], [k])
    assert len(nodes) == 2
    merged = soft_merge_by_normalized_text(nodes)
    assert len(merged) == 1
    assert set(merged[0].labels) == {"person", "keyphrase"}
    assert merged[0].mention_count == 2


def test_merge_assigns_ids():
    e = RawEntity(text="A", label="x", score=1.0, start=0, end=1)
    doc_id = "doc_test"
    nodes = merge_entities_and_keyphrases([e], [], doc_id=doc_id, soft_merge_same_normalized=False)
    assert len(nodes) == 1
    assert nodes[0].node_id == stable_node_id(doc_id, nodes[0].canonical_text, nodes[0].labels)
    assert nodes[0].node_id.startswith("n_")
    assert len(nodes[0].node_id) == 2 + 12  # "n_" + 12 hex chars


def test_merge_node_ids_deterministic_per_doc():
    e = RawEntity(text="Москва", label="location", score=0.9, start=0, end=6)
    k = RawKeyphrase(text="Москва", score=0.5)
    doc_id = "row_42"
    a = merge_entities_and_keyphrases([e], [k], doc_id=doc_id, soft_merge_same_normalized=True)
    b = merge_entities_and_keyphrases([e], [k], doc_id=doc_id, soft_merge_same_normalized=True)
    assert [n.model_dump() for n in a] == [n.model_dump() for n in b]
    other = merge_entities_and_keyphrases([e], [k], doc_id="other_doc", soft_merge_same_normalized=True)
    assert other[0].node_id != a[0].node_id
