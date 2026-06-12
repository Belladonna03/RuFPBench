from src.graph_builder import (
    build_cooccurrence_edges,
    interval_gap_half_open,
    split_sentences_with_spans,
)
from src.schemas import MergedNode, RawMention


def _edge_triples(edges):
    return {(e.source_node_id, e.target_node_id, e.edge_type) for e in edges}


def test_split_sentences_keeps_indices():
    text = "Первое. Второе\nТретье"
    spans = split_sentences_with_spans(text)
    assert len(spans) >= 2
    for s in spans:
        assert text[s.start : s.end].strip()


def test_split_empty_text():
    assert split_sentences_with_spans("") == []


def test_split_whitespace_only():
    assert split_sentences_with_spans("   \n\n  \t  ") == []


def test_split_single_sentence_no_punct():
    text = "одно целое предложение"
    spans = split_sentences_with_spans(text)
    assert len(spans) == 1
    assert spans[0].sentence_id == 0
    assert text[spans[0].start : spans[0].end] == text.strip()


def test_split_multiple_punct_run():
    text = "Стоп!!! \nДальше"
    spans = split_sentences_with_spans(text)
    assert len(spans) == 2
    assert "Стоп" in text[spans[0].start : spans[0].end]


def test_interval_gap_half_open():
    assert interval_gap_half_open(0, 2, 2, 4) == 0  # touching
    assert interval_gap_half_open(0, 2, 5, 7) == 3
    assert interval_gap_half_open(0, 10, 3, 5) == 0  # overlap


def test_same_prompt_same_sentence_and_ordering():
    text = "Альфа бета. Гамма"
    nodes = [
        MergedNode(
            node_id="n1",
            canonical_text="альфа",
            normalized_text="альфа",
            raw_mentions=[
                RawMention(
                    text="Альфа",
                    label="x",
                    extraction_source="gliner",
                    score=0.9,
                    start=0,
                    end=5,
                )
            ],
            labels=["x"],
            sources=["gliner"],
            max_score=0.9,
            mention_count=1,
        ),
        MergedNode(
            node_id="n2",
            canonical_text="бета",
            normalized_text="бета",
            raw_mentions=[
                RawMention(
                    text="бета",
                    label="x",
                    extraction_source="gliner",
                    score=0.8,
                    start=6,
                    end=10,
                )
            ],
            labels=["x"],
            sources=["gliner"],
            max_score=0.8,
            mention_count=1,
        ),
        MergedNode(
            node_id="n3",
            canonical_text="гамма",
            normalized_text="гамма",
            raw_mentions=[
                RawMention(
                    text="Гамма",
                    label="x",
                    extraction_source="gliner",
                    score=0.7,
                    start=text.index("Гамма"),
                    end=text.index("Гамма") + len("Гамма"),
                )
            ],
            labels=["x"],
            sources=["gliner"],
            max_score=0.7,
            mention_count=1,
        ),
    ]
    edges = build_cooccurrence_edges(text, nodes, nearby_window_chars=20)
    types = _edge_triples(edges)
    assert any(t[2] == "same_prompt" for t in types)
    assert ("n1", "n2", "same_sentence") in types
    for e in edges:
        assert e.source_node_id <= e.target_node_id


def test_edges_deterministic_two_runs():
    text = "a. b\nc"
    nodes = [
        MergedNode(
            node_id="z",
            canonical_text="z",
            normalized_text="z",
            raw_mentions=[
                    RawMention(
                        text="z",
                        label="l",
                        extraction_source="gliner",
                        score=0.5,
                        start=0,
                        end=1,
                    ),
                ],
            labels=["l"],
            sources=["gliner"],
            max_score=0.5,
            mention_count=1,
        ),
        MergedNode(
            node_id="y",
            canonical_text="y",
            normalized_text="y",
            raw_mentions=[
                    RawMention(
                        text="y",
                        label="l",
                        extraction_source="gliner",
                        score=0.5,
                        start=4,
                        end=5,
                    ),
                ],
            labels=["l"],
            sources=["gliner"],
            max_score=0.5,
            mention_count=1,
        ),
    ]
    a = build_cooccurrence_edges(text, nodes, nearby_window_chars=10)
    b = build_cooccurrence_edges(text, nodes, nearby_window_chars=10)
    assert [e.model_dump() for e in a] == [e.model_dump() for e in b]


def test_same_sentence_multiple_shared_sentences():
    text = "one two. three four."
    # nA: mentions in sentence 0 and 1; nB: same
    nodes = [
        MergedNode(
            node_id="na",
            canonical_text="a",
            normalized_text="a",
                raw_mentions=[
                    RawMention(
                        text="o",
                        label="x",
                        extraction_source="gliner",
                        score=0.9,
                        start=0,
                        end=2,
                    ),
                    RawMention(
                        text="three",
                        label="x",
                        extraction_source="gliner",
                        score=0.8,
                        start=text.index("three"),
                        end=text.index("three") + 5,
                    ),
                ],
            labels=["x"],
            sources=["gliner"],
            max_score=0.9,
            mention_count=2,
        ),
        MergedNode(
            node_id="nb",
            canonical_text="b",
            normalized_text="b",
                raw_mentions=[
                    RawMention(
                        text="two",
                        label="x",
                        extraction_source="gliner",
                        score=0.7,
                        start=4,
                        end=7,
                    ),
                    RawMention(
                        text="four",
                        label="x",
                        extraction_source="gliner",
                        score=0.6,
                        start=text.index("four"),
                        end=text.index("four") + 4,
                    ),
                ],
            labels=["x"],
            sources=["gliner"],
            max_score=0.7,
            mention_count=2,
        ),
    ]
    edges = build_cooccurrence_edges(text, nodes, nearby_window_chars=5)
    ss = [e for e in edges if e.edge_type == "same_sentence" and {e.source_node_id, e.target_node_id} == {"na", "nb"}]
    assert len(ss) == 1
    assert set(ss[0].evidence.sentence_ids) == {0, 1}


def test_nearby_window_uses_min_gap_between_intervals():
    text = "abcdefghij"
    nodes = [
        MergedNode(
            node_id="a",
            canonical_text="a",
            normalized_text="a",
            raw_mentions=[
                RawMention(
                    text="a",
                    label="k",
                    extraction_source="keybert",
                    score=0.5,
                    start=0,
                    end=1,
                ),
                RawMention(
                    text="cde",
                    label="k",
                    extraction_source="keybert",
                    score=0.4,
                    start=2,
                    end=5,
                ),
            ],
            labels=["keyphrase"],
            sources=["keybert"],
            max_score=0.5,
            mention_count=2,
        ),
        MergedNode(
            node_id="b",
            canonical_text="b",
            normalized_text="b",
            raw_mentions=[
                RawMention(
                    text="b",
                    label="k",
                    extraction_source="keybert",
                    score=0.5,
                    start=6,
                    end=7,
                )
            ],
            labels=["keyphrase"],
            sources=["keybert"],
            max_score=0.5,
            mention_count=1,
        ),
    ]
    edges = build_cooccurrence_edges(text, nodes, nearby_window_chars=10)
    nw = [e for e in edges if e.edge_type == "nearby_window" and {e.source_node_id, e.target_node_id} == {"a", "b"}]
    assert len(nw) == 1
    # min gap: from [2,5) to [6,7) is 1 (between 5 and 6)
    assert nw[0].evidence.char_distance == 1
    assert nw[0].weight > 0.05


def test_nearby_window_requires_offsets():
    text = "abcdefghij"
    nodes = [
        MergedNode(
            node_id="a",
            canonical_text="a",
            normalized_text="a",
            raw_mentions=[
                RawMention(
                    text="a",
                    label="k",
                    extraction_source="keybert",
                    score=0.5,
                    start=0,
                    end=1,
                )
            ],
            labels=["keyphrase"],
            sources=["keybert"],
            max_score=0.5,
            mention_count=1,
        ),
        MergedNode(
            node_id="b",
            canonical_text="b",
            normalized_text="b",
            raw_mentions=[
                RawMention(
                    text="b",
                    label="k",
                    extraction_source="keybert",
                    score=0.5,
                    start=2,
                    end=3,
                )
            ],
            labels=["keyphrase"],
            sources=["keybert"],
            max_score=0.5,
            mention_count=1,
        ),
    ]
    edges = build_cooccurrence_edges(text, nodes, nearby_window_chars=5)
    nw = [e for e in edges if e.edge_type == "nearby_window"]
    assert nw


def test_same_prompt_fixed_weight():
    text = "x y"
    nodes = [
        MergedNode(
            node_id="n1",
            canonical_text="n1",
            normalized_text="n1",
            raw_mentions=[
                RawMention(
                    text="x",
                    label="l",
                    extraction_source="gliner",
                    score=1.0,
                    start=0,
                    end=1,
                )
            ],
            labels=["l"],
            sources=["gliner"],
            max_score=1.0,
            mention_count=1,
        ),
        MergedNode(
            node_id="n2",
            canonical_text="n2",
            normalized_text="n2",
            raw_mentions=[
                RawMention(
                    text="y",
                    label="l",
                    extraction_source="gliner",
                    score=1.0,
                    start=2,
                    end=3,
                )
            ],
            labels=["l"],
            sources=["gliner"],
            max_score=1.0,
            mention_count=1,
        ),
    ]
    sp = [e for e in build_cooccurrence_edges(text, nodes) if e.edge_type == "same_prompt"]
    assert len(sp) == 1
    assert sp[0].weight == 1.0
