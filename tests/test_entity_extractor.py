from src.entity_extractor import resolve_overlapping_entities
from src.schemas import RawEntity


def test_resolve_overlaps_keeps_higher_score():
    a = RawEntity(text="foo bar", label="x", score=0.9, start=0, end=7)
    b = RawEntity(text="foo", label="x", score=0.5, start=0, end=3)
    out = resolve_overlapping_entities([a, b], strategy="keep_highest_score")
    assert len(out) == 1
    assert out[0].text == "foo bar"


def test_resolve_none_returns_copy():
    e = RawEntity(text="x", label="l", score=1.0, start=0, end=1)
    out = resolve_overlapping_entities([e], strategy="none")
    assert len(out) == 1
