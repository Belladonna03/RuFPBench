import pytest

from src.normalization import normalize_for_dedup, normalize_text


def test_normalize_trims_and_spaces():
    assert normalize_text("  a  b  ") == "a b"
    assert normalize_text("x\t\t y") == "x y"


def test_normalize_newlines():
    assert normalize_text("a\n\n\nb") == "a\n\nb"


def test_normalize_for_dedup_lowercase():
    assert normalize_for_dedup("  Hello  World  ") == "hello world"


def test_empty():
    assert normalize_text("") == ""
    assert normalize_for_dedup("") == ""
