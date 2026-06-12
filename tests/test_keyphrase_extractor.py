"""post_filter_keyphrases: rule-based low-information filtering (see keyphrase_filter_rules)."""

from src.keyphrase_extractor import post_filter_keyphrases


def _texts(rows):
    return [x.text for x in rows]


def test_only_function_words_russian_dropped():
    cands = [("как и для", 0.9), ("нормальная фраза", 0.5)]
    out = post_filter_keyphrases(cands, frozenset(), min_chars=2, max_chars=256)
    assert _texts(out) == ["нормальная фраза"]


def test_only_function_words_english_dropped():
    cands = [("the and or", 0.9), ("machine learning", 0.6)]
    out = post_filter_keyphrases(cands, frozenset(), min_chars=2, max_chars=256)
    assert _texts(out) == ["machine learning"]


def test_substantive_multilingual_phrase_kept():
    cands = [("обработка данных и API", 0.7)]
    out = post_filter_keyphrases(cands, frozenset(), min_chars=2, max_chars=256)
    assert len(out) == 1
    assert "обработка" in out[0].text


def test_punctuation_junk_dropped():
    cands = [("!!!", 0.9), ("???", 0.8), ("ухо глаз", 0.5)]
    out = post_filter_keyphrases(cands, frozenset(), min_chars=2, max_chars=256)
    assert _texts(out) == ["ухо глаз"]


def test_short_single_token_dropped():
    cands = [("ab", 0.9), ("abcd", 0.5)]
    out = post_filter_keyphrases(
        cands,
        frozenset(),
        min_chars=2,
        max_chars=256,
        single_token_min_chars=3,
    )
    assert _texts(out) == ["abcd"]


def test_no_letters_or_digits_dropped():
    cands = [("———", 0.9), ("тест", 0.5)]
    out = post_filter_keyphrases(cands, frozenset(), min_chars=2, max_chars=256)
    assert _texts(out) == ["тест"]


def test_extra_blocklist_whole_phrase():
    cands = [("specialjunk", 0.9), ("good phrase here", 0.5)]
    out = post_filter_keyphrases(
        cands,
        frozenset({"specialjunk"}),
        min_chars=2,
        max_chars=256,
    )
    assert _texts(out) == ["good phrase here"]


def test_dedup_by_normalized_form():
    cands = [("Hello", 0.9), ("hello", 0.8)]
    out = post_filter_keyphrases(cands, frozenset(), min_chars=2, max_chars=256)
    assert len(out) == 1
