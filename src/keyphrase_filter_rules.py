"""
Rule-based low-information filtering for KeyBERT phrases (no large manual blacklist).

Uses small closed-class / function-word sets plus structural checks. Optional
``extra_blocklist`` matches the whole normalized phrase for residual junk.
"""

from __future__ import annotations

import re
from src.normalization import normalize_for_dedup, rough_word_tokens

# Closed-class words (lowercase). Not exhaustive linguistics — MVP coverage for RU/EN mix.
RU_FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        "и",
        "а",
        "но",
        "да",
        "или",
        "либо",
        "ни",
        "как",
        "что",
        "чтобы",
        "же",
        "ли",
        "бы",
        "не",
        "нет",
        "нибудь",
        "это",
        "то",
        "вот",
        "все",
        "всё",
        "весь",
        "сам",
        "само",
        "так",
        "уже",
        "ещё",
        "еще",
        "там",
        "тут",
        "где",
        "куда",
        "откуда",
        "когда",
        "пока",
        "если",
        "хотя",
        "пусть",
        "лишь",
        "только",
        "очень",
        "он",
        "она",
        "оно",
        "они",
        "мы",
        "вы",
        "ты",
        "я",
        "мне",
        "меня",
        "нас",
        "вам",
        "вас",
        "ему",
        "его",
        "её",
        "их",
        "мой",
        "твой",
        "наш",
        "ваш",
        "свой",
        "этот",
        "эта",
        "это",
        "эти",
        "тот",
        "та",
        "те",
        "такой",
        "в",
        "во",
        "на",
        "под",
        "над",
        "за",
        "перед",
        "при",
        "про",
        "для",
        "без",
        "до",
        "от",
        "из",
        "к",
        "ко",
        "у",
        "о",
        "об",
        "по",
        "с",
        "со",
        "между",
        "сквозь",
        "через",
        "можно",
        "нужно",
        "надо",
        "будто",
        "ведь",
        "разве",
        "почему",
        "зачем",
        "кто",
        "чего",
        "чем",
        "какой",
        "какая",
        "какие",
        "какое",
        "сколько",
        "почти",
        "едва",
        "вряд",
    }
)

EN_FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "where",
        "why",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "not",
        "no",
        "nor",
        "so",
        "as",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "into",
        "from",
        "to",
        "in",
        "on",
        "up",
        "down",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "just",
        "also",
        "both",
        "any",
    }
)

_PUNCT_OR_NON_ALNUM = re.compile(r"^[\s\W\d_]+$", re.UNICODE)


def all_function_words() -> frozenset[str]:
    """RU ∪ EN function words (lowercase)."""
    return RU_FUNCTION_WORDS | EN_FUNCTION_WORDS


def has_letter_or_digit(s: str) -> bool:
    """True if any character is a Unicode letter or decimal digit."""
    return any(ch.isalpha() or ch.isdecimal() for ch in s)


def is_punct_or_symbol_junk(phrase_normalized: str) -> bool:
    """No real word tokens: only punctuation, spaces, underscore, or digits-only shell."""
    return bool(_PUNCT_OR_NON_ALNUM.match(phrase_normalized))


def phrase_is_only_function_words(phrase_normalized: str, function_words: frozenset[str]) -> bool:
    """Every token is in ``function_words``, or there are no tokens after rough split."""
    tokens = rough_word_tokens(phrase_normalized)
    if not tokens:
        return True
    return all(t in function_words for t in tokens)


def single_token_too_short(phrase_normalized: str, min_len: int) -> bool:
    tokens = rough_word_tokens(phrase_normalized)
    return len(tokens) == 1 and len(tokens[0]) < min_len


def is_low_information_phrase(
    phrase_normalized: str,
    *,
    function_words: frozenset[str],
    extra_blocklist: frozenset[str],
    single_token_min_len: int,
) -> bool:
    """
    Return True if the phrase should be dropped (after ``normalize_text``).

    Rules (any match → drop):
    - empty / whitespace-only
    - only punctuation-like characters (no word letters as rough_word_tokens would use)
    - no letters and no decimal digits anywhere
    - all tokens are function words (RU/EN closed-class)
    - single token shorter than ``single_token_min_len``
    - whole-phrase dedup key in ``extra_blocklist``
    """
    t = phrase_normalized.strip()
    if not t:
        return True
    if is_punct_or_symbol_junk(t):
        return True
    if not has_letter_or_digit(t):
        return True
    if phrase_is_only_function_words(t, function_words):
        return True
    if single_token_too_short(t, single_token_min_len):
        return True
    if normalize_for_dedup(t) in extra_blocklist:
        return True
    return False
