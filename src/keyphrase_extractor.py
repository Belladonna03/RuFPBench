"""KeyBERT + sentence-transformers backend, filters, and approximate offsets."""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.config import PipelineConfig
from src.keyphrase_filter_rules import all_function_words, is_low_information_phrase
from src.normalization import normalize_for_dedup, normalize_text
from src.schemas import RawKeyphrase

logger = logging.getLogger(__name__)


def keybert_raw_to_candidates(keywords: Any) -> list[tuple[str, float]]:
    """Normalize KeyBERT return value to (phrase, score) pairs."""
    out: list[tuple[str, float]] = []
    if not keywords:
        return out
    for item in keywords:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                out.append((str(item[0]), float(item[1])))
            except (TypeError, ValueError):
                logger.debug("Skip bad KeyBERT tuple: %r", item)
        elif isinstance(item, str):
            out.append((item, 0.0))
    return out


def approximate_phrase_span(phrase: str, full_text: str) -> tuple[Optional[int], Optional[int]]:
    """
    First literal match of phrase in full_text; then case-insensitive match (casefold).
    Indices refer to full_text codepoints. Returns (None, None) if not found reliably.
    """
    if not phrase or not full_text:
        return None, None

    i = full_text.find(phrase)
    if i != -1:
        return i, i + len(phrase)

    lo_t = full_text.casefold()
    lo_p = phrase.casefold()
    j = lo_t.find(lo_p)
    if j == -1:
        return None, None
    end = j + len(phrase)
    if end > len(full_text):
        return None, None
    return j, end


def post_filter_keyphrases(
    candidates: list[tuple[str, float]],
    extra_blocklist: frozenset[str],
    min_chars: int = 2,
    max_chars: int = 256,
    *,
    full_text: Optional[str] = None,
    single_token_min_chars: int = 3,
    function_words: Optional[frozenset[str]] = None,
) -> list[RawKeyphrase]:
    """
    Dedup and drop low-information phrases (rule-based; see ``keyphrase_filter_rules``).

    ``extra_blocklist``: optional exact normalized whole-phrase strings to remove.
    """
    fw = function_words if function_words is not None else all_function_words()
    out: list[RawKeyphrase] = []
    seen_norm: set[str] = set()

    for phrase, score in candidates:
        try:
            sc = float(score)
        except (TypeError, ValueError):
            sc = 0.0
        if sc < 0.0:
            sc = 0.0

        t = normalize_text(phrase)
        if len(t) < min_chars or len(t) > max_chars:
            continue
        if len(t.strip()) <= 1:
            continue
        if is_low_information_phrase(
            t,
            function_words=fw,
            extra_blocklist=extra_blocklist,
            single_token_min_len=single_token_min_chars,
        ):
            continue

        dedup_key = normalize_for_dedup(t)
        if dedup_key in seen_norm:
            continue
        seen_norm.add(dedup_key)

        start: Optional[int] = None
        end: Optional[int] = None
        if full_text is not None:
            start, end = approximate_phrase_span(t, full_text)

        out.append(
            RawKeyphrase(
                text=t,
                score=sc,
                start=start,
                end=end,
            )
        )
    return out


class KeyphraseExtractor:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._kw: Any = None

    def _load(self) -> Any:
        if self._kw is None:
            from keybert import KeyBERT
            from sentence_transformers import SentenceTransformer

            st = SentenceTransformer(self.config.keybert_model, device=self.config.device)
            self._kw = KeyBERT(model=st)
        return self._kw

    def extract(self, text: str) -> list[RawKeyphrase]:
        if not text.strip():
            return []
        if not normalize_text(text):
            return []

        try:
            kw_model = self._load()
            keywords = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=self.config.keyphrase_ngram_range,
                top_n=self.config.top_n_keyphrases,
                use_mmr=self.config.use_mmr,
                diversity=self.config.diversity,
            )
        except Exception:
            logger.exception("KeyBERT extract_keywords failed")
            raise

        candidates = keybert_raw_to_candidates(keywords)
        return post_filter_keyphrases(
            candidates,
            extra_blocklist=self.config.keyphrase_stop_phrases,
            min_chars=self.config.keyphrase_min_chars,
            max_chars=self.config.keyphrase_max_chars,
            full_text=text,
            single_token_min_chars=self.config.keyphrase_single_token_min_chars,
        )
