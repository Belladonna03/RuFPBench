"""Russian / mixed-language EDA helpers: text cleanup, NLTK stopwords, optional pymorphy3/pymorphy2 lemmas."""

from __future__ import annotations

import html
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from shared.logging_utils import get_logger

_log = get_logger("shared.ru_eda_text")

_HTML_TAG_RE = re.compile(r"<[^>]+>", re.IGNORECASE)
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_DIGIT_RUN_RE = re.compile(r"\d+")
_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+", re.UNICODE)

# Tiny technical noise list (HTML/XML leftovers); main stoplist is NLTK Russian (+ English for mixed corpora).
_TECH_BLACKLIST = frozenset(
    {"br", "nbsp", "amp", "quot", "lt", "gt", "ndash", "mdash", "hellip", "laquo", "raquo", "apos"},
)

_STOPWORDS_CACHE: tuple[frozenset[str], frozenset[str]] | None = None
_MORPH: Any | None = False  # False = uninitialized, None = unavailable


def _try_morph_analyzer() -> Any | None:
    """Prefer pymorphy3 (Python 3.11+), then pymorphy2; return None if both fail."""
    for modname in ("pymorphy3", "pymorphy2"):
        try:
            mod = __import__(modname, fromlist=["MorphAnalyzer"])
            return mod.MorphAnalyzer()
        except Exception as e:
            _log.debug("MorphAnalyzer %s: %s", modname, e)
            continue
    return None


def normalize_text_cell(raw: Any) -> str:
    """Safe string for EDA: None / NaN / non-string → empty or str."""
    if raw is None:
        return ""
    try:
        import pandas as pd

        if pd.isna(raw):
            return ""
    except Exception:
        pass
    try:
        if raw != raw:  # float NaN without pandas
            return ""
    except (TypeError, ValueError):
        pass
    s = str(raw)
    if s.lower() in ("nan", "<na>", "none", "nat"):
        return ""
    return s


def clean_text_for_word_stats(text: Any) -> str:
    """
    Normalize for tokenization: lowercase, unescape entities, strip tags/URLs/digits, collapse spaces.
    """
    s = normalize_text_cell(text)
    if not s:
        return ""
    s = s.lower()
    s = html.unescape(s)
    s = _HTML_TAG_RE.sub(" ", s)
    s = _URL_RE.sub(" ", s)
    s = _DIGIT_RUN_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _ensure_nltk_stopwords() -> tuple[frozenset[str], frozenset[str]]:
    global _STOPWORDS_CACHE
    if _STOPWORDS_CACHE is not None:
        return _STOPWORDS_CACHE
    try:
        from nltk.corpus import stopwords as nltk_sw

        ru = frozenset(w.lower() for w in nltk_sw.words("russian"))
        en = frozenset(w.lower() for w in nltk_sw.words("english"))
    except LookupError:
        try:
            import nltk

            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords as nltk_sw

            ru = frozenset(w.lower() for w in nltk_sw.words("russian"))
            en = frozenset(w.lower() for w in nltk_sw.words("english"))
        except Exception as e:
            _log.warning("NLTK stopwords unavailable (%s); using technical blacklist only.", e)
            ru, en = frozenset(), frozenset()
    except Exception as e:
        _log.warning("NLTK stopwords failed (%s); using technical blacklist only.", e)
        ru, en = frozenset(), frozenset()

    _STOPWORDS_CACHE = (ru, en)
    return _STOPWORDS_CACHE


def combined_stopwords(*, include_english: bool) -> frozenset[str]:
    ru, en = _ensure_nltk_stopwords()
    out = set(ru) | _TECH_BLACKLIST
    if include_english:
        out |= set(en)
    return frozenset(out)


def _get_morph():
    global _MORPH
    if _MORPH is not False:
        return _MORPH
    _MORPH = _try_morph_analyzer()
    if _MORPH is None:
        _log.info("pymorphy3/2 not available; top words use surface forms (no lemmatization).")
    return _MORPH


def _lemma(token: str, morph: Any) -> str:
    if morph is None:
        return token.lower()
    try:
        return str(morph.parse(token)[0].normal_form).lower()
    except Exception:
        return token.lower()


def _morph_backend_label(morph: Any) -> str | None:
    if morph is None:
        return None
    mod = type(morph).__module__
    if "pymorphy3" in mod:
        return "pymorphy3"
    if "pymorphy2" in mod:
        return "pymorphy2"
    return "morph"


def iter_content_lemmas(
    text: Any,
    *,
    morph: Any,
    stops: frozenset[str],
    min_len: int = 3,
) -> Iterator[str]:
    s = clean_text_for_word_stats(text)
    if not s:
        return
    for m in _TOKEN_RE.finditer(s):
        tok = m.group(0).lower()
        if len(tok) < min_len or tok.isdigit():
            continue
        lem = _lemma(tok, morph)
        if len(lem) < min_len or lem.isdigit():
            continue
        if lem in stops:
            continue
        yield lem


def extract_top_ru_content_words(
    texts: Iterable[str],
    *,
    top_k: int = 20,
    min_token_len: int = 3,
    include_english_stopwords: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Count content words (optional lemmas via pymorphy3/pymorphy2), NLTK RU (+ EN) stopwords, Counter.

    Returns:
        rows: [{"word": lemma_or_token, "count": n}, ...] length ≤ top_k
        meta: diagnostics for EDA reports
    """
    morph = _get_morph()
    ru_sw, en_sw = _ensure_nltk_stopwords()
    has_nltk = bool(ru_sw) or bool(en_sw)
    stops = combined_stopwords(include_english=include_english_stopwords)
    cnt: Counter[str] = Counter()
    for t in texts:
        for lem in iter_content_lemmas(t, morph=morph, stops=stops, min_len=min_token_len):
            cnt[lem] += 1
    rows = [{"word": w, "count": c} for w, c in cnt.most_common(top_k)]
    if has_nltk:
        sw_label = "nltk_russian" + ("+nltk_english" if include_english_stopwords else "")
    else:
        sw_label = "technical_blacklist_only"
    meta = {
        "lemmatized": morph is not None,
        "lemma_backend": _morph_backend_label(morph),
        "stopwords": sw_label,
        "min_token_len": min_token_len,
        "top_k": top_k,
    }
    return rows, meta


def plot_top_content_words(
    rows: list[dict[str, Any]],
    out_path: str | Path,
    *,
    title: str | None = None,
) -> None:
    """Horizontal bar chart for precomputed top-word rows; no-op if rows empty."""
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    words = [r["word"] for r in rows]
    counts = [int(r["count"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.22 * len(words))))
    ax.barh(words[::-1], counts[::-1], color="teal")
    ax.set_title(title or f"Top {len(rows)} content words")
    ax.set_xlabel("count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


__all__ = [
    "clean_text_for_word_stats",
    "normalize_text_cell",
    "extract_top_ru_content_words",
    "plot_top_content_words",
    "combined_stopwords",
    "iter_content_lemmas",
]
