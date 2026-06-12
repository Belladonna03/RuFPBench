"""GLiNER zero-shot entity extraction + optional overlap resolution (separate step)."""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.schemas import RawEntity

logger = logging.getLogger(__name__)


def _safe_number_to_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def _safe_float_score(val: Any) -> float:
    try:
        s = float(val)
    except (TypeError, ValueError):
        return 0.0
    return s if s >= 0.0 else 0.0


def _clip_span(start: int, end: int, text_len: int) -> tuple[int, int]:
    """Clamp to [0, text_len] and ensure end >= start."""
    start = max(0, min(start, text_len))
    end = max(0, min(end, text_len))
    if end < start:
        start, end = end, start
    return start, end


def parse_gliner_prediction(item: Any, source_text: str) -> Optional[RawEntity]:
    """
    Turn one GLiNER prediction dict into RawEntity, or None if unusable.
    Tolerates missing keys, bad types, and out-of-range spans (clipped to text).
    """
    if not isinstance(item, dict):
        logger.debug("Skip non-dict GLiNER prediction: %r", type(item).__name__)
        return None

    n = len(source_text)
    label = str(item.get("label", "") or "").strip()
    if not label:
        return None

    start_raw = _safe_number_to_int(item.get("start"))
    end_raw = _safe_number_to_int(item.get("end"))
    if start_raw is None:
        start_raw = 0
    if end_raw is None:
        end_raw = start_raw

    start, end = _clip_span(start_raw, end_raw, n)

    span_text = str(item.get("text", "") or "")
    if not span_text.strip():
        if start < end:
            span_text = source_text[start:end]
        else:
            return None

    score = _safe_float_score(item.get("score"))

    try:
        return RawEntity(
            text=span_text,
            label=label,
            score=score,
            start=start,
            end=end,
        )
    except Exception as ex:
        logger.warning("Skip GLiNER span after validation: %r (%s)", item, ex)
        return None


def entities_from_gliner_raw(raw: Any, source_text: str) -> list[RawEntity]:
    """Parse full model output (list or None) into entities."""
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        logger.warning("Unexpected GLiNER predict_entities return type: %s", type(raw).__name__)
        return []

    out: list[RawEntity] = []
    for item in raw:
        ent = parse_gliner_prediction(item, source_text)
        if ent is not None:
            out.append(ent)
    return out


class GlinerEntityExtractor:
    """Lazy-loaded GLiNER model."""

    def __init__(
        self,
        model_name: str,
        labels: list[str],
        threshold: float = 0.35,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.labels = list(labels)
        self.threshold = threshold
        self.device = device
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from gliner import GLiNER

            self._model = GLiNER.from_pretrained(self.model_name)
            if self.device:
                self._model = self._model.to(self.device)
        return self._model

    def extract(self, text: str) -> list[RawEntity]:
        if not text.strip():
            return []
        try:
            model = self._load()
            raw = model.predict_entities(text, self.labels, threshold=self.threshold)
        except Exception:
            logger.exception("GLiNER predict_entities failed")
            raise
        return entities_from_gliner_raw(raw, text)


def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Half-open intervals [start, end) style overlap (touching at boundary = no overlap)."""
    return not (a_end <= b_start or b_end <= a_start)


def _overlap_resolution_sort_key(ent: RawEntity) -> tuple:
    """
    Order for greedy keep_highest_score:
    higher score first, then longer span (more coverage), then earlier start, shorter end (stable tie-break).
    """
    span_len = ent.end - ent.start
    return (-ent.score, -span_len, ent.start, ent.end)


def resolve_overlapping_entities(
    entities: list[RawEntity],
    strategy: str = "keep_highest_score",
) -> list[RawEntity]:
    """
    Post-process overlapping spans (optional, separate from extraction).

    Strategies:
      - keep_highest_score: process entities in sort order (see _overlap_resolution_sort_key);
        keep a span if it does not overlap any already kept span (greedy set packing).
      - none: return a shallow copy of the list, unchanged order.
    """
    if strategy == "none":
        return list(entities)
    if len(entities) <= 1:
        return list(entities)
    if strategy != "keep_highest_score":
        raise ValueError(f"Unknown overlap strategy: {strategy}")

    sorted_e = sorted(entities, key=_overlap_resolution_sort_key)
    kept: list[RawEntity] = []
    for ent in sorted_e:
        if any(spans_overlap(ent.start, ent.end, k.start, k.end) for k in kept):
            continue
        kept.append(ent)
    kept.sort(key=lambda x: (x.start, x.end))
    return kept
