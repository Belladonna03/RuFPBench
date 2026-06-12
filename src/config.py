"""Default GLiNER labels, optional keyphrase blocklist, and pipeline settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

InputFormat = Literal["auto", "jsonl", "csv"]

DEFAULT_GLINER_LABELS: list[str] = [
    "person",
    "group",
    "role",
    "organization",
    "institution",
    "location",
    "event",
    "ideology",
    "weapon",
    "substance",
    "chemical",
    "drug",
    "disease",
    "body_part",
    "tool",
    "artifact",
    "platform",
    "service",
    "document",
    "law",
    "money",
    "time",
    "quantity",
    "procedure",
    "target",
]

# Optional: drop specific normalized phrases (residual junk). Main filtering is rule-based
# in ``keyphrase_filter_rules`` (function words + structure). Default: none.
DEFAULT_KEYPHRASE_EXTRA_BLOCKLIST: frozenset[str] = frozenset()


@dataclass
class PipelineConfig:
    """Runtime options for models, graph, and I/O."""

    gliner_model: str = "urchade/gliner_medium-v2.1"
    keybert_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    entity_threshold: float = 0.35
    entity_labels: list[str] = field(default_factory=lambda: list(DEFAULT_GLINER_LABELS))

    keyphrase_ngram_range: tuple[int, int] = (1, 3)
    top_n_keyphrases: int = 10
    use_mmr: bool = True
    diversity: float = 0.5
    keyphrase_min_chars: int = 2
    keyphrase_max_chars: int = 256
    # Drop single-token phrases shorter than this (after normalization), unless multi-token.
    keyphrase_single_token_min_chars: int = 3
    # Whole-phrase blocklist (normalized). Prefer rules in keyphrase_filter_rules; use this for
    # rare fixed strings (e.g. product-specific noise).
    keyphrase_stop_phrases: frozenset[str] = field(
        default_factory=lambda: frozenset(DEFAULT_KEYPHRASE_EXTRA_BLOCKLIST)
    )

    nearby_window_chars: int = 64
    soft_merge_by_normalized_text: bool = True

    text_column: str = "text"
    id_column: str = "id"
    input_format: InputFormat = "auto"

    device: Optional[str] = None
