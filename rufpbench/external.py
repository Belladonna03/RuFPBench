from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import AppConfig
from .llm import LLMRouter
from .prompts import TRANSLATE_TO_RU_SYSTEM
from .utils import append_jsonl, ensure_dir, stable_id


def import_hf_dataset(
    *,
    dataset_name: str,
    split: str,
    output_path: Path,
    text_column: str | None = None,
    label_column: str | None = None,
    limit: int = 500,
) -> int:
    """Import external HF prompts as raw seeds. Requires `pip install -e .[hf]`."""
    try:
        from datasets import load_dataset
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Install optional dependency: pip install -e .[hf]") from e

    ds = load_dataset(dataset_name, split=split)
    rows = []
    for i, item in enumerate(ds):
        if i >= limit:
            break
        if text_column is None:
            # Try common columns.
            for cand in ["prompt", "question", "instruction", "text", "safe_prompt"]:
                if cand in item and item[cand]:
                    text_column = cand
                    break
        if text_column is None or text_column not in item:
            continue
        text = str(item[text_column]).strip()
        if not text:
            continue
        rows.append(
            {
                "external_seed_id": stable_id("hf", dataset_name, split, i, text),
                "dataset": dataset_name,
                "split": split,
                "text": text,
                "label": str(item.get(label_column, "")) if label_column else "",
                "raw": dict(item),
            }
        )
    append_jsonl(output_path, rows)
    return len(rows)


def translate_external_seeds_to_ru(
    *,
    cfg: AppConfig,
    client: LLMRouter,
    input_path: Path,
    output_path: Path,
    limit: int = 500,
) -> int:
    ensure_dir(output_path.parent)
    count = 0
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if count >= limit:
                break
            row = json.loads(line)
            text = row.get("text", "")
            if not text:
                continue
            data = client.json_call(
                step_name="translate_to_ru",
                system=TRANSLATE_TO_RU_SYSTEM,
                user=json.dumps({"source_prompt": text, "source_dataset": row.get("dataset")}, ensure_ascii=False),
                expected="object",
            )
            out = {
                **row,
                "ru_prompt": data.get("ru_prompt", ""),
                "translation_notes": data.get("notes", ""),
                "translator_model": client.model_for_step("translate_to_ru", default=cfg.models.translator_model),
            }
            append_jsonl(output_path, [out])
            count += 1
    return count
