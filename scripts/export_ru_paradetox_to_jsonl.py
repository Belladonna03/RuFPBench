#!/usr/bin/env python3
"""
Export Hugging Face dataset s-nlp/ru_paradetox to JSONL for src.cli run.

Requires: pip install datasets

Each output line has:
  - id, text (chosen side), source, category
  - ru_toxic_comment, ru_neutral_comment (both kept for metadata.extra in the pipeline)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(
        description="Export s-nlp/ru_paradetox to JSONL for pseudo-graph pipeline."
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path (UTF-8, one object per line).",
    )
    p.add_argument(
        "--side",
        choices=("toxic", "neutral"),
        required=True,
        help="Which column becomes 'text': toxic → ru_toxic_comment, neutral → ru_neutral_comment.",
    )
    p.add_argument(
        "--split",
        default="train",
        help="Dataset split name (default: train). Dataset also has 'validation'.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Export only the first N rows (smoke test).",
    )
    args = p.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Missing package: pip install datasets", file=sys.stderr)
        return 1

    ds = load_dataset("s-nlp/ru_paradetox", split=args.split)
    n = len(ds)
    if args.limit is not None:
        n = min(n, max(0, args.limit))
        ds = ds.select(range(n))

    text_key = "ru_toxic_comment" if args.side == "toxic" else "ru_neutral_comment"
    category = "toxic" if args.side == "toxic" else "neutral"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as f:
        for i in range(len(ds)):
            row = ds[i]
            toxic = row.get("ru_toxic_comment") or ""
            neutral = row.get("ru_neutral_comment") or ""
            body = row.get(text_key) or ""
            obj = {
                "id": f"ru_paradetox_{args.split}_{args.side}_{i}",
                "text": str(body),
                "source": "s-nlp/ru_paradetox",
                "category": category,
                "split": args.split,
                "side": args.side,
                "ru_toxic_comment": str(toxic),
                "ru_neutral_comment": str(neutral),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} line(s) to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
