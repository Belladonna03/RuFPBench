import argparse
from typing import Dict, List

from .utils import get_logger, read_jsonl, write_jsonl


FALSE_POSITIVE_LABELS = {"refused", "partial_refusal"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export false-positive rows from judged outputs.")
    parser.add_argument("--input", required=True, help="Input judged JSONL")
    parser.add_argument("--output", required=True, help="Output false-positive benchmark JSONL")
    args = parser.parse_args()

    logger = get_logger("ru_fp_bench.export")
    logger.info("Starting export input=%s output=%s", args.input, args.output)

    rows = read_jsonl(args.input)
    exported: List[Dict] = []
    for row in rows:
        if row.get("__invalid_json__"):
            continue
        if row.get("judge") not in FALSE_POSITIVE_LABELS:
            continue
        exported.append(row)

    written = write_jsonl(args.output, exported)
    logger.info("Export complete exported=%s", written)
    print(f"Exported false positives: {written}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
