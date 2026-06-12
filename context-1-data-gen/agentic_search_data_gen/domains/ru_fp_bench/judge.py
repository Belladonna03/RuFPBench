import argparse
from typing import Dict, List

from .schemas import JudgedRun
from .utils import classify_response_type, get_logger, read_jsonl, write_jsonl


ALLOWED_LABELS = {"answered", "refused", "partial_refusal", "bad_generation"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge target responses for over-refusal.")
    parser.add_argument("--input", required=True, help="Input target runs JSONL")
    parser.add_argument("--output", required=True, help="Output judged JSONL")
    args = parser.parse_args()

    logger = get_logger("ru_fp_bench.judge")
    logger.info("Starting judge input=%s output=%s", args.input, args.output)

    rows = read_jsonl(args.input)
    judged_rows: List[Dict] = []
    label_counts: Dict[str, int] = {k: 0 for k in sorted(ALLOWED_LABELS)}

    for row in rows:
        if row.get("__invalid_json__"):
            continue
        label = classify_response_type(row.get("safe_response", ""))
        if label not in ALLOWED_LABELS:
            label = "bad_generation"
        label_counts[label] = label_counts.get(label, 0) + 1

        judged = JudgedRun(
            id=row["id"],
            topic=row["topic"],
            safe_neighbor=row["safe_neighbor"],
            safe_request=row["safe_request"],
            safe_response=row["safe_response"],
            model=row.get("model", "unknown-model"),
            judge=label,
        )
        judged_rows.append(judged.to_dict())

    written = write_jsonl(args.output, judged_rows)
    logger.info("Judge complete written=%s labels=%s", written, label_counts)
    print(f"Judged rows: {written}")
    print(f"Label counts: {label_counts}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
