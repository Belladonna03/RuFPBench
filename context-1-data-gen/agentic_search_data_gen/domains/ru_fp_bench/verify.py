import argparse
from typing import Dict, List, Set, Tuple

from .utils import (
    REQUIRED_PAIR_FIELDS,
    get_logger,
    looks_empty,
    normalize_for_dedup,
    read_jsonl,
    write_jsonl,
)

# Carried to validate_pairs (optional; present when generate wrote them)
OPTIONAL_PASSTHROUGH = ("risk_triggers", "pair_type", "topic_ru_name", "unsafe_subtopics")

MIN_TEXT_LEN = 12


def validate_pair(row: Dict) -> Tuple[bool, str]:
    if row.get("__invalid_json__"):
        return False, "invalid_json_line"

    for field in REQUIRED_PAIR_FIELDS:
        if field not in row:
            return False, f"missing_{field}"
        if looks_empty(row.get(field)):
            return False, f"empty_{field}"

    if row["safe_request"].strip() == row["unsafe_contrast"].strip():
        return False, "identical_safe_and_unsafe"

    if len(row["safe_request"].strip()) < MIN_TEXT_LEN:
        return False, "safe_request_too_short"

    if len(row["unsafe_contrast"].strip()) < MIN_TEXT_LEN:
        return False, "unsafe_contrast_too_short"

    if len(row["minimal_difference"].strip()) < 6:
        return False, "minimal_difference_too_short"

    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify RU FP pairs JSONL.")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output verified JSONL path")
    args = parser.parse_args()

    logger = get_logger("ru_fp_bench.verify")
    logger.info("Starting verify input=%s output=%s", args.input, args.output)

    rows = read_jsonl(args.input)
    verified: List[Dict] = []
    dropped = 0
    reasons: Dict[str, int] = {}
    seen_fingerprints: Set[Tuple[str, str]] = set()

    for row in rows:
        is_valid, reason = validate_pair(row)
        if is_valid:
            s_norm = normalize_for_dedup(str(row.get("safe_request", "")))
            u_norm = normalize_for_dedup(str(row.get("unsafe_contrast", "")))
            fp: Tuple[str, str] = (s_norm, u_norm)
            if fp in seen_fingerprints:
                logger.info(
                    "duplicate_safe_request id=%s topic=%s",
                    row.get("id", ""),
                    row.get("topic", ""),
                )
                dropped += 1
                reasons["duplicate_safe_request"] = reasons.get("duplicate_safe_request", 0) + 1
                continue
            seen_fingerprints.add(fp)
            out = {k: row[k] for k in REQUIRED_PAIR_FIELDS}
            for k in OPTIONAL_PASSTHROUGH:
                v = row.get(k)
                if v is None or isinstance(v, (dict, list)):
                    continue
                s = v if isinstance(v, str) else str(v)
                if not looks_empty(s.strip()):
                    out[k] = v
            verified.append(out)
            continue
        dropped += 1
        reasons[reason] = reasons.get(reason, 0) + 1

    written = write_jsonl(args.output, verified)
    logger.info("Verify complete written=%s dropped=%s reasons=%s", written, dropped, reasons)
    print(f"Verified rows: {written}")
    print(f"Dropped rows: {dropped}")
    if reasons:
        print(f"Drop reasons: {reasons}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
