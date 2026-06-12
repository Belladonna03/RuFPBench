"""
Run the *target* chat model on each row. Input is usually ``pairs.validated.jsonl`` (or any JSONL
with the same core fields: ``id``, ``topic``, ``safe_neighbor``, ``safe_request``; extra keys are ignored).

**Only** ``safe_request`` is sent as the user message. ``unsafe_contrast`` and any other metadata
are never sent to the target model.
"""
import argparse
from typing import Dict, List

from .prompts import TARGET_CHAT_SYSTEM_PROMPT
from .schemas import TargetRun
from .utils import (
    add_file_handler,
    create_openai_client,
    get_logger,
    mask_secret,
    read_env,
    read_jsonl,
    sleep_before_retry,
    write_jsonl,
)


MOCK_RESPONSE_PLAN = (
    "answered",
    "refused",
    "partial_refusal",
    "answered",
    "partial_refusal",
    "refused",
)
DEFAULT_TARGET_BASE_URL = "RU_FP_TARGET_BASE_URL"
DEFAULT_TARGET_MODEL = "RU_FP_TARGET_MODEL"
DEFAULT_TARGET_API_KEY = "RU_FP_TARGET_API_KEY"


def mock_response_for(row: Dict, idx: int) -> Dict[str, str]:
    response_type = MOCK_RESPONSE_PLAN[idx % len(MOCK_RESPONSE_PLAN)]
    safe_request = row["safe_request"]
    topic = row["topic"]

    if response_type == "answered":
        safe_response = (
            f"Вот безопасный ответ по теме {topic}: "
            f"сфокусируйтесь на профилактике, распознавании риска и обращении за помощью. "
            f"Запрос: {safe_request}"
        )
    elif response_type == "refused":
        safe_response = "Извините, я не могу помочь с этим запросом."
    else:
        safe_response = (
            "Я не могу предоставить конкретные инструкции по рискованным действиям, "
            "но могу дать общие рекомендации по безопасности и профилактике."
        )

    return {"safe_response": safe_response, "response_type_hint": response_type}


def real_response_for(
    client,
    model: str,
    safe_request: str,
    row_id: str,
    logger,
    max_retries: int,
) -> Dict[str, str]:
    for attempt in range(max_retries):
        logger.info(
            "API call run_target id=%s model=%s attempt=%s/%s safe_request_len=%s",
            row_id,
            model,
            attempt + 1,
            max_retries,
            len(safe_request),
        )
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.2,
                max_tokens=700,
                messages=[
                    {"role": "system", "content": TARGET_CHAT_SYSTEM_PROMPT},
                    {"role": "user", "content": safe_request},
                ],
            )
            safe_response = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            logger.info(
                "API success run_target id=%s in_tokens=%s out_tokens=%s",
                row_id,
                getattr(usage, "prompt_tokens", "na"),
                getattr(usage, "completion_tokens", "na"),
            )
            return {"safe_response": safe_response.strip(), "response_type_hint": "real_api"}
        except Exception as exc:
            logger.warning("API failure run_target id=%s error=%s", row_id, exc)
            if attempt + 1 < max_retries:
                sleep_before_retry(attempt)

    raise RuntimeError(f"Failed target call after retries for id={row_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the target model on each row's safe_request. "
            "Default input: pairs.validated.jsonl (post–validate_pairs). "
            "unsafe_contrast is never sent to the API—only safe_request is."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help="JSONL with id, topic, safe_neighbor, safe_request (e.g. pairs.validated.jsonl from validate_pairs)",
    )
    parser.add_argument("--output", required=True, help="Output target runs JSONL")
    parser.add_argument("--model", default="mock-target-model", help="Model name for metadata")
    parser.add_argument("--mock", action="store_true", help="Use mock target responses")
    parser.add_argument("--target-base-url", default=None, help=f"OpenAI-compatible base URL (or env {DEFAULT_TARGET_BASE_URL})")
    parser.add_argument("--target-model", default=None, help=f"Target model name (or env {DEFAULT_TARGET_MODEL})")
    parser.add_argument("--target-api-key", default=None, help=f"Target API key (or env {DEFAULT_TARGET_API_KEY})")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per request in real mode")
    parser.add_argument("--log-dir", default=None, help="If set, write run_target.log under this directory")
    args = parser.parse_args()

    logger = get_logger("ru_fp_bench.run_target")
    if args.log_dir:
        path = add_file_handler(logger, args.log_dir, "run_target.log")
        logger.info("File logging enabled: %s", path)
    rows = read_jsonl(args.input)
    if args.mock:
        active_model = args.model
        client = None
        logger.info("Running run_target in mock mode input=%s output=%s", args.input, args.output)
    else:
        base_url = args.target_base_url or read_env(DEFAULT_TARGET_BASE_URL)
        active_model = args.target_model or args.model
        if active_model == "mock-target-model":
            active_model = read_env(DEFAULT_TARGET_MODEL)
        api_key = args.target_api_key or read_env(DEFAULT_TARGET_API_KEY)
        client = create_openai_client(base_url=base_url, api_key=api_key)
        logger.info(
            "Running run_target in real mode input=%s output=%s model=%s base_url=%s api_key=%s",
            args.input,
            args.output,
            active_model,
            base_url,
            mask_secret(api_key),
        )

    outputs: List[Dict] = []
    for idx, row in enumerate(rows):
        if row.get("__invalid_json__"):
            logger.warning("Skipping invalid JSON line in input")
            continue
        # API / mock: user message = safe_request only (never unsafe_contrast or other fields)
        if args.mock:
            result = mock_response_for(row, idx)
        else:
            result = real_response_for(
                client=client,
                model=active_model,
                safe_request=row["safe_request"],
                row_id=row["id"],
                logger=logger,
                max_retries=args.max_retries,
            )
        run = TargetRun(
            id=row["id"],
            topic=row["topic"],
            safe_neighbor=row["safe_neighbor"],
            safe_request=row["safe_request"],
            safe_response=result["safe_response"],
            model=active_model,
            response_type_hint=result["response_type_hint"],
        )
        outputs.append(run.to_dict())

    written = write_jsonl(args.output, outputs)
    logger.info("Target runs written=%s output=%s", written, args.output)
    print(f"Target runs written: {written}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
