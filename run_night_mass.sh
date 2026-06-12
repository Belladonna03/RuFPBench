#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/night_mass.yaml}"
ENV_FILE="${ENV_FILE:-night.env}"
RUN_DIR="${RUN_DIR:-runs/night_mass_$(date +%Y%m%d_%H%M%S)}"
mkdir -p logs

python -m rufpbench run \
  --config "$CONFIG" \
  --env "$ENV_FILE" \
  --run-dir "$RUN_DIR" \
  --mode cascade_mining \
  --target-raw "${TARGET_RAW:-10000}" \
  --min-borderline "${MIN_BORDERLINE:-300}" \
  --raw-batch-size "${RAW_BATCH_SIZE:-600}" \
  --jobs-output-count "${JOBS_OUTPUT_COUNT:-16}" \
  --max-rounds "${MAX_ROUNDS:-16}" \
  --max-workers "${MAX_WORKERS:-25}" \
  2>&1 | tee "logs/$(basename "$RUN_DIR").log"

echo "Done: $RUN_DIR"
echo "Final: $RUN_DIR/final/rufpbench_borderline.jsonl"
echo "Report: $RUN_DIR/reports/report.md"
