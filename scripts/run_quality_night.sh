#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/night_quality_local.yaml}"
ENV_FILE="${ENV_FILE:-quality.env}"
RUN_NAME="${RUN_NAME:-quality_night_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-runs/$RUN_NAME}"
TARGET_RAW="${TARGET_RAW:-3000}"
MIN_BORDERLINE="${MIN_BORDERLINE:-120}"
MAX_ROUNDS="${MAX_ROUNDS:-24}"
MAX_WORKERS="${MAX_WORKERS:-6}"

mkdir -p logs

echo "RuFPBench quality run"
echo "config: $CONFIG"
echo "env: $ENV_FILE"
echo "run_dir: $RUN_DIR"
echo "target_raw=$TARGET_RAW min_borderline=$MIN_BORDERLINE max_rounds=$MAX_ROUNDS max_workers=$MAX_WORKERS"

python -m rufpbench run \
  --config "$CONFIG" \
  --env "$ENV_FILE" \
  --mode cascade_mining_quality \
  --run-dir "$RUN_DIR" \
  --target-raw "$TARGET_RAW" \
  --min-borderline "$MIN_BORDERLINE" \
  --max-rounds "$MAX_ROUNDS" \
  --max-workers "$MAX_WORKERS"
