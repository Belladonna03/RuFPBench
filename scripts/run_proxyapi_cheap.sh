#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/proxyapi_cheap.yaml}"
ENV_FILE="${ENV_FILE:-proxy.env}"
RUN_DIR="${RUN_DIR:-runs/proxyapi_cheap_$(date +%Y%m%d_%H%M%S)}"
TARGET_RAW="${TARGET_RAW:-800}"
MIN_BORDERLINE="${MIN_BORDERLINE:-25}"
RAW_BATCH_SIZE="${RAW_BATCH_SIZE:-300}"
JOBS_OUTPUT_COUNT="${JOBS_OUTPUT_COUNT:-24}"
MAX_ROUNDS="${MAX_ROUNDS:-8}"
MAX_WORKERS="${MAX_WORKERS:-4}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE. Use proxy.env or copy .env.proxyapi.example to $ENV_FILE and set PROXYAPI_API_KEY." >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

if [[ -z "${PROXYAPI_API_KEY:-}" || "${PROXYAPI_API_KEY}" == "PUT_PROXYAPI_KEY_HERE" ]]; then
  echo "PROXYAPI_API_KEY is empty in $ENV_FILE" >&2
  exit 2
fi

if [[ "${SKIP_PROBE:-0}" != "1" ]]; then
  echo "Running ProxyAPI connectivity probes. Set SKIP_PROBE=1 to skip."
  CONFIG="$CONFIG" ENV_FILE="$ENV_FILE" PROBE_OPENROUTER="${PROBE_OPENROUTER:-1}" scripts/proxyapi_probe.sh
fi

python -m rufpbench run \
  --config "$CONFIG" \
  --env "$ENV_FILE" \
  --run-dir "$RUN_DIR" \
  --mode cascade_mining \
  --target-raw "$TARGET_RAW" \
  --min-borderline "$MIN_BORDERLINE" \
  --raw-batch-size "$RAW_BATCH_SIZE" \
  --jobs-output-count "$JOBS_OUTPUT_COUNT" \
  --max-rounds "$MAX_ROUNDS" \
  --max-workers "$MAX_WORKERS"

echo
printf 'Run finished. Inspect:\n  %s/final/rufpbench_borderline.jsonl\n  %s/reports/report.md\n  %s/state.json\n' "$RUN_DIR" "$RUN_DIR" "$RUN_DIR"
