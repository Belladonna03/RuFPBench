#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/proxyapi_cheap.yaml}"
ENV_FILE="${ENV_FILE:-proxy.env}"

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

MODELS=(
  "openai/gpt-5.4-nano"
  "openai/gpt-4o-mini"
)

if [[ "${PROBE_OPENROUTER:-1}" == "1" ]]; then
  MODELS+=("openrouter/openai/gpt-oss-20b")
fi

for model in "${MODELS[@]}"; do
  echo "==> Probing $model"
  python -m rufpbench probe --config "$CONFIG" --env "$ENV_FILE" --model "$model" --max-tokens 96
  echo
done
