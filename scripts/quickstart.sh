#!/usr/bin/env bash
set -euo pipefail
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp -n .env.example .env || true
cp -n .env.proxyapi.example proxy.env || true
cat <<'MSG'
Edit .env, then run:
  rufpbench probe --model oss
  rufpbench probe-json --model qwen3.6-35b-a3b --output-count 3
  rufpbench run --mode cascade_mining --run-dir runs/rufpbench_cascade_001 --target-raw 8000 --min-borderline 500 --max-workers 16

ProxyAPI cheap main-pipeline run:
  edit proxy.env, then run scripts/proxyapi_probe.sh
  scripts/run_proxyapi_cheap.sh

No-key check:
  rufpbench run --mock --run-dir runs/smoke_mock --target-raw 40 --min-borderline 5 --max-rounds 4
MSG
