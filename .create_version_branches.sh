#!/usr/bin/env bash
set -euo pipefail

REPO="/Users/dekovaleva/PythonProjects/ru_fp_bench_1"
cd "$REPO"

MINIMAL_GITIGNORE='.venv/
__pycache__/
*.pyc
.pytest_cache/
.DS_Store
.env
.env.proxyapi
'

RSYNC_EXCLUDES=(
  --exclude='.git/'
  --exclude='.venv/'
  --exclude='__pycache__/'
  --exclude='.pytest_cache/'
  --exclude='.DS_Store'
  --exclude='.env'
  --exclude='.env.proxyapi'
)

create_branch() {
  local version="$1"
  local source="$2"
  local message="$3"
  local branch="ru_fp_bench_v${version}"
  local extra_excludes=("${@:4}")

  echo "=== Creating ${branch} from ${source} ==="
  git checkout master
  git checkout -B "$branch"
  git rm -rf . >/dev/null 2>&1 || true

  rsync -a "${RSYNC_EXCLUDES[@]}" "${extra_excludes[@]}" "${source}/" .

  if [[ "$version" =~ ^(2|3|4)$ ]]; then
    printf '%s' "$MINIMAL_GITIGNORE" > .gitignore
    if [[ "$version" == "2" ]]; then
      echo 'data/ru_paradetox/' >> .gitignore
    fi
  fi

  git add -A
  if git status --porcelain | grep -E '\.env$|\.venv/'; then
    echo "ERROR: secrets or venv staged on ${branch}" >&2
    git status
    exit 1
  fi
  git commit -m "$message"
  echo "Done: ${branch} ($(git rev-parse --short HEAD))"
}

create_branch 2 "/Users/dekovaleva/PythonProjects/ru_fp_bench_2" \
  "Add RuFPBench v2: pseudo-graph extraction (GLiNER + KeyBERT)" \
  --exclude='data/ru_paradetox/'

create_branch 3 "/Users/dekovaleva/PythonProjects/ru_fp_bench_3" \
  "Add RuFPBench v3: Stage 1 generation pipeline"

create_branch 4 "/Users/dekovaleva/PythonProjects/ru_fp_bench_4" \
  "Add RuFPBench v4: agentic search data gen and unsafe topics taxonomy"

create_branch 5 "/Users/dekovaleva/PythonProjects/rufpbench_v5" \
  "Add RuFPBench v5: cascade miner (early version)"

create_branch 6 "/Users/dekovaleva/PythonProjects/rufpbench_v6" \
  "Add RuFPBench v6: cascade miner"

create_branch 7 "/Users/dekovaleva/PythonProjects/rufpbench_v7" \
  "Add RuFPBench v7: cascade miner with quality hardening"

create_branch 8 "/Users/dekovaleva/PythonProjects/rufpbench_v8" \
  "Add RuFPBench v8: cascade miner with scenario-first"

create_branch 9 "/Users/dekovaleva/PythonProjects/rufpbench_v9" \
  "Add RuFPBench v9: cascade miner with pipeline hardening"

create_branch 10 "/Users/dekovaleva/PythonProjects/rufpbench_v10" \
  "Add RuFPBench v10: cascade miner (false-reject product final)"

git checkout master
echo "All branches created."
