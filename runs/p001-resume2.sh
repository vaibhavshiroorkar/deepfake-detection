#!/usr/bin/env bash
# P-001 resume 2 (2026-11-15): visual and audio paths, then neutral copies of the targets.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
stamp() { echo "[$(date '+%a %H:%M:%S')] $*"; }
stamp "visual and audio paths, 2 cache shards"
SHARDS=2 bash scripts/run_p001_branches.sh
stamp "targets"
# DFDC and the LAV-DF test sample belong to runs/p001-2day.sh; two writers corrupt outputs.
for spec in "data/manifests/lavdf-dev-sample.csv:LAV-DF:lavdf-dev" \
            "data/manifests/lavdf-train-sample.csv:LAV-DF:lavdf-train"; do
    IFS=: read -r manifest dataset name <<< "$spec"
    $PY scripts/neutralize_dataset.py --manifest "$manifest" --dataset "$dataset" --output-root data/neutral \
        --manifest-out "data/manifests/$name-neutral.csv" --workers 6 | grep -A20 rows_failed
done
stamp "resume2 done"
