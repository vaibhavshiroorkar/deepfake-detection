#!/usr/bin/env bash
# P-001 resume after the memory stop on 2026-11-14: one stage at a time, lighter settings.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
stamp() { echo "[$(date '+%a %H:%M:%S')] $*"; }
FILTER='^W0000|^I0000|Feedback manager|xnnpack|XNNPACK'
stamp "AV-HuBERT features, 4 workers"
$PY scripts/extract_avhubert_features.py --manifest data/manifests/fakeavceleb-neutral.csv \
    --cuts data/manifests/fakeavceleb-neutral.cuts.csv --windows 2 --step 1 \
    --output runs/avhubert-fakeavceleb-neutral --workers 4 2>&1 | grep -Ev "$FILTER" || true
stamp "sync path"
$PY scripts/train_sync_path.py --features runs/avhubert-fakeavceleb-neutral --split-dir runs/p001/split \
    --output runs/p001/sync
stamp "visual and audio paths, 2 cache shards"
SHARDS=2 bash scripts/run_p001_branches.sh
stamp "targets"
for spec in "runs/streams-20260905/dfdc-manifest.csv:DFDC:dfdc" \
            "data/manifests/lavdf-test-sample.csv:LAV-DF:lavdf-test" \
            "data/manifests/lavdf-dev-sample.csv:LAV-DF:lavdf-dev" \
            "data/manifests/lavdf-train-sample.csv:LAV-DF:lavdf-train"; do
    IFS=: read -r manifest dataset name <<< "$spec"
    $PY scripts/neutralize_dataset.py --manifest "$manifest" --dataset "$dataset" --output-root data/neutral \
        --manifest-out "data/manifests/$name-neutral.csv" --workers 6 | grep -A20 rows_failed
done
stamp "resume done"
