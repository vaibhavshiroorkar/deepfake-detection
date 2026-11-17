#!/usr/bin/env bash
# Rerun the scoring, evaluation and H4 step of runs/p001-2day.sh for one target (same commands).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
# Usage: p001-rescore.sh NAME [MANIFEST] [FEATURES]; defaults follow the 2-day chain.
t=$1
MANIFEST=${2:-data/manifests/$t-neutral.csv}
FEATURES=${3:-runs/avhubert-$t-neutral}
$PY scripts/score_sync_targets.py --checkpoints runs/p001/sync --features "$FEATURES" \
    --manifest "$MANIFEST" --name "$t" --output runs/p001/sync-targets || exit 1
for s in 17 29 43; do
    $PY scripts/research_artifacts.py --receipt "runs/p001/sync-targets/receipts/evaluate-$t-seed$s.json" \
        --outputs "runs/p001/sync-targets/sync-$t-seed$s-metrics.json" \
        --inputs "runs/p001/sync-targets/sync-$t-seed$s-predictions.csv" "runs/p001/sync/sync-seed$s-threshold.json" -- \
        $PY -m deepfake_detection.cli evaluate predictions --predictions "runs/p001/sync-targets/sync-$t-seed$s-predictions.csv" \
        --threshold-file "runs/p001/sync/sync-seed$s-threshold.json" \
        --output "runs/p001/sync-targets/sync-$t-seed$s-metrics.json" --bootstrap-samples 1000 --seed 17 > /dev/null
    $PY scripts/run_h4.py --predictions "runs/p001/sync-targets/sync-$t-seed$s-predictions.csv" \
        --source-threshold "runs/p001/sync/sync-seed$s-threshold.json" \
        --output "runs/p001/h4/sync-$t-seed$s.json" || echo "H4 not run for $t seed $s"
done
