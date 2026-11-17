#!/usr/bin/env bash
# P-001 two-day plan (2026-11-15): targets -> features -> frozen sync scores -> H4; neutral audits.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
stamp() { echo "[$(date '+%a %H:%M:%S')] $*"; }
FILTER='^W0000|^I0000|Feedback manager|xnnpack|XNNPACK'
until [ -f results/shortcut-audit/hifiavdf/summary.json ]; do sleep 60; done
stamp "raw HiFi-AVDF audit done"
declare -A SRC=([hifiavdf]=data/manifests/hifiavdf.csv [dfdc]=runs/streams-20260905/dfdc-manifest.csv [lavdf-test]=data/manifests/lavdf-test-sample.csv)
declare -A DS=([hifiavdf]=HiFi-AVDF [dfdc]=DFDC [lavdf-test]=LAV-DF)
for t in hifiavdf dfdc lavdf-test; do
    stamp "neutralize $t"
    SIZE=(); [ "$t" = hifiavdf ] && SIZE=(--size 1280x720)   # P-001 A7
    $PY scripts/neutralize_dataset.py --manifest "${SRC[$t]}" --dataset "${DS[$t]}" --output-root data/neutral "${SIZE[@]}" \
        --manifest-out "data/manifests/$t-neutral.csv" --workers 6 | grep -A16 rows_failed
done
for t in hifiavdf lavdf-test dfdc; do
    stamp "features $t"
    $PY scripts/extract_avhubert_features.py --manifest "data/manifests/$t-neutral.csv" \
        --cuts "data/manifests/$t-neutral.cuts.csv" --windows 2 --step 1 \
        --output "runs/avhubert-$t-neutral" --workers 4 2>&1 | grep -Ev "$FILTER" | tail -2
    stamp "score $t"
    $PY scripts/score_sync_targets.py --checkpoints runs/p001/sync --features "runs/avhubert-$t-neutral" \
        --manifest "data/manifests/$t-neutral.csv" --name "$t" --output runs/p001/sync-targets
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
done
for t in hifiavdf dfdc lavdf-test; do
    stamp "neutral audit $t"
    $PY scripts/shortcut_audit.py --manifest "data/manifests/$t-neutral.csv" --dataset "$t-neutral" \
        --output "results/shortcut-audit/$t-neutral" --workers 4 | grep -v '^[0-9]*/[0-9]*$'
done
stamp "two-day chain done"
