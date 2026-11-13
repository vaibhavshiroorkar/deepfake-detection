#!/usr/bin/env bash
# P-001: visual and audio paths on neutral FakeAVCeleb, three seeds each.
#
# Steps: thin the training partition (amendment A1), cache views for the clips
# we use, keep the clips each branch can read, train, select a threshold on
# validation, and score validation and test with that threshold file.
# Every output lands in runs/p001/. Copy metrics and predictions to results/
# only after the run, with scripts that read them from here.
#
#   bash scripts/run_p001_branches.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
DDF=(".venv/Scripts/python.exe" -m deepfake_detection.cli)
RUN=runs/p001
SHARDS=${SHARDS:-3}
SEEDS=(17 29 43)
mkdir -p "$RUN/logs" "$RUN/checkpoints" "$RUN/evaluation" "$RUN/usable" "$RUN/receipts"

stage() { echo "[$(date '+%H:%M:%S')] $*"; }
# Each stage leaves a receipt: the command, a hash of every input, of every
# Python file in src/ and scripts/, and of every output (research_artifacts.py).
# A rerun with the same code and inputs verifies the old outputs instead.
receipt() { local name=$1; shift; $PY scripts/research_artifacts.py --receipt "$RUN/receipts/$name.json" "$@"; }

test -f "$RUN/split/train.csv" || { echo "missing $RUN/split; run the sync chain first"; exit 1; }

stage "subsample training partition"
"${DDF[@]}" split subsample --split-dir "$RUN/split" --output-dir "$RUN/split-train3" \
    --dataset FakeAVCeleb --seed 17 --fake-ratio 3

# One cache for every clip we read: thinned train, full val, full test.
MANIFEST="$RUN/cache-manifest.csv"
{ head -1 "$RUN/split-train3/train.csv"
  for part in "$RUN/split-train3/train.csv" "$RUN/split/val.csv" "$RUN/split/test.csv"; do tail -n +2 "$part"; done
} > "$MANIFEST"

stage "cache build, $SHARDS shards"
pids=()
for ((i = 0; i < SHARDS; i++)); do
    "${DDF[@]}" cache build --manifest "$MANIFEST" --dataset-root data --cache-root "$RUN/cache" \
        --index "$RUN/cache-index-$i.csv" --audit "$RUN/cache-audit-$i.json" --dataset FakeAVCeleb \
        --device cuda --code-version p001 --skip-cached --shard "$i/$SHARDS" \
        > "$RUN/logs/cache-$i.log" 2>&1 &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
"${DDF[@]}" cache merge --indexes $(for ((i = 0; i < SHARDS; i++)); do echo "$RUN/cache-index-$i.csv"; done) \
    --audits $(for ((i = 0; i < SHARDS; i++)); do echo "$RUN/cache-audit-$i.json"; done) \
    --index "$RUN/cache-index.csv" --audit "$RUN/cache-audit.json"
PREP=$($PY -c "import json;print(json.load(open('$RUN/cache-audit.json'))['preprocessing_hash'])")
SPLIT_HASH=$($PY -c "import json;print(json.load(open('$RUN/split/audit.json'))['split_hash'])")

usable() {  # source target branch
    "${DDF[@]}" manifest usable --manifest "$1" --cache-index "$RUN/cache-index.csv" --cache-root "$RUN/cache" \
        --output "$2" --audit "${2%.csv}.json" --dataset FakeAVCeleb --branch "$3" --preprocessing-hash "$PREP"
}
for branch in visual audio; do
    usable "$RUN/split-train3/train.csv" "$RUN/usable/train-$branch.csv" "$branch"
    for part in val test test-identity-strict; do
        usable "$RUN/split/$part.csv" "$RUN/usable/$part-$branch.csv" "$branch"
    done
done

for branch in visual audio; do
    epochs=$([ "$branch" = visual ] && echo 8 || echo 6)
    for seed in "${SEEDS[@]}"; do
        name="$branch-seed$seed"
        stage "train $name"
        receipt "train-$name" --outputs "$RUN/checkpoints/$name.pt" "$RUN/checkpoints/$name-history.json" \
            --inputs "$RUN/usable/train-$branch.csv" "$RUN/usable/val-$branch.csv" "$RUN/cache-index.csv" -- \
            "${DDF[@]}" train "$branch" --train-manifest "$RUN/usable/train-$branch.csv" \
                --validation-manifest "$RUN/usable/val-$branch.csv" \
                --cache-index "$RUN/cache-index.csv" --cache-root "$RUN/cache" --dataset FakeAVCeleb \
                --checkpoint "$RUN/checkpoints/$name.pt" --history "$RUN/checkpoints/$name-history.json" \
                --run-id "p001-$name" --split-hash "$SPLIT_HASH" --preprocessing-hash "$PREP" \
                --device cuda --epochs "$epochs" --learning-rate 0.0001 --weight-decay 0.0001 \
                --patience 2 --workers 0 --seed "$seed" --batch-size 8 --accumulation-steps 2 \
                --freeze-epochs 1 > "$RUN/logs/train-$name.log" 2>&1
        stage "evaluate $name"
        "${DDF[@]}" evaluate branch --branch "$branch" --checkpoint "$RUN/checkpoints/$name.pt" \
            --manifest "$RUN/usable/val-$branch.csv" --cache-index "$RUN/cache-index.csv" --cache-root "$RUN/cache" \
            --dataset FakeAVCeleb --threshold 0.5 --evidence-scope development_validation \
            --output "$RUN/evaluation/$name-val-metrics.json" --predictions "$RUN/evaluation/$name-val-predictions.csv" \
            --preprocessing-hash "$PREP"
        "${DDF[@]}" threshold --predictions "$RUN/evaluation/$name-val-predictions.csv" --split val \
            --output "$RUN/evaluation/$name-threshold.json"
        for part in test test-identity-strict; do
            receipt "evaluate-$name-$part" --outputs "$RUN/evaluation/$name-$part-metrics.json" \
                "$RUN/evaluation/$name-$part-predictions.csv" \
                --inputs "$RUN/checkpoints/$name.pt" "$RUN/usable/$part-$branch.csv" \
                "$RUN/evaluation/$name-threshold.json" -- \
            "${DDF[@]}" evaluate branch --branch "$branch" --checkpoint "$RUN/checkpoints/$name.pt" \
                --manifest "$RUN/usable/$part-$branch.csv" --cache-index "$RUN/cache-index.csv" --cache-root "$RUN/cache" \
                --dataset FakeAVCeleb --threshold-file "$RUN/evaluation/$name-threshold.json" \
                --evidence-scope development_test \
                --output "$RUN/evaluation/$name-$part-metrics.json" \
                --predictions "$RUN/evaluation/$name-$part-predictions.csv" --preprocessing-hash "$PREP"
        done
    done
done
stage "branches done"
