#!/usr/bin/env bash
# P-001 amendment A11 (exploratory): frozen CLIP ViT-L/14 visual stream and frozen XLS-R audio
# branch, three seeds, on the P-001 splits, caches and locked targets. Starts once the P-001
# branches are done (the GPU is free) and the H2 chain has built the target manifests.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
DDF=(".venv/Scripts/python.exe" -m deepfake_detection.cli)
RUN=runs/p001; OUT=$RUN/a11
stamp() { echo "[$(date '+%a %H:%M:%S')] $*"; }
mkdir -p "$OUT/checkpoints" "$OUT/evaluation" "$OUT/logs"
XLSR=facebook/wav2vec2-xls-r-300m
declare -A DS=([dfdc]=DFDC [lavdf-test]=LAV-DF [hifiavdf]=HiFi-AVDF)

# Never alongside the H2 caching: both hangs (16 and 17 November) came with CLIP training beside it.
# Start once the whole H2 chain, HiFi-AVDF included, is done.
until grep -q "h2 chain done" runs/p001-h2.log; do sleep 300; done
# Targets are scored as the H2 chain makes their manifests usable (HiFi-AVDF last).
wait_usable() { until [ -f "$RUN/usable/$1-$2.csv" ]; do sleep 300; done; }
PREP=$($PY -c "import json;print(json.load(open('$RUN/cache-audit.json'))['preprocessing_hash'])")
SPLIT_HASH=$($PY -c "import json;print(json.load(open('$RUN/split/audit.json'))['split_hash'])")
stamp "start"

evaluate() {  # name part predictions threshold
    "${DDF[@]}" evaluate predictions --predictions "$3" --threshold-file "$4" \
        --output "$OUT/evaluation/$1-$2-metrics.json" --bootstrap-samples 1000 --seed 17 > /dev/null
}

for seed in 17 29 43; do
    name="clip-seed$seed"
    # Rerun after the 11-17 XLS-R failure (handbook mistake 73): a model trained and scored on every target is kept.
    [ -f "$OUT/evaluation/$name-hifiavdf-metrics.json" ] && { stamp "keep $name, already scored"; continue; }
    stamp "train $name"
    "${DDF[@]}" train visual-stream --backbone clip --freeze-backbone --temporal gru --frame-chunk-size 16 \
        --train-manifest "$RUN/usable/train-visual.csv" --validation-manifest "$RUN/usable/val-visual.csv" \
        --cache-index "$RUN/cache-index.csv" --cache-root "$RUN/cache" --dataset FakeAVCeleb \
        --checkpoint "$OUT/checkpoints/$name.pt" --history "$OUT/checkpoints/$name-history.json" \
        --run-id "p001-a11-$name" --split-hash "$SPLIT_HASH" --preprocessing-hash "$PREP" --device cuda \
        --epochs 8 --batch-size 4 --accumulation-steps 4 --learning-rate 0.0001 --weight-decay 0.0001 \
        --patience 2 --workers 0 --seed "$seed" > "$OUT/logs/train-$name.log" 2>&1 || { stamp "train $name failed"; continue; }
    for part in val test test-identity-strict; do
        $PY scripts/score_visual_stream.py --checkpoint "$OUT/checkpoints/$name.pt" --manifest "$RUN/usable/$part-visual.csv" \
            --cache-index "$RUN/cache-index.csv" --cache-root "$RUN/cache" --dataset FakeAVCeleb --preprocessing-hash "$PREP" \
            --output "$OUT/evaluation/$name-$part-predictions.csv"
    done
    "${DDF[@]}" threshold --predictions "$OUT/evaluation/$name-val-predictions.csv" --split val \
        --output "$OUT/evaluation/$name-threshold.json"
    for part in test test-identity-strict; do evaluate "$name" "$part" "$OUT/evaluation/$name-$part-predictions.csv" "$OUT/evaluation/$name-threshold.json"; done
    for t in dfdc lavdf-test hifiavdf; do
        wait_usable "$t" visual
        $PY scripts/score_visual_stream.py --checkpoint "$OUT/checkpoints/$name.pt" --manifest "$RUN/usable/$t-visual.csv" \
            --cache-index "$RUN/cache-$t-index.csv" --cache-root "$RUN/cache-targets" --dataset "${DS[$t]}" --preprocessing-hash "$PREP" \
            --output "$OUT/evaluation/$name-$t-predictions.csv" && \
        evaluate "$name" "$t" "$OUT/evaluation/$name-$t-predictions.csv" "$OUT/evaluation/$name-threshold.json"
    done
done

for seed in 17 29 43; do
    name="xlsr-seed$seed"
    # Rerun after the 11-17 XLS-R failure (handbook mistake 73): a model trained and scored on every target is kept.
    [ -f "$OUT/evaluation/$name-hifiavdf-metrics.json" ] && { stamp "keep $name, already scored"; continue; }
    stamp "train $name"
    "${DDF[@]}" train audio --audio-model "$XLSR" --freeze-epochs 6 \
        --train-manifest "$RUN/usable/train-audio.csv" --validation-manifest "$RUN/usable/val-audio.csv" \
        --cache-index "$RUN/cache-index.csv" --cache-root "$RUN/cache" --dataset FakeAVCeleb \
        --checkpoint "$OUT/checkpoints/$name.pt" --history "$OUT/checkpoints/$name-history.json" \
        --run-id "p001-a11-$name" --split-hash "$SPLIT_HASH" --preprocessing-hash "$PREP" --device cuda \
        --epochs 6 --batch-size 8 --accumulation-steps 2 --learning-rate 0.0001 --weight-decay 0.0001 \
        --patience 2 --workers 0 --seed "$seed" > "$OUT/logs/train-$name.log" 2>&1 || { stamp "train $name failed"; continue; }
    score() {  # part manifest index root dataset
        "${DDF[@]}" evaluate branch --branch audio --audio-model "$XLSR" --checkpoint "$OUT/checkpoints/$name.pt" \
            --manifest "$2" --cache-index "$3" --cache-root "$4" --dataset "$5" --threshold 0.5 \
            --evidence-scope development_validation --preprocessing-hash "$PREP" \
            --output "$OUT/evaluation/$name-$1-branch-report.json" --predictions "$OUT/evaluation/$name-$1-predictions.csv" > /dev/null
    }
    for part in val test test-identity-strict; do score "$part" "$RUN/usable/$part-audio.csv" "$RUN/cache-index.csv" "$RUN/cache" FakeAVCeleb; done
    "${DDF[@]}" threshold --predictions "$OUT/evaluation/$name-val-predictions.csv" --split val \
        --output "$OUT/evaluation/$name-threshold.json"
    for part in test test-identity-strict; do evaluate "$name" "$part" "$OUT/evaluation/$name-$part-predictions.csv" "$OUT/evaluation/$name-threshold.json"; done
    for t in dfdc lavdf-test hifiavdf; do
        wait_usable "$t" audio
        score "$t" "$RUN/usable/$t-audio.csv" "$RUN/cache-$t-index.csv" "$RUN/cache-targets" "${DS[$t]}" && \
        evaluate "$name" "$t" "$OUT/evaluation/$name-$t-predictions.csv" "$OUT/evaluation/$name-threshold.json"
    done
done
stamp "a11 done v2"
