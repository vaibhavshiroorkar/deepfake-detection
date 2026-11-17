#!/usr/bin/env bash
# P-001 H2 (amendment A8), relaunched 2026-11-16 14:30 for speed. Full-HD targets cached at
# about 290 clips an hour with two shards, which put H2 at Wednesday. Three shards per corpus
# (six processes, about 3.5 GB private each) ran the PC down to 2.4 GB free RAM, so two per corpus. Now:
#   1. DFDC and LAV-DF cache in parallel, two shards each (resume skips cached clips);
#   2. HiFi-AVDF caches in the background, two shards;
#   3. H2 runs on its Holm family (DFDC, LAV-DF, FakeAVCeleb strict) as soon as those are scored;
#   4. H2 runs again once HiFi-AVDF, a reported-only target, is scored. Same family inputs, same family result.
# Resumed 2026-11-17 after the second hang (mistake 71): both hangs came while A11's CLIP training
# ran beside target caching. Finished stages are skipped, and HiFi-AVDF now caches in the foreground at
# four shards (the load that ran stable for seven hours on 11-16) before anything else uses the GPU.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
DDF=(".venv/Scripts/python.exe" -m deepfake_detection.cli)
RUN=runs/p001
SHARDS=2
stamp() { echo "[$(date '+%a %H:%M:%S')] $*"; }
receipt() { local name=$1; shift; $PY scripts/research_artifacts.py --receipt "$RUN/receipts/$name.json" "$@"; }
declare -A DS=([dfdc]=DFDC [lavdf-test]=LAV-DF [hifiavdf]=HiFi-AVDF)

until grep -q "two-day chain done" runs/p001-2day.log && [ -f "$RUN/cache-audit.json" ]; do sleep 120; done
PREP=$($PY -c "import json;print(json.load(open('$RUN/cache-audit.json'))['preprocessing_hash'])")
stamp "start (v2); reference preprocessing hash $PREP"

STARTED=()
cache_shards() {  # target: start SHARDS cache processes in this shell; their pids go to STARTED
    local t=$1
    for ((i = 0; i < SHARDS; i++)); do
        "${DDF[@]}" cache build --manifest "data/manifests/$t-neutral.csv" --dataset-root data \
            --cache-root "$RUN/cache-targets" --index "$RUN/cache-$t-index-$i-of-$SHARDS.csv" \
            --audit "$RUN/cache-$t-audit-$i-of-$SHARDS.json" --dataset "${DS[$t]}" --device cuda --code-version p001 \
            --skip-cached --shard "$i/$SHARDS" > "$RUN/logs/cache-$t-$i-of-$SHARDS.log" 2>&1 &
        STARTED+=($!)
    done
}
merge_and_usable() {  # target
    local t=$1 idx=() aud=()
    for ((i = 0; i < SHARDS; i++)); do idx+=("$RUN/cache-$t-index-$i-of-$SHARDS.csv"); aud+=("$RUN/cache-$t-audit-$i-of-$SHARDS.json"); done
    "${DDF[@]}" cache merge --indexes "${idx[@]}" --audits "${aud[@]}" --index "$RUN/cache-$t-index.csv" --audit "$RUN/cache-$t-audit.json"
    local got
    got=$($PY -c "import json;print(json.load(open('$RUN/cache-$t-audit.json'))['preprocessing_hash'])")
    [ "$got" = "$PREP" ] || { stamp "preprocessing hash differs for $t: $got"; exit 1; }
    for branch in visual audio; do
        "${DDF[@]}" manifest usable --manifest "data/manifests/$t-neutral.csv" --cache-index "$RUN/cache-$t-index.csv" \
            --cache-root "$RUN/cache-targets" --output "$RUN/usable/$t-$branch.csv" --audit "$RUN/usable/$t-$branch.json" \
            --dataset "${DS[$t]}" --branch "$branch" --preprocessing-hash "$PREP"
    done
    stamp "cached and usable: $t"
}
score_target() {  # target: every visual and audio model on it, waiting for each model to exist
    local t=$1
    for branch in visual audio; do
        for seed in 17 29 43; do
            local name="$branch-seed$seed"
            until [ -f "$RUN/evaluation/$name-test-identity-strict-predictions.csv" ]; do sleep 300; done
            [ -f "$RUN/evaluation/$name-$t-predictions.csv" ] && continue
            stamp "score $name on $t"
            receipt "evaluate-$name-$t" --outputs "$RUN/evaluation/$name-$t-metrics.json" "$RUN/evaluation/$name-$t-predictions.csv" \
                --inputs "$RUN/checkpoints/$name.pt" "$RUN/usable/$t-$branch.csv" "$RUN/evaluation/$name-threshold.json" -- \
            "${DDF[@]}" evaluate branch --branch "$branch" --checkpoint "$RUN/checkpoints/$name.pt" \
                --manifest "$RUN/usable/$t-$branch.csv" --cache-index "$RUN/cache-$t-index.csv" --cache-root "$RUN/cache-targets" \
                --dataset "${DS[$t]}" --threshold-file "$RUN/evaluation/$name-threshold.json" \
                --evidence-scope generalization_cross_corpus --preprocessing-hash "$PREP" \
                --output "$RUN/evaluation/$name-$t-metrics.json" --predictions "$RUN/evaluation/$name-$t-predictions.csv"
        done
    done
}

if grep -q "h2 family done" runs/p001-h2.log; then
    stamp "resume: dfdc and lavdf-test cached and scored, h2 family done"
else
    stamp "cache dfdc and lavdf-test, $SHARDS shards each"
    STARTED=(); cache_shards dfdc; cache_shards lavdf-test
    for pid in "${STARTED[@]}"; do wait "$pid"; done
    merge_and_usable dfdc
    merge_and_usable lavdf-test
    score_target dfdc
    score_target lavdf-test
    stamp "fusion on the Holm family"
    receipt "h2-family" --outputs "$RUN/h2/h2.json" -- $PY scripts/run_h2.py --sync-targets "$RUN/sync-targets" --output "$RUN/h2"
    stamp "h2 family done"
fi

SHARDS=4
stamp "cache hifiavdf alone, $SHARDS shards"
STARTED=(); cache_shards hifiavdf
for pid in "${STARTED[@]}"; do wait "$pid"; done
merge_and_usable hifiavdf
score_target hifiavdf
stamp "fusion with hifiavdf"
receipt "h2-full" --outputs "$RUN/h2-full/h2.json" -- $PY scripts/run_h2.py --sync-targets "$RUN/sync-targets" --output "$RUN/h2-full"
stamp "h2 chain done"
