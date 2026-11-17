#!/usr/bin/env bash
# Website milestone 1 spike: two neutral FakeAVCeleb test clips through every P-001 check on CPU,
# with 2 threads to approximate a 2-vCPU Hugging Face Space. Times each stage.
# CUDA is hidden with -1: on Windows an empty CUDA_VISIBLE_DEVICES is dropped and the GPU stays
# visible. scripts/extract_avhubert_features.py moves AV-HuBERT to cuda unconditionally, so with
# the GPU hidden that stage fails; its CPU cost was measured separately (project log, 11-17).
set -uo pipefail
cd "$(dirname "$0")/.."
S=${1:?usage: web_spike_timing.sh ABSOLUTE_WORKDIR (holding neutral.csv and neutral.cuts.csv from scripts/neutralize_dataset.py)}
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=.venv/Scripts/python.exe
DDF=("$PY" -m deepfake_detection.cli)
PREP=$($PY -c "import json;print(json.load(open('runs/p001/cache-audit.json'))['preprocessing_hash'])")
timed() { local name=$1; shift; local t0; t0=$(date +%s%N); "$@" > "$S/$name.log" 2>&1; local rc=$?
          echo "$name: $(( ($(date +%s%N) - t0) / 1000000 )) ms, exit $rc"; }

timed features "$PY" scripts/extract_avhubert_features.py --manifest "$S/neutral.csv" --data-dir "$S" \
    --cuts "$S/neutral.cuts.csv" --windows 2 --step 1 --output "$S/avhubert" --workers 1
timed sync "$PY" scripts/score_sync_targets.py --checkpoints runs/p001/sync --features "$S/avhubert" \
    --manifest "$S/neutral.csv" --name spike --output "$S/sync" --device cpu
timed cache "${DDF[@]}" cache build --manifest "$S/neutral.csv" --dataset-root "$S" --cache-root "$S/cache" \
    --index "$S/cache-index.csv" --audit "$S/cache-audit.json" --dataset FakeAVCeleb --device cpu --code-version p001
echo "preprocessing hash: $($PY -c "import json;print(json.load(open('$S/cache-audit.json'))['preprocessing_hash'])") (P-001: $PREP)"
for branch in visual audio; do
    "${DDF[@]}" manifest usable --manifest "$S/neutral.csv" --cache-index "$S/cache-index.csv" --cache-root "$S/cache" \
        --output "$S/usable-$branch.csv" --audit "$S/usable-$branch.json" --dataset FakeAVCeleb --branch "$branch" \
        --preprocessing-hash "$PREP" > "$S/usable-$branch.log" 2>&1
    for seed in 17 29 43; do
        name=$branch-seed$seed
        timed "eval-$name" "${DDF[@]}" evaluate branch --branch "$branch" --checkpoint "runs/p001/checkpoints/$name.pt" \
            --manifest "$S/usable-$branch.csv" --cache-index "$S/cache-index.csv" --cache-root "$S/cache" \
            --dataset FakeAVCeleb --threshold-file "runs/p001/evaluation/$name-threshold.json" \
            --evidence-scope generalization_cross_corpus --preprocessing-hash "$PREP" --device cpu \
            --output "$S/eval/$name-metrics.json" --predictions "$S/eval/$name-predictions.csv"
    done
done
