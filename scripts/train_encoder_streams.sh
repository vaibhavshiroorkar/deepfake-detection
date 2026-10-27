#!/bin/bash
# Read the preprocessing contract from the source cache and keep new evidence separate.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
RUN=${1:-runs/encoder-contract-v2-20261024}
SOURCE=${2:-runs/program-20260906}
mkdir -p "$RUN/logs"
prep=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["preprocessing_hash"])' "$SOURCE/cache-audit.json")
for spec in "lipsync avhubert sync-usable" "emotion hsemotion usable"; do
  read -r stream encoder subset <<< "$spec"
  name="stream-$stream-$encoder"
  checkpoint="$RUN/checkpoints/$name.pt"
  history="$RUN/checkpoints/$name-history.json"
  "$PY" scripts/research_artifacts.py --receipt "$RUN/receipts/$name.json" \
    --outputs "$checkpoint" "$history" --inputs "$SOURCE/split/train-$subset.csv" \
    "$SOURCE/split/val-$subset.csv" "$SOURCE/cache-index.csv" "$SOURCE/cache-audit.json" -- \
    "$PY" -m deepfake_detection.cli train stream --stream "$stream" --encoder "$encoder" \
    --train-manifest "$SOURCE/split/train-$subset.csv" --validation-manifest "$SOURCE/split/val-$subset.csv" \
    --cache-index "$SOURCE/cache-index.csv" --cache-root "$SOURCE/cache" --dataset FakeAVCeleb \
    --checkpoint "$checkpoint" --history "$history" --run-id "$name" --split-hash "$(basename "$SOURCE")" \
    --preprocessing-hash "$prep" --device cuda --epochs 10 --batch-size 4 --accumulation-steps 4 \
    --freeze-epochs 2 --patience 3 --learning-rate 0.0001 --weight-decay 0.0001 \
    --encoder-learning-rate 0.000005 --workers 0 --seed 17 > "$RUN/logs/train-$name.log" 2>&1
done
