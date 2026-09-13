#!/bin/sh
# Train the visual stream on FaceForensics++ c23, then score it zero-shot on the
# corpora it has never seen. This is the protocol behind nearly every published
# cross-dataset table, and it is the first time a number from this project can
# be compared with one.
#
# Waits for the cache build to finish first, so it can be launched while that is
# still running. Everything here is idempotent: a stage whose output exists is
# skipped.
set -e
cd "C:/Users/vaibh/Documents/GitHub/deepfake-generalization"
PY=.venv/Scripts/python.exe
RUN=runs/ffpp-20260913
SRC=runs/program-20260906

say() { echo "=== $1 $(date +%H:%M:%S)"; }

say "waiting for the cache build"
while [ "$(powershell -NoProfile -Command '@(Get-Process python -ErrorAction SilentlyContinue).Count' | tr -d '\r')" != "0" ]; do
  sleep 60
done
say "cache build finished, $(find $RUN/cache -name '*.npz' | wc -l) clips"

if [ ! -f "$RUN/cache-index.csv" ]; then
  say "merging shard indexes"
  $PY -m deepfake_detection.cli cache merge \
    --indexes $RUN/cache-index-0.csv $RUN/cache-index-1.csv $RUN/cache-index-2.csv $RUN/cache-index-3.csv $RUN/cache-index-4.csv \
    --audits $RUN/cache-audit-0.json $RUN/cache-audit-1.json $RUN/cache-audit-2.json $RUN/cache-audit-3.json $RUN/cache-audit-4.json \
    --index $RUN/cache-index.csv --audit $RUN/cache-audit.json
fi

HASH=$($PY -c "import json;print(json.load(open('$RUN/cache-audit.json'))['preprocessing_hash'])")
say "preprocessing hash $HASH"

for PART in train val test; do
  if [ ! -f "$RUN/split/$PART-usable.csv" ]; then
    say "filtering $PART to clips with a visual view"
    $PY -m deepfake_detection.cli manifest usable \
      --manifest $RUN/split/$PART.csv --cache-index $RUN/cache-index.csv \
      --cache-root $RUN/cache --output $RUN/split/$PART-usable.csv \
      --audit $RUN/split/usable-$PART.json --dataset FaceForensics++ \
      --branch visual --preprocessing-hash "$HASH"
  fi
done

if [ ! -f "$RUN/checkpoints/visual-efficientnet.pt" ]; then
  say "training visual-efficientnet on FF++"
  $PY -m deepfake_detection.cli train visual-stream --backbone efficientnet \
    --freeze-epochs 2 --batch-size 8 --accumulation-steps 2 --frame-chunk-size 8 \
    --train-manifest $RUN/split/train-usable.csv \
    --validation-manifest $RUN/split/val-usable.csv \
    --cache-index $RUN/cache-index.csv --cache-root $RUN/cache \
    --dataset FaceForensics++ \
    --checkpoint $RUN/checkpoints/visual-efficientnet.pt \
    --history $RUN/checkpoints/visual-efficientnet-history.json \
    --run-id ffpp-visual-efficientnet --split-hash ffpp-20260913 \
    --preprocessing-hash "$HASH" \
    --device cuda --epochs 10 --workers 0 2>&1 | tee $RUN/logs/train-visual.log
fi

say "done"
