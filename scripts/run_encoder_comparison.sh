#!/bin/bash
# Settle A-encoder-comparison, the registry's one pending row.
#
# The claim it holds is that Perception Encoder reads 72.7 macro on DF26 against
# DINOv3's 57.0 and CLIP's 55.6, so "the encoder choice is worth 15 points". That
# drove the frozen-probe recommendation for weeks and has no artifact behind it.
# Its DINOv3 figure also contradicts this project's own registered probe, which
# reads 0.6786 frame-rate-controlled and 0.6942 uncontrolled, so the comparison
# cannot have come from this pipeline.
#
# The recorded reason it was never run was that `open_clip_torch` could not be
# installed. It installs in under a second and is now the `probe` extra.
#
# Two encoders to add, on the frame-rate-controlled corpus so the arms are
# comparable with the existing DINOv3 row rather than with the uncontrolled one.
#
#   bash scripts/run_encoder_comparison.sh
#
# Waits for any training already on the GPU before starting, because feature
# extraction is I/O and GPU heavy and contending with a training run slows both.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
PROBE=runs/probe-20260914
FOLDER=data/df26-norm

# No wait loop. Two attempts at one were both wrong: grepping tasklist for python
# also matches any verification command running alongside, and
# --query-compute-apps lists every process holding a GPU context including desktop
# applications, so both waited forever. Run this when the GPU is free instead; a
# check that is subtly wrong is worse than a precondition stated plainly.

for ENCODER in pe clip; do
  FEATURES="$PROBE/df26-norm-$ENCODER.npz"
  if [ ! -f "$FEATURES" ]; then
    echo "=== extract $ENCODER ==="
    $PY scripts/extract_probe_features.py \
      --folder "$FOLDER" --output "$FEATURES" --encoder "$ENCODER" --device cuda
    echo "extract $ENCODER exit=$?"
  fi
  echo "=== probe $ENCODER ==="
  # Modes grouped under one model: an image-to-video and a text-to-video output
  # of the same model share weights, so holding out one while training on its
  # sibling is not an unseen generator.
  $PY scripts/train_logo_probe.py \
    --features "$FEATURES" --modes-are-one-generator
  echo "probe $ENCODER exit=$?"
done

echo "=== the DINOv3 arm already on disk, for comparison ==="
$PY -c "
import json
for name in ('df26-norm-dinov3-logo.json',):
    d = json.load(open('runs/probe-20260914/' + name))
    print(name, 'macro_mean', round(d['macro_mean'], 4))
"
echo "ENCODER_COMPARISON_DONE"
