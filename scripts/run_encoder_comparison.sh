#!/bin/bash
# Re-extract corrected features without replacing historical evidence.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
RUN=${1:-runs/encoder-normalisation-v2-20261024}
FOLDER=${2:-data/df26-norm}
mkdir -p "$RUN"
for encoder in dinov3 pe clip; do
  features="$RUN/df26-$encoder.npz"
  "$PY" scripts/extract_probe_features.py --folder "$FOLDER" --output "$features" \
    --encoder "$encoder" --device cuda --resume
  for seed in 17 29 43; do
    report="$RUN/df26-$encoder-logo-seed$seed.json"
    predictions="$RUN/df26-$encoder-logo-seed$seed-predictions.json"
    "$PY" scripts/research_artifacts.py --receipt "$RUN/receipts/$encoder-seed$seed.json" \
      --outputs "$report" "$predictions" --inputs "$features" "${features%.npz}.json" -- \
      "$PY" scripts/train_logo_probe.py --features "$features" \
      --modes-are-one-generator --seed "$seed" --output "$report"
  done
done
for seed in 17 29 43; do
  report="$RUN/encoder-comparison-seed$seed.json"
  "$PY" scripts/research_artifacts.py --receipt "$RUN/receipts/comparison-seed$seed.json" \
    --outputs "$report" --inputs "$RUN/df26-dinov3.npz" "$RUN/df26-pe.npz" "$RUN/df26-clip.npz" -- \
    "$PY" scripts/compare_probe_encoders.py --features "$RUN/df26-dinov3.npz" \
    "$RUN/df26-pe.npz" "$RUN/df26-clip.npz" --seed "$seed" --output "$report"
done
