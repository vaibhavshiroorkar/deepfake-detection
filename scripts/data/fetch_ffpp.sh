#!/bin/bash
# Runs the FaceForensics++ download script supplied by the dataset authors (after the user's approval,
# D-020) for the datasets chosen in D-021, one dataset after another.
# Usage: scripts/data/fetch_ffpp.sh <dataset> [<dataset> ...]   e.g. original Deepfakes
# Datasets: original Deepfakes Face2Face FaceSwap NeuralTextures DeepFakeDetection_original DeepFakeDetection
# Never use "-d all": it also pulls DFD and FaceShifter (about 25 GB more).
# The script prompts for confirmation of the FaceForensics terms of use, which the user accepted in the
# request form; the prompt is answered with Enter. The script skips files that already exist, so a
# re-run resumes. Server EU2 is the only available one (approval email, 2026-10-24).
cd "$(dirname "$0")/../.." || exit 1
for d in "$@"; do
  case "$d" in
    DeepFakeDetection_original|DeepFakeDetection) out="data/dfd" ;;
    *) out="data/ffpp" ;;
  esac
  echo "[$(date +%FT%T)] start $d -> $out"
  printf '\n' | uv run --no-project --with tqdm python download-FaceForensics.py "$out" \
      -d "$d" -c c23 -t videos --server EU2
  echo "[$(date +%FT%T)] finished $d (exit $?)"
done
