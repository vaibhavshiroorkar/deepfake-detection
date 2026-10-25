#!/bin/bash
# The freeze scoring chain: heads (CPU), crops of the sealed test videos (CPU), then the score file of every iteration of the staircase (GPU, one job at a time).
#
#   bash scripts/freeze/run_all.sh --final       the real run: reads the sealed splits. Start it only at the freeze.
#   bash scripts/freeze/run_all.sh --dry-run     smoke test of everything that can run without the GPU and without a sealed split (small, --limit 200)
#
# Logs: data/_logs/freeze_<step>.log. Marker data/_final/DONE (final) or data/_final_dry/DONE (dry). Resumable: heads, crops, features and fine-tuned
# scores are cached (data/_final/features, data/_final/ftcache), so after an interruption run the same command again.
# This script does not run the evaluation; the commands for that are in docs/_agents/FREEZE.md (scripts/freeze/run_eval.sh).
set -u
MODE="${1:-}"
if [ "$MODE" != "--final" ] && [ "$MODE" != "--dry-run" ]; then
  echo "usage: bash scripts/freeze/run_all.sh --final | --dry-run"; exit 2
fi
cd "$(dirname "$0")/../.." || exit 2
export PYTHONPATH=src
PY=.venv/Scripts/python.exe
LOG=data/_logs
if [ "$MODE" = "--final" ]; then ROOT=data/_final; EXTRA=(--final); else ROOT=data/_final_dry; EXTRA=(--dry-run --limit 200); fi
mkdir -p "$ROOT"
rm -f "$ROOT/DONE"

busy() { used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${used:-0}" -gt 2000 ]; }
if [ "$MODE" = "--final" ] && busy; then
  echo "refused: the GPU holds more than 2000 MiB, another GPU job is running (one GPU job at a time)"; exit 3
fi

step() { echo "$(date +%H:%M:%S) $*"; }

step "heads"
$PY scripts/freeze/prepare_heads.py > $LOG/freeze_heads.log 2>&1 || { step "heads FAILED, see $LOG/freeze_heads.log"; exit 1; }
if [ "$MODE" = "--final" ]; then
  step "crops of the sealed test videos"
  $PY scripts/freeze/prepare_crops.py --workers 8 > $LOG/freeze_crops.log 2>&1 || { step "crops FAILED"; exit 1; }
  for f in data/_results/baseline/DONE data/_results/baseline/DONE_b; do [ -f "$f" ] || { step "the Xception baseline ($f, E-013 / E-013b) has not finished: BASE will fail"; }; done
fi

for TAG in I-0 I-1 I-2 I-3 I-4 BASE I-5 I-6 I-6b; do
  if [ "$TAG" = "BASE" ] && { [ ! -f data/_results/baseline/DONE ] || [ ! -f data/_results/baseline/DONE_b ]; } && [ "$MODE" = "--final" ]; then
    step "BASE skipped: data/_results/baseline/DONE or DONE_b is missing"; continue
  fi
  step "scoring $TAG"
  $PY scripts/freeze/score_final.py --iteration $TAG "${EXTRA[@]}" > $LOG/freeze_score_$TAG.log 2>&1 || { step "$TAG FAILED, see $LOG/freeze_score_$TAG.log"; exit 1; }
done

step "checking the score files"
if [ "$MODE" = "--final" ]; then $PY scripts/freeze/check_score_files.py > $LOG/freeze_check.log 2>&1; else $PY scripts/freeze/check_score_files.py --dry > $LOG/freeze_check.log 2>&1; fi
RC=$?
tail -n 40 $LOG/freeze_check.log
date +"%Y-%m-%d %H:%M:%S" > "$ROOT/DONE"
step "done (check exit code $RC); next: the evaluation, scripts/freeze/run_eval.sh (docs/_agents/FREEZE.md)"
exit $RC
