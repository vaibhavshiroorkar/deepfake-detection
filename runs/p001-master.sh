#!/usr/bin/env bash
# P-001 full run on neutralizer v4 (amendment A4).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
stamp() { echo "[$(date '+%a %H:%M:%S')] $*"; }
FILTER='^W0000|^I0000|Feedback manager|xnnpack|XNNPACK'

stamp "neutralize FakeAVCeleb"
$PY scripts/neutralize_dataset.py --manifest data/manifests/fakeavceleb.csv --dataset FakeAVCeleb \
    --output-root data/neutral --manifest-out data/manifests/fakeavceleb-neutral.csv --workers 14 | tail -40
stamp "H1 audit"
$PY scripts/shortcut_audit.py --manifest data/manifests/fakeavceleb-neutral.csv --dataset FakeAVCeleb-neutral-v4 \
    --output results/shortcut-audit/fakeavceleb-neutral --workers 14 | grep -v '^[0-9]*/[0-9]*$'

stamp "targets in background"
(
  for spec in "runs/streams-20260905/dfdc-manifest.csv:DFDC:dfdc" \
              "data/manifests/lavdf-test-sample.csv:LAV-DF:lavdf-test" \
              "data/manifests/lavdf-dev-sample.csv:LAV-DF:lavdf-dev" \
              "data/manifests/lavdf-train-sample.csv:LAV-DF:lavdf-train"; do
      IFS=: read -r manifest dataset name <<< "$spec"
      $PY scripts/neutralize_dataset.py --manifest "$manifest" --dataset "$dataset" --output-root data/neutral \
          --manifest-out "data/manifests/$name-neutral.csv" --workers 6 | grep -A20 rows_failed
  done
  echo "[$(date '+%a %H:%M:%S')] targets done"
) > runs/p001-targets.log 2>&1 &

stamp "split"
.venv/Scripts/ddf.exe split build --manifest data/manifests/fakeavceleb-neutral.csv \
    --output-dir runs/p001/split --dataset FakeAVCeleb --seed 17
stamp "AV-HuBERT features"
$PY scripts/extract_avhubert_features.py --manifest data/manifests/fakeavceleb-neutral.csv \
    --cuts data/manifests/fakeavceleb-neutral.cuts.csv --windows 2 --step 1 \
    --output runs/avhubert-fakeavceleb-neutral --workers 8 2>&1 | grep -Ev "$FILTER" || true
stamp "sync path"
$PY scripts/train_sync_path.py --features runs/avhubert-fakeavceleb-neutral --split-dir runs/p001/split \
    --output runs/p001/sync
stamp "visual and audio paths"
bash scripts/run_p001_branches.sh
stamp "master done"
