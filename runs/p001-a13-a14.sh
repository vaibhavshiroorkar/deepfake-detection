#!/usr/bin/env bash
# P-001 amendments A13 and A14 (exploratory), after A11 has finished, one at a time:
# A13 reads every LAV-DF real-video cache file (CPU and disk), A14 rescores the audio models (GPU).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
RUN=runs/p001
stamp() { echo "[$(date '+%a %H:%M:%S')] $*"; }
until grep -q "a11 done v2" runs/p001-a11.log; do sleep 300; done
stamp "start"
$PY scripts/research_artifacts.py --receipt "$RUN/receipts/a13.json" --outputs "$RUN/a13/a13.json" "$RUN/a13/statistics.csv" -- \
    $PY scripts/run_a13.py --output "$RUN/a13" && stamp "a13 done" || stamp "a13 failed"
$PY scripts/research_artifacts.py --receipt "$RUN/receipts/a14.json" --outputs "$RUN/a14/summary.json" -- \
    $PY scripts/run_a14.py --output "$RUN/a14" && stamp "a14 done" || stamp "a14 failed"
stamp "a13-a14 chain done"
