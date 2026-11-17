#!/usr/bin/env bash
# After runs/p001-2day.sh: copy the target and H4 outputs into results/, then rebuild every number,
# figure and deliverable from them. Nothing is committed here; read the outputs first.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
grep -q "two-day chain done" runs/p001-2day.log || { echo "2-day chain not finished"; exit 1; }
grep -q "a10 done" runs/p001-a10.log || { echo "A10 rescoring not finished"; exit 1; }
# Rerun safely after the H2 chain too: every step overwrites its outputs from runs/.
mkdir -p results/p001-sync-targets results/p001-h4
# Amendment A9: the neutral LAV-DF audit under the window label, once.
[ -f results/shortcut-audit/lavdf-window-neutral/summary.json ] ||     $PY scripts/shortcut_audit.py --manifest data/manifests/lavdf-test-window.csv --dataset lavdf-window-neutral         --output results/shortcut-audit/lavdf-window-neutral --workers 4 | grep -v '^[0-9]*/[0-9]*$'
cp -r runs/p001/sync-targets/. results/p001-sync-targets/
cp -r runs/p001/h4/. results/p001-h4/
# Results from before amendment A10 stay on record next to the corrected ones.
mkdir -p results/p001-sync-targets-pre-a10 results/p001-h4-pre-a10
[ -d runs/p001/sync-targets-pre-a10 ] && cp -r runs/p001/sync-targets-pre-a10/. results/p001-sync-targets-pre-a10/
[ -d runs/p001/h4-pre-a10 ] && cp -r runs/p001/h4-pre-a10/. results/p001-h4-pre-a10/
# After the H2 chain: visual and audio results (with their threshold files) and the fusion test.
if [ -f runs/p001/h2/h2.json ]; then
    mkdir -p results/p001-branches results/p001-h2
    cp runs/p001/evaluation/*-predictions.csv runs/p001/evaluation/*-threshold.json results/p001-branches/
    # The branch reports have no identity intervals, so each test and target set is scored again with the
    # same evaluator as the sync path. The branch reports are kept as *-branch-report.json. Validation is
    # left out: its clips chose the threshold, and the evaluator refuses that overlap.
    for pred in runs/p001/evaluation/*-predictions.csv; do
        base=$(basename "$pred" -predictions.csv)
        case "$base" in *-val) continue ;; esac
        name=${base%%-seed*}; seed=${base#*-seed}; seed=${seed%%-*}
        cp "runs/p001/evaluation/$base-metrics.json" "results/p001-branches/$base-branch-report.json"
        $PY -m deepfake_detection.cli evaluate predictions --predictions "$pred"             --threshold-file "runs/p001/evaluation/$name-seed$seed-threshold.json"             --output "results/p001-branches/$base-metrics.json" --bootstrap-samples 1000 --seed 17 > /dev/null
    done
    # The full run (with HiFi-AVDF) has its own folder; its family targets match runs/p001/h2 byte for byte.
    H2SRC=runs/p001/h2; [ -f runs/p001/h2-full/h2.json ] && H2SRC=runs/p001/h2-full
    cp -r "$H2SRC/." results/p001-h2/
fi
$PY scripts/build_numbers.py
$PY scripts/make_figures.py
$PY deliverables/overleaf.py || $PY deliverables/overleaf.py --draft
$PY deliverables/slides/build_deck.py || $PY deliverables/slides/build_deck.py --draft
$PY deliverables/poster/build_poster.py || $PY deliverables/poster/build_poster.py --draft
powershell -NoProfile -ExecutionPolicy Bypass -File deliverables/slides/export.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File deliverables/slides/export.ps1 -Name Deepfake_Detection_Poster
echo "finalize done"
