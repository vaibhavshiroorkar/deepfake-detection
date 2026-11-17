#!/usr/bin/env bash
# P-001 amendment A10 (2026-11-16): redo target windows whose face share is below 0.8 with the
# YuNet locator, then rescore. Windows that passed are copied unchanged. Results from before A10
# move to runs/p001/sync-targets-pre-a10 and runs/p001/h4-pre-a10; nothing is deleted.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
stamp() { echo "[$(date '+%a %H:%M:%S')] $*"; }
FILTER='^W0000|^I0000|Feedback manager|xnnpack|XNNPACK|inference_feedback'
mkdir -p runs/p001/sync-targets-pre-a10/receipts runs/p001/h4-pre-a10
for t in dfdc hifiavdf lavdf-test; do
    src="runs/avhubert-$t-neutral"; dst="runs/avhubert-$t-neutral-a10"
    stamp "seed $dst with windows that already pass"
    $PY - "$src" "$dst" <<'EOF'
import csv, shutil, sys
from pathlib import Path
sys.path.insert(0, "src")
from deepfake_detection.avsync.features import cache_path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
if (dst / "index.csv").exists():
    print("already seeded"); raise SystemExit(0)
dst.mkdir(parents=True, exist_ok=True)
with (src / "index.csv").open(newline="", encoding="utf-8") as h:
    reader = csv.DictReader(h); fields = reader.fieldnames
    keep = [r for r in reader if r["status"] == "ok" and float(r["face"]) >= 0.8]
for r in keep:
    target = cache_path(dst, r["window_id"]); target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cache_path(src, r["window_id"]), target)
with (dst / "index.csv").open("w", newline="", encoding="utf-8") as h:
    w = csv.DictWriter(h, fieldnames=fields); w.writeheader(); w.writerows(keep)
print(f"copied {len(keep)} passing windows")
EOF
    stamp "extract $t with the locator"
    $PY scripts/extract_avhubert_features.py --manifest "data/manifests/$t-neutral.csv" \
        --cuts "data/manifests/$t-neutral.cuts.csv" --windows 2 --step 1 --output "$dst" --workers 4 \
        --locator artifacts/models/face_detection_yunet_2026may.onnx 2>&1 | grep -Ev "$FILTER" | tail -2
done
for spec in "dfdc:data/manifests/dfdc-neutral.csv:dfdc" "hifiavdf:data/manifests/hifiavdf-neutral.csv:hifiavdf" \
            "lavdf-test:data/manifests/lavdf-test-neutral.csv:lavdf-test" "lavdf-window:data/manifests/lavdf-test-window.csv:lavdf-test"; do
    IFS=: read -r name manifest feat <<< "$spec"
    stamp "rescore $name"
    for f in runs/p001/sync-targets/sync-$name-seed*; do [ -e "$f" ] && mv "$f" runs/p001/sync-targets-pre-a10/; done
    for f in runs/p001/sync-targets/receipts/evaluate-$name-seed*; do [ -e "$f" ] && mv "$f" runs/p001/sync-targets-pre-a10/receipts/; done
    for f in runs/p001/h4/sync-$name-seed*; do [ -e "$f" ] && mv "$f" runs/p001/h4-pre-a10/; done
    bash runs/p001-rescore.sh "$name" "$manifest" "runs/avhubert-$feat-neutral-a10" 2>&1 | grep -v '^\s*$' | tail -12
done
stamp "a10 done"
