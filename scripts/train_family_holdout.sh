#!/bin/sh
# Leave one manipulation family out, six times.
#
# The protocol calls this mandatory and it had never been run. Training on five
# families and testing on the sixth is the closest thing available to "what
# happens when the next generator arrives", which is the property the whole
# design is organised around.
#
# Each arm trains on real clips plus five families and is scored against the same
# real clips plus the held-out family's fakes, so its row is directly comparable
# with the corresponding row of the seen-family baseline in
# `runs/ffpp-20260913/evaluation/by-family-seen.json`.
#
# Sequential on purpose: each arm wants the whole GPU. Re-running skips arms
# whose checkpoint already exists.
set -e
cd "C:/Users/vaibh/Documents/GitHub/deepfake-generalization"
PY=.venv/Scripts/python.exe
RUN=runs/ffpp-20260913
HASH=a6fe6c0d041538d9fda06a0898bec876f35ebb7f2e0090f025af29fe8f6df6c6
HOLDOUT_DIR="$RUN/holdout"
mkdir -p "$HOLDOUT_DIR" "$RUN/logs"

FAMILIES="ffpp-deepfakes ffpp-face2face ffpp-faceshifter ffpp-faceswap ffpp-neuraltextures ffpp-deepfakedetection"

for FAMILY in $FAMILIES; do
  echo "=== $FAMILY $(date +%H:%M:%S)"
  CKPT="$RUN/checkpoints/holdout-$FAMILY.pt"
  if [ -f "$CKPT" ]; then echo "    already trained, skipping"; continue; fi

  # Split the manifests around the held-out family. Real clips stay in both
  # sides: the arm must still learn what authentic looks like.
  $PY - "$FAMILY" "$RUN" "$HOLDOUT_DIR" <<'PYEOF'
import csv, sys
from pathlib import Path

family, run, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
for part in ("train", "val", "test"):
    source = run / "split" / f"{part}-usable.csv"
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames or []
    if part == "test":
        # The held-out family plus every real clip, which is what the arm is
        # scored on.
        kept = [r for r in rows if r["method"] in (family, "real")]
    else:
        kept = [r for r in rows if r["method"] != family]
    target = out / f"{family}-{part}.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
    fakes = sum(1 for r in kept if r["clip_fake"] == "1")
    print(f"    {part}: {len(kept)} clips, {fakes} fake -> {target.name}")
PYEOF

  $PY -m deepfake_detection.cli train visual-stream --backbone efficientnet \
    --freeze-epochs 2 --batch-size 8 --accumulation-steps 2 --frame-chunk-size 8 \
    --train-manifest "$HOLDOUT_DIR/$FAMILY-train.csv" \
    --validation-manifest "$HOLDOUT_DIR/$FAMILY-val.csv" \
    --cache-index "$RUN/cache-index.csv" --cache-root "$RUN/cache" \
    --dataset FaceForensics++ \
    --checkpoint "$CKPT" \
    --history "$RUN/checkpoints/holdout-$FAMILY-history.json" \
    --run-id "holdout-$FAMILY" --split-hash ffpp-20260913 \
    --preprocessing-hash "$HASH" \
    --device cuda --epochs 10 --workers 0 \
    > "$RUN/logs/holdout-$FAMILY.log" 2>&1
  echo "    trained exit=$? $(date +%H:%M:%S)"

  $PY scripts/score_by_family.py --run-dir "$RUN" --checkpoint "$CKPT" \
    --manifest "$HOLDOUT_DIR/$FAMILY-test.csv" --label "unseen-$FAMILY" \
    >> "$RUN/logs/holdout-$FAMILY.log" 2>&1
  echo "    scored $(date +%H:%M:%S)"
done

echo "=== all family holdouts done $(date +%H:%M:%S)"
