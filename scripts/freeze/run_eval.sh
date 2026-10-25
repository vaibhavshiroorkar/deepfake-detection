#!/bin/bash
# The one-time final evaluation of every score file (PROTOCOL.md section 7). Each iteration tag is evaluated once; a second evaluation of the same
# iteration and dataset is refused by the guard unless --repeat "<reason>" is added by hand (the reason is stored in the ledger and listed under
# "Mistakes and corrections" in docs/STATUS.md). The ledger entries are written only after an evaluation succeeded.
#
#   bash scripts/freeze/run_eval.sh               all tags that have a score file in data/_final/scores
#   bash scripts/freeze/run_eval.sh I-0 I-1       only these tags
#
# Run it only after scripts/freeze/check_score_files.py reported "all checks passed" on the real score files.
set -u
cd "$(dirname "$0")/../.." || exit 2
export PYTHONPATH=src
PY=.venv/Scripts/python.exe
mkdir -p data/_final/eval
if [ $# -gt 0 ]; then TAGS="$*"; else TAGS=$(ls data/_final/scores/*.csv | xargs -n1 basename | sed 's/\.csv$//' | grep -v partial); fi
for TAG in $TAGS; do
  echo "== final evaluation of $TAG"
  $PY -m dfd.eval.run --scores data/_final/scores/$TAG.csv --final --iteration $TAG --out data/_final/eval/$TAG.json > data/_logs/freeze_eval_$TAG.log 2>&1
  RC=$?
  tail -n 14 data/_logs/freeze_eval_$TAG.log
  [ $RC -eq 0 ] || { echo "$TAG: evaluation exited with $RC (2 = refused by the guard); stopping"; exit $RC; }
done
echo "done; reports in data/_final/eval, ledger entries in ledger/final_ledger.jsonl (commit both)"
