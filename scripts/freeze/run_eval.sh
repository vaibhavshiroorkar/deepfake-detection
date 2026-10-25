#!/bin/bash
# The one-time final evaluation of every score file (PROTOCOL.md section 7). Each iteration tag is evaluated once; a second evaluation of the same
# iteration and dataset is refused by the guard unless --repeat "<reason>" is added by hand (the reason is stored in the ledger and listed under
# "Mistakes and corrections" in docs/STATUS.md). The ledger entries are written only after an evaluation succeeded.
#
#   bash scripts/freeze/run_eval.sh               all tags that have a score file in data/_final/scores
#   bash scripts/freeze/run_eval.sh I-0 I-1       only these tags
#   DRY=1 bash scripts/freeze/run_eval.sh         readiness test (D-062, STEP 2): the dry-run score files of data/_final_dry (val, dev_x, LAV-DF val only),
#                                                 evaluated WITHOUT --final, ledger and reports under data/_final_dry; it reads no sealed split
#
# Run it only after scripts/freeze/check_score_files.py reported "all checks passed" on the real score files.
set -u
cd "$(dirname "$0")/../.." || exit 2
export PYTHONPATH=src
PY=.venv/Scripts/python.exe
if [ "${DRY:-0}" = "1" ]; then ROOT=data/_final_dry; FLAGS=(--ledger data/_final_dry/ledger.jsonl); LOGP=freeze_evaldry; else ROOT=data/_final; FLAGS=(--final); LOGP=freeze_eval; fi
mkdir -p $ROOT/eval
if [ $# -gt 0 ]; then TAGS="$*"; else TAGS=$(ls $ROOT/scores/*.csv | xargs -n1 basename | sed 's/\.csv$//' | grep -v partial); fi
for TAG in $TAGS; do
  echo "== evaluation of $TAG ($ROOT)"
  $PY -m dfd.eval.run --scores $ROOT/scores/$TAG.csv "${FLAGS[@]}" --iteration $TAG --out $ROOT/eval/$TAG.json > data/_logs/${LOGP}_$TAG.log 2>&1
  RC=$?
  tail -n 14 data/_logs/${LOGP}_$TAG.log
  [ $RC -eq 0 ] || { echo "$TAG: evaluation exited with $RC (2 = refused by the guard); stopping"; exit $RC; }
done
echo "done; reports in $ROOT/eval" ; [ "${DRY:-0}" = "1" ] || echo "ledger entries in ledger/final_ledger.jsonl (commit both)"
