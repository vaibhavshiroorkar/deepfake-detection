#!/bin/sh
# Rerun extract.py for eval and small until every shard of both subsets is extracted (W1-I). Run from the repo root.
export PYTHONPATH=src
while :; do
  ne=$(ls data/_results/modern/e028/cache/*.npz 2>/dev/null | wc -l)
  ns=$(ls data/_results/modern/e029/cache_small/*.npz 2>/dev/null | wc -l)
  echo "$(date '+%H:%M:%S') eval $ne/92 small $ns/15"
  [ "$ne" -ge 92 ] && [ "$ns" -ge 15 ] && break
  .venv/Scripts/python.exe scripts/modern_image/extract.py eval 6 2>&1 | grep -E "rows|done|Error|Traceback" | tail -3
  .venv/Scripts/python.exe scripts/modern_image/extract.py small 6 2>&1 | grep -E "rows|done|Error|Traceback" | tail -3
  sleep 120
done
