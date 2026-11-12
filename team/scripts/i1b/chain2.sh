#!/bin/bash
# reruns E-013b seed 2 (it failed when finetune.py was edited under the running process, see STATUS.md mistakes) after the I-1c runs
cd "$(dirname "$0")/../.."
export PYTHONPATH="src;scripts/baseline"
while [ ! -f data/_results/I-1c/DONE ]; do sleep 20; done
.venv/Scripts/python.exe scripts/baseline/run_baseline_b.py > data/_logs/baseline_b_rerun_s2.log 2>&1
