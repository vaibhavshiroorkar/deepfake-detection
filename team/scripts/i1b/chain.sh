#!/bin/bash
# waits for the E-013b baseline run, then I-1b (E-016), then the I-1c members (E-017); one GPU job at a time
cd "$(dirname "$0")/../.."
export PYTHONPATH=src
while [ ! -f data/_results/baseline/DONE_b ]; do sleep 20; done
.venv/Scripts/python.exe scripts/i1b/run_i1b.py > data/_logs/i1b_run.log 2>&1
.venv/Scripts/python.exe scripts/i1b/run_i1c.py > data/_logs/i1c_run.log 2>&1
