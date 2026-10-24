#!/bin/bash
# Re-measure fine-tuning step times with the GPU free: waits for the E-001 queue (run_e001.py) to exit, then runs step_time.
cd "d:/Final Year Project"
while powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object { \$_.CommandLine -match 'run_e001|extract_video' }) { exit 0 } else { exit 1 }"; do sleep 30; done
HF_HUB_CACHE='D:\cache\hf\hub' PYTHONPATH=src .venv/Scripts/python.exe -m dfd.sweep.step_time --tag alone > data/_logs/step_time_alone.log 2>&1
python - <<'PY'
import time
from datetime import date
p="docs/STATUS.md"
t=open(p,encoding="utf-8").read()
t=t.rstrip("\n")+f"\n- {date.today().isoformat()} {time.strftime('%H:%M')} Step times with the GPU free measured (data/_logs/step_time_alone.json); D-042 numbers to be checked against them.\n"
open(p,"w",encoding="utf-8",newline="").write(t)
PY
