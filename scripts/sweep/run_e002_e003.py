"""E-002 and E-003 queue: wait for the GPU, extract features for every candidate, then measure the section 7 cost record.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/sweep/run_e002_e003.py

Order: wait until no E-001 extraction runs; images (E-002) for the 7 candidates; speech (E-003) for the 4 candidates; the cost record
of T-V, T-I and T-A (GPU otherwise idle). Every step is resumable (finished feature files are skipped). The ranking chain
(scripts/sweep/chain_rank.py) picks each task up when its features and cost record exist. One line per finished block goes to
docs/STATUS.md. Nothing is chosen here.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("HF_HUB_CACHE", r"D:\cache\hf\hub")
os.environ.setdefault("TORCH_HOME", r"D:\cache\torch")
PY = sys.executable
LOG = Path("data/_logs/e002_e003_queue.log")


def status_line(text: str) -> None:
    p = Path("docs/STATUS.md")
    t = p.read_text(encoding="utf-8")
    if "## Background job log" not in t:
        t = t.rstrip("\n") + "\n\n## Background job log\n\n"
    p.write_text(t.rstrip("\n") + f"\n- {date.today().isoformat()} {time.strftime('%H:%M')} {text}\n", encoding="utf-8", newline="")


def gpu_busy() -> bool:
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "if (Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object { $_.CommandLine -match 'extract_video|step_time|dfd.sweep.train_reference' }) { exit 0 } else { exit 1 }"])
    return r.returncode == 0


def run(mod: str, *args: str) -> int:
    with open(LOG, "a") as fh:
        fh.write(f"\n$ python -m {mod} {' '.join(args)}\n")
        fh.flush()
        return subprocess.run([PY, "-m", mod, *args], stdout=fh, stderr=subprocess.STDOUT).returncode


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    while gpu_busy():
        time.sleep(30)
    from dfd.sweep import encoders
    fails = []
    for cid in encoders.IMAGE_IDS:
        if run("dfd.sweep.extract_images", "--cand", cid):
            fails.append(cid)
    status_line(f"E-002 extraction queue finished (T-I, 7 image encoders); failures: {fails or 'none'}; log data/_logs/e002_e003_queue.log.")
    n = len(fails)
    for cid in encoders.AUDIO_IDS:
        if run("dfd.sweep.extract_audio", "--cand", cid):
            fails.append(cid)
    status_line(f"E-003 extraction queue finished (T-A, 4 speech encoders); failures: {fails[n:] or 'none'}.")
    for task in ("T-V", "T-I", "T-A"):
        if run("dfd.sweep.cost_record", "--task", task):
            fails.append("cost:" + task)
    status_line(f"Section 7 cost record written to data/_results/cost_record.json (failures: {[f for f in fails if f.startswith('cost')] or 'none'}); the ranking chain now ranks T-V (again, with this cost record), T-I and T-A.")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
