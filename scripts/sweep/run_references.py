"""Train the from-scratch references (IMG-20 for T-V and T-I, AU-REF for T-A) after the cost record, with the GPU otherwise idle.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/sweep/run_references.py

Waits until the section 7 cost record exists for all three tasks and no extraction or other sweep job runs, then trains the references
under the fixed budget of docs/O007_SWEEP.md section 8 (dfd.sweep.train_reference). Resumable: a task whose reference_scores.npz exists
is skipped. The ranking chain needs the reference scores, so it ranks each task after this. One line per task goes to docs/STATUS.md.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("HF_HUB_CACHE", r"D:\cache\hf\hub")
RESULTS = Path("data/_results")
LOG = Path("data/_logs/references_queue.log")


def status_line(text: str) -> None:
    p = Path("docs/STATUS.md")
    t = p.read_text(encoding="utf-8")
    if "## Background job log" not in t:
        t = t.rstrip("\n") + "\n\n## Background job log\n\n"
    p.write_text(t.rstrip("\n") + f"\n- {date.today().isoformat()} {time.strftime('%H:%M')} {text}\n", encoding="utf-8", newline="")


def busy() -> bool:
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "if (Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object { $_.CommandLine -match 'extract_video|extract_images|extract_audio|cost_record|step_time' }) { exit 0 } else { exit 1 }"])
    return r.returncode == 0


def cost_done() -> bool:
    f = RESULTS / "cost_record.json"
    return f.is_file() and all(t in json.loads(f.read_text()) for t in ("T-V", "T-I", "T-A"))


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + 24 * 3600
    while (not cost_done() or busy()) and time.time() < deadline:
        time.sleep(30)
    fails = []
    for task in ("T-V", "T-I", "T-A"):
        if (RESULTS / task / "reference_scores.npz").is_file():
            continue
        with open(LOG, "a") as fh:
            fh.write(f"\n$ train_reference --task {task}\n")
            fh.flush()
            rc = subprocess.run([sys.executable, "-m", "dfd.sweep.train_reference", "--task", task], stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc:
            fails.append(task)
            status_line(f"From-scratch reference for {task} FAILED (return code {rc}); see data/_logs/references_queue.log.")
        else:
            r = json.loads((RESULTS / task / "reference.json").read_text())
            aucs = [round(s["devx_auc"], 4) for s in r["per_seed"]]
            status_line(f"From-scratch reference for {task} trained (10 epochs, 3 seeds, excluded from the selection); dev_x AUC per seed {aucs}; the ranking chain will include it.")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
