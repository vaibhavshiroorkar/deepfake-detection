"""Wait until all seven image encoder weights are downloaded and hash-verified, then start two jobs side by side:
the LAV-DF download/verify job (network and disk) and the E-001 feature queue (GPU). Writes one line to
docs/STATUS.md when the E-001 queue finishes (the LAV-DF job writes its own).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/sweep/after_weights.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

LOG = Path("data/_logs/encoder_weights.json")
NEED = ["IMG-01", "IMG-04", "IMG-07", "IMG-08", "IMG-13", "IMG-15", "IMG-17"]


def status_line(text):
    p = Path("docs/STATUS.md")
    t = p.read_text(encoding="utf-8")
    if "## Background job log" not in t:
        t = t.rstrip("\n") + "\n\n## Background job log\n\n"
    p.write_text(t.rstrip("\n") + f"\n- {date.today().isoformat()} {time.strftime('%H:%M')} {text}\n", encoding="utf-8", newline="")


def main():
    deadline = time.time() + 8 * 3600
    while time.time() < deadline:
        recs = {r["id"]: r for r in json.loads(LOG.read_text())} if LOG.is_file() else {}
        bad = [i for i in NEED if i in recs and not recs[i]["match"]]
        if bad:
            status_line(f"Encoder weights: hash MISMATCH for {bad}; LAV-DF download and E-001 NOT started.")
            return 1
        if all(i in recs for i in NEED):
            break
        time.sleep(30)
    else:
        status_line("Encoder weights not verified within 8 hours; LAV-DF download and E-001 NOT started.")
        return 1
    status_line("Seven image encoder weights downloaded, all SHA-256 verified; starting the LAV-DF job and the E-001 queue.")
    lav = subprocess.Popen([sys.executable, "scripts/data/lavdf_job.py"], stdout=open("data/_logs/lavdf_job.log", "a"), stderr=subprocess.STDOUT)
    r = subprocess.run([sys.executable, "scripts/sweep/run_e001.py"], stdout=open("data/_logs/e001_queue.log", "a"), stderr=subprocess.STDOUT)
    feats = sorted(Path("data/_features").glob("IMG-*/T-V/*.json"))
    cands = sorted({f.parts[-3] for f in feats})
    status_line(f"E-001 queue finished (exit {r.returncode}): {len(feats)} feature files for {len(cands)} of 7 candidates {cands}; log data/_logs/e001_queue.log. Next: the linear heads and the E-001 result entry.")
    lav.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
