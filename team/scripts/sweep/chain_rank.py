"""Run the ranking and the report of each task automatically once that task's extraction has finished.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/sweep/chain_rank.py

A task is ready when every feature file and sidecar it expects exists (dfd.sweep.tasks.is_ready; the sidecar is written last).
Ready tasks are ranked by dfd.sweep.rank and written into docs/EXPERIMENTS.md by dfd.sweep.report, one after the other, and one
line is added to docs/STATUS.md per finished task. Nothing is chosen by hand. The state is kept in data/_results/_chain_state.json,
so a rerun resumes. The loop waits up to 48 hours for tasks whose extraction has not been built or has not finished yet.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

from dfd.sweep.tasks import RESULTS, TASKS, is_ready

STATE = RESULTS / "_chain_state.json"
LOG = Path("data/_logs/chain_rank.log")


def status_line(text: str) -> None:
    p = Path("docs/STATUS.md")
    t = p.read_text(encoding="utf-8")
    if "## Background job log" not in t:
        t = t.rstrip("\n") + "\n\n## Background job log\n\n"
    p.write_text(t.rstrip("\n") + f"\n- {date.today().isoformat()} {time.strftime('%H:%M')} {text}\n", encoding="utf-8", newline="")


def run(args: list[str]) -> int:
    with open(LOG, "a") as fh:
        fh.write(f"\n$ {' '.join(args)}\n")
        fh.flush()
        return subprocess.run([sys.executable, "-m", *args], stdout=fh, stderr=subprocess.STDOUT).returncode


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.is_file() else {}
    deadline = time.time() + 48 * 3600
    while time.time() < deadline and any(t not in state for t in TASKS):
        for tid, task in TASKS.items():
            if tid in state or not is_ready(task):
                continue
            rc = run(["dfd.sweep.rank", "--task", tid])
            if rc == 0:
                rc = run(["dfd.sweep.report", "--task", tid])
            if rc == 0:
                res = json.loads((RESULTS / tid / "rank.json").read_text())
                state[tid] = {"finished": time.strftime("%Y-%m-%d %H:%M"), "winner": res["winner"], "leader": res["leader"], "verdict": res["criteria"]["verdict"]}
                status_line(f"{task.experiment} ({tid}) ranked and written into EXPERIMENTS.md by script: leader {res['leader']}, tie set {res['tie_set']}, "
                            f"winner under the D-036 rule {res['winner']}, mechanical verdict {res['criteria']['verdict']}; results in data/_results/{tid}/.")
            else:
                state[tid] = {"failed": time.strftime("%Y-%m-%d %H:%M"), "returncode": rc}
                status_line(f"{task.experiment} ({tid}) ranking FAILED (return code {rc}); see data/_logs/chain_rank.log. Rerun after fixing; delete its entry in data/_results/_chain_state.json.")
            STATE.write_text(json.dumps(state, indent=1))
        time.sleep(60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
