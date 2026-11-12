"""E-017 (I-1c): seeds 1 and 2 of IMG-07 and IMG-08 (learning rate 1e-5, E-005 recipe unchanged), the members missing for a 3-seed ensemble.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/i1b/run_i1c.py

Seed 0 of both and the three IMG-17 seeds are the E-005 runs (data/_results/I-1). Output data/_results/I-1c, weights checkpoints/I-1/. Resumable.
"""
from __future__ import annotations

import time

from dfd.train import finetune as FT

OUT = "data/_results/I-1c"


def main():
    for cand in ("IMG-08", "IMG-07"):
        for s in (1, 2):
            print(time.strftime("%H:%M:%S"), cand, "seed", s, flush=True)
            FT.run(cand, 1e-5, s, OUT, workers=10)
    open(OUT + "/DONE", "w").write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


if __name__ == "__main__":
    main()
