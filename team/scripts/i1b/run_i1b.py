"""E-016 (I-1b): IMG-17 fine-tuned with perturbation augmentation and mixup, learning rate 1e-5, seeds 0, 1, 2.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/i1b/run_i1b.py

Recipe of E-005 unchanged apart from dfd.train.finetune.run(aug=True, mixup=0.2). Output data/_results/I-1b/IMG-17_lr1e-05_s<seed>_aug,
weights checkpoints/I-1b/ (not in git). Resumable.
"""
from __future__ import annotations

import time

from dfd.train import finetune as FT

OUT = "data/_results/I-1b"


def main():
    for s in (0, 1, 2):
        print(time.strftime("%H:%M:%S"), "IMG-17 aug seed", s, flush=True)
        FT.run("IMG-17", 1e-5, s, OUT, workers=10, aug=True, mixup=0.2, variant="aug", wdir_root="checkpoints/I-1b")
    open(OUT + "/DONE", "w").write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


if __name__ == "__main__":
    main()
