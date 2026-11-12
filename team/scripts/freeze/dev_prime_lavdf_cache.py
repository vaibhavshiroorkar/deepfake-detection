"""Test helper (dry-run root only): fill data/_final_dry/ftcache for the LAV-DF train and val splits from the score files that E-010, E-011 and E-012
already wrote, so that the LAV-DF assembly code (temperatures, logistic regression, the label schemes) can be tested on the CPU.

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/freeze/dev_prime_lavdf_cache.py

Never writes to data/_final and never touches a sealed split (those files hold train and val only).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import common as K

R = Path("data/_results/I-6")
ROOT = K.DRY_ROOT / "ftcache"


def put(model, split, sample_id, p):
    z = np.load(R / f"scores_{split}.npz", allow_pickle=True)
    assert np.array_equal(z["sample_id"], sample_id)
    K.save_npz(ROOT / model / f"lavdf_{split}.npz", sample_id=sample_id, label=z["y_fused"], group_id=z["group"], p=np.asarray(p, float), n_listed=len(sample_id))


for s in K.SEEDS:
    for split in ("train", "val"):
        z = np.load(R / f"scores_{split}.npz", allow_pickle=True)
        put(f"IMG-17_lr1e-05_s{s}", split, z["sample_id"], z["vis_p"][s])
        f = R / f"visual_lavdf_s{s}.npz"
        if f.is_file():
            put(f"IMG-17_lavdf_s{s}", split, z["sample_id"], np.load(f, allow_pickle=True)[f"{split}_p"])
        f = R / f"visual_lavdf_produced_s{s}.npz"
        if f.is_file():
            put(f"IMG-17_lavdf_produced_s{s}", split, z["sample_id"], np.load(f, allow_pickle=True)[f"{split}_p"])
print("primed", sorted(p.name for p in ROOT.iterdir()))
