"""E-016 manipulation check: realised rates of the augmentation on 2,000 training frames (CPU, no model).

    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src .venv/Scripts/python.exe scripts/i1b/check_aug.py [--save-grid out.png]

Expected rates 0.3 (blur), 0.5 (resize), 0.5 (JPEG), each within 0.03. Prints the mean absolute pixel change of the augmented frames.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from dfd.train import finetune as FT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--save-grid", default=None)
    a = ap.parse_args()
    tr = FT.clip_table("train", None)
    n = len(tr["paths"])
    idx = np.random.default_rng(0).choice(n, a.n, replace=False)
    cnt = dict(blur=0, resize=0, jpeg=0, any=0)
    dif, grid = [], []
    for j, i in enumerate(idx):
        img = cv2.imread(str(tr["paths"][i]), cv2.IMREAD_COLOR)
        if img.shape[:2] != (224, 224):
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        rng = np.random.default_rng([0, int(i)])
        out, ev = FT.augment_frame(img, rng, return_events=True)
        assert out.shape == img.shape and out.dtype == np.uint8
        for k in ("blur", "resize", "jpeg"):
            cnt[k] += int(ev[k])
        cnt["any"] += int(any(ev.values()))
        dif.append(float(np.abs(out.astype(np.int16) - img.astype(np.int16)).mean()))
        if j < 8:
            grid.append(np.concatenate([img, out], axis=0))
    r = {k: v / a.n for k, v in cnt.items()}
    print("realised rates", r, "expected blur 0.3, resize 0.5, jpeg 0.5")
    print("mean abs pixel change, all frames %.3f, augmented frames %.3f" % (np.mean(dif), np.mean([d for d in dif if d > 0]) if any(d > 0 for d in dif) else 0))
    ok = abs(r["blur"] - 0.3) <= 0.03 and abs(r["resize"] - 0.5) <= 0.03 and abs(r["jpeg"] - 0.5) <= 0.03
    print("MANIPULATION CHECK", "PASSED" if ok else "FAILED")
    if a.save_grid:
        cv2.imwrite(a.save_grid, np.concatenate(grid, axis=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
