"""Feature extraction for E-020 from canonical clips (canonical-v1, produced by scripts/canonical/canonicalise.py).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/genvideo/extract_features.py deepaction
    PYTHONPATH=src .venv/Scripts/python.exe scripts/genvideo/extract_features.py df26      # only after the E-020 model is frozen (evaluation-only dataset)

Per clip: the 21 ReStraV-style trajectory values and the mean CLS embedding (frozen DINOv2-small), the 8 CLS embeddings, and 7 cheap non-semantic
pixel statistics (control baseline of E-020). Labels are read from the folder names (DeepAction) or metadata/videos.csv (DF26) only to write the
table; no label is used by this script's computations.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from dfd.platform.genvideo import canonical
from dfd.platform.genvideo.features import DinoRunner

ROOT = Path(__file__).resolve().parents[2]
CANON = ROOT / "data" / "_canonical_v3"      # protocol canonical-v3 (E-019 v3 addendum); never the v1 or v2 copies
OUT = ROOT / "data" / "_results" / "genvideo"
STAT_NAMES = ["fdiff_mean", "fdiff_std", "lapvar_mean", "lapvar_std", "luma_mean", "sat_mean", "luma_std"]


def pixel_stats(fr: np.ndarray) -> np.ndarray:
    g = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in fr]).astype(np.float32)
    fd = np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))
    lv = np.array([cv2.Laplacian(x, cv2.CV_32F).var() for x in g])
    sat = np.mean([cv2.cvtColor(f, cv2.COLOR_RGB2HSV)[..., 1].mean() for f in fr])
    return np.array([fd.mean(), fd.std(), lv.mean(), lv.std(), g.mean(), sat, g.std(axis=(1, 2)).mean()], dtype=np.float32)


def deepaction_items():
    for p in sorted((CANON / "deepaction").rglob("*.mp4")):
        rel = p.relative_to(CANON / "deepaction").parts
        src, prompt = rel[0], rel[1]
        yield {"path": str(p), "id": "/".join(rel), "label": 0 if src == "Pexels" else 1, "generator": "real" if src == "Pexels" else src,
               "group": prompt}


def df26_items():
    with open(ROOT / "data" / "df26" / "metadata" / "videos.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            p = CANON / "df26" / r["video_path"]
            if p.is_file():
                grp = r["paired_real_path"] or r["video_path"]
                yield {"path": str(p), "id": r["video_path"], "label": 0 if r["label"] == "real" else 1, "generator": r["generator"], "group": grp,
                       "access": r["generator_access"]}


def main(argv) -> int:
    name = argv[0]
    items = list(deepaction_items() if name == "deepaction" else df26_items())
    OUT.mkdir(parents=True, exist_ok=True)
    runner = DinoRunner()
    feats, cls_mean, cls_all, stats, meta, bad = [], [], [], [], [], []
    t0 = time.time()
    for k, it in enumerate(items, 1):
        try:
            fr = canonical.read_frames(it["path"])
            f = runner.features(fr)
            feats.append(f["restrav21"]); cls_mean.append(f["cls_mean"]); cls_all.append(f["cls_all"]); stats.append(pixel_stats(fr))
            meta.append(it)
        except Exception as e:  # noqa: BLE001 - recorded, not hidden
            bad.append({"id": it["id"], "error": f"{type(e).__name__}: {str(e)[:120]}"})
        if k % 200 == 0:
            print(f"{k}/{len(items)} {round(time.time() - t0)} s", flush=True)
    np.savez_compressed(OUT / f"features_{name}.npz", restrav21=np.stack(feats), cls_mean=np.stack(cls_mean), cls_all=np.stack(cls_all),
                        stats=np.stack(stats), stat_names=np.array(STAT_NAMES), label=np.array([m["label"] for m in meta]),
                        generator=np.array([m["generator"] for m in meta]), group=np.array([m["group"] for m in meta]),
                        id=np.array([m["id"] for m in meta]), access=np.array([m.get("access", "") for m in meta]),
                        dino_revision=np.array([runner.revision]))
    (OUT / f"features_{name}_failures.json").write_text(__import__("json").dumps(bad, indent=1), encoding="utf-8")
    print(f"done {len(meta)} clips, {len(bad)} failures, {round(time.time() - t0)} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
