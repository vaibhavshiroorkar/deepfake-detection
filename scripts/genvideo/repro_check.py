"""Reproducibility check of the live generated_video path against the E-020 experiment path (an integrity check, not an accuracy result).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/genvideo/repro_check.py

For N clips of each corpus (fixed seed): canonicalise the ORIGINAL file with the module's own function, compare the decoded frames with the stored
canonical-v3 clip used in the experiment, run the module's feature path, and compare the mean-CLS feature and the head_C2 score with the stored
feature table (features_*.npz). Nothing is fitted; DF26 clips are only read for this consistency check.
"""
import json
import random
import tempfile
from pathlib import Path

import numpy as np

from dfd.platform.genvideo import canonical
from dfd.platform.genvideo.features import shared_runner
from dfd.platform.genvideo.model import load_head

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "data" / "_results" / "genvideo"
CANON = ROOT / "data" / "_canonical_v3"
N = 8


def main():
    head = load_head(ROOT / "checkpoints" / "genvideo-E020" / "head.npz")
    frozen = load_head(ROOT / "checkpoints" / "genvideo-E020" / "head_C2.npz")
    runner = shared_runner()
    rows = []
    for name, src_root in (("deepaction", ROOT / "data" / "deepaction_v1"), ("df26", ROOT / "data" / "df26")):
        z = np.load(RES / f"features_{name}.npz", allow_pickle=False)
        ids = [str(i) for i in z["id"]]
        rng = random.Random(20261026)
        for i in rng.sample(range(len(ids)), N):
            rel = ids[i]
            src = src_root / rel
            src = src if src.is_file() else None
            if src is None:
                rows.append({"corpus": name, "id": rel, "error": "source file not located"})
                continue
            with tempfile.TemporaryDirectory() as td:
                dst = Path(td) / "c.mp4"
                canonical.canonicalise(src, dst)
                fr = canonical.read_frames(dst)
            stored_fr = canonical.read_frames(CANON / name / rel)
            feats = runner.features(fr)
            s_live = float(frozen.score(feats["cls_mean"])[0])
            s_installed = float(head.score(feats["cls_mean"])[0])
            s_stored = float(frozen.score(z["cls_mean"][i].astype(np.float64))[0])
            rows.append({"corpus": name, "id": rel, "frames_identical": bool(np.array_equal(fr, stored_fr)),
                         "max_abs_pixel_diff": int(np.abs(fr.astype(int) - stored_fr.astype(int)).max()),
                         "cls_max_abs_diff": float(np.abs(feats["cls_mean"] - z["cls_mean"][i]).max()),
                         "score_live": round(s_live, 6), "score_stored": round(s_stored, 6), "score_installed_head": round(s_installed, 6),
                         "abs_score_diff": round(abs(s_live - s_stored), 6)})
    ok = [r for r in rows if "error" not in r]
    summary = {"n": len(rows), "located": len(ok), "frames_identical": sum(r["frames_identical"] for r in ok),
               "max_abs_score_diff": max((r["abs_score_diff"] for r in ok), default=None),
               "installed_equals_frozen": all(r["score_live"] == r["score_installed_head"] for r in ok)}
    (RES / "repro_check.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary), *(r.get("error", "") for r in rows if "error" in r))


if __name__ == "__main__":
    main()
