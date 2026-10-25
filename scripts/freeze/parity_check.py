"""D-044 parity check for a version 2 checkpoint (to be run by the owner of the GPU; not run by the integration work).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/freeze/parity_check.py [--checkpoint checkpoints/I-5-freeze-candidate] [--n 3]

The same item is scored (a) from stored training-time results and the checkpoint's own temperature, and (b) by the real adapter through the inference
library from the raw file. Cases (validation material only, no test and no dev_x item):
  T-V  FF++ validation videos, a fake and a real one (n each): stored = the E-005 clip score of the shipped run (data/_results/I-1/IMG-17_lr1e-05_s0/scores.npz).
  T-A  ASVspoof 2019 LA development files (n of them, unsaturated ensemble scores preferred so that 1.0 is not compared with 1.0): stored = logit mean of the
       three seed 0 heads on the stored E-007 features (data/_features_trimnorm), then the checkpoint temperature.
  T-I  two ProGAN training images in which no face is found, as in the I-0 check, through the unchanged T-I head (stored features of data/_features).
Tolerance on the calibrated score: 0.02 absolute (fp16 encoder, JPEG round trip of the crop, resize rules; D-044). Writes
data/_results/freeze/parity.json. Exit status 1 on a failure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dfd.data import loaders as L
from dfd.eval import calibration as C
from dfd.infer.checkpoint import load_checkpoint
from dfd.infer.factory import make_adapter
from dfd.infer.library import analyse_file
from dfd.sweep import extract_audio, extract_images, extract_video
from dfd.sweep.tasks import TASKS, feature_path

TOL = 0.02
FT_RUN = "IMG-17_lr1e-05_s0"
KEY = {"T-V": "visual", "T-I": "image", "T-A": "audio"}


def cal(p, T):
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(C.sigmoid(C.logit(p) / T))


def stored_tv(meta, sample_id):
    z = np.load(f"data/_results/I-1/{FT_RUN}/scores.npz", allow_pickle=True)
    i = int(np.nonzero(z["val_sample_id"].astype(str) == sample_id)[0][0])
    return cal(float(z["val_scores"][i]), float(meta["tasks"]["T-V"]["calibration"]["temperature"]))


def ensemble_logits(heads, sample_ids=None):
    """Logit mean of the T-A members on the stored E-007 development features; returns (logits, sample_id, label)."""
    zs = []
    for m in heads["T-A"].members:
        f = np.load(f"data/_features_trimnorm/{m.cid}/T-A/asvspoof_dev.npz", allow_pickle=False)
        zs.append(m.head.logit_rows(f["feats"].astype(np.float32)))
        sid, lab = f["sample_id"].astype(str), f["label"]
    return np.mean(zs, axis=0), sid, lab


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/I-5-freeze-candidate")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args(argv)
    meta, heads = load_checkpoint(a.checkpoint)
    adapter = make_adapter("real", a.checkpoint)
    cases = []
    # T-V: validation clips of FF++ (the shipped fine-tuned model)
    j = extract_video.JOBS[1]
    recs = L.select_records(j[1], j[2], j[4], modality="video")
    for lab in (1, 0):
        for r in [r for r in recs if r.label == lab][5:5 + a.n]:
            cases.append(("T-V", r, stored_tv(meta, r.sample_id)))
    # T-A: development files of ASVspoof 2019 LA
    if "T-A" in heads:
        z, sid, lab = ensemble_logits(heads)
        T = float(meta["tasks"]["T-A"]["calibration"]["temperature"])
        pc = C.sigmoid(z / T)
        j = extract_audio.JOBS[1]
        by_id = {r.sample_id: r for r in L.select_records(j[1], j[2], j[3], modality="audio")}
        mid = np.nonzero((pc > 0.05) & (pc < 0.95))[0]
        for k in mid[:a.n]:
            cases.append(("T-A", by_id[str(sid[k])], float(pc[k])))
    # T-I: unchanged head, two unsaturated training images (as in the I-0 check)
    if "T-I" in heads:
        j = extract_images.JOBS[0]
        zf = np.load(feature_path("IMG-13", TASKS["T-I"], "progan_train"), allow_pickle=False)
        T = float(meta["tasks"]["T-I"]["calibration"]["temperature"])
        p_all = C.sigmoid(C.logit(heads["T-I"].prob_rows(zf["feats"].astype(np.float32))) / T)
        by_id = {r.sample_id: r for r in L.select_records(j[1], j[2], j[3], modality="image")}
        for k in np.nonzero((p_all > 0.05) & (p_all < 0.95))[0][:2]:
            cases.append(("T-I", by_id[str(zf["sample_id"][k])], float(p_all[k])))
    rows, ok_all = [], True
    for tid, r, want in cases:
        path = Path("data") / r.path
        resp = analyse_file(path, path.name, adapter)
        per = (resp["result"] or {}).get("per_modality", {})
        got = per.get(KEY[tid], {}).get("score")
        diff = None if got is None else abs(got - want)
        ok = got is not None and diff <= TOL and not resp["errors"] and resp["inference_source"] == "real"
        ok_all &= ok
        rows.append(dict(task=tid, sample_id=r.sample_id, label=r.label, stored_score=round(want, 6), adapter_score=got, abs_diff=None if diff is None else round(diff, 6),
                         path=per.get(KEY[tid], {}).get("path"), n_units=per.get(KEY[tid], {}).get("n_units"), inference_source=resp["inference_source"],
                         state=resp["status"]["state"], passed=bool(ok)))
        print(json.dumps(rows[-1]), flush=True)
    out = Path("data/_results/freeze")
    out.mkdir(parents=True, exist_ok=True)
    (out / "parity.json").write_text(json.dumps(dict(checkpoint=meta["checkpoint_id"], tolerance=TOL, passed=bool(ok_all), cases=rows), indent=1))
    print("PARITY", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
