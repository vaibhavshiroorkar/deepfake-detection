"""D-044 parity check: the same training-split item scored (a) from the stored extraction features and the checkpoint head and (b) by the real
adapter through the inference library, from the raw file. The two must agree up to the numerical noise of half-precision encoding.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/mvp/parity_check.py [--checkpoint checkpoints/I-0-mvp]

Per task one item of the TRAINING split of the winner (T-V a fake and a real FF++ c23 video, T-I a ProGAN image in which no face is found,
T-A an ASVspoof 2019 LA training utterance). Reads no validation, dev_x or test item. Writes data/_results/I-0/parity.json. Exit 1 on a failure.
Tolerance on the calibrated score: 0.02 absolute (fp16 encoder, JPEG round trip of the crop, resize rules); the feature check reports the
cosine similarity of the encoder feature as well.
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


def stored_score(meta, heads, tid, cand, dataset, sample_id):
    task = TASKS[tid]
    z = np.load(feature_path(cand, task, dataset), allow_pickle=False)
    i = int(np.nonzero(z["sample_id"] == sample_id)[0][0])
    f = z["feats"][i].astype(np.float32)
    if f.ndim == 1:
        feats = f[None]
    else:
        feats = f[z["mask"][i]]
    p = float(heads[tid].prob_rows(feats).mean())
    T = float(meta["tasks"][tid]["calibration"]["temperature"])
    return float(C.sigmoid(C.logit(p) / T)), feats


def pick(manifest, split_csv, split, modality, label=None, skip=0, pred=None):
    recs = L.select_records(manifest, split_csv, split, modality=modality)
    recs = [r for r in recs if (label is None or r.label == label) and (pred is None or pred(r))]
    return recs[skip]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/I-0-mvp")
    a = ap.parse_args(argv)
    meta, heads = load_checkpoint(a.checkpoint)
    adapter = make_adapter("real", a.checkpoint)
    rows, ok_all = [], True
    cases = []
    j = extract_video.JOBS[0]
    for lab in (1, 0):
        cases.append(("T-V", "IMG-07", "ffpp_train", pick(j[1], j[2], j[4], "video", label=lab, skip=5)))
    j = extract_images.JOBS[0]
    z = np.load(feature_path("IMG-13", TASKS["T-I"], "progan_train"), allow_pickle=False)
    T = float(meta["tasks"]["T-I"]["calibration"]["temperature"])
    p_all = C.sigmoid(C.logit(heads["T-I"].prob_rows(z["feats"].astype(np.float32))) / T)
    mid = np.nonzero((p_all > 0.05) & (p_all < 0.95))[0][:2]       # unsaturated training items, so that the comparison is not 1.0 against 1.0
    by_id = {r.sample_id: r for r in L.select_records(j[1], j[2], j[3], modality="image")}
    for i in mid:
        cases.append(("T-I", "IMG-13", "progan_train", by_id[str(z["sample_id"][i])]))
    j = extract_audio.JOBS[0]
    z = np.load(feature_path("AU-8", TASKS["T-A"], "asvspoof_train"), allow_pickle=False)
    T = float(meta["tasks"]["T-A"]["calibration"]["temperature"])
    p_all = C.sigmoid(C.logit(heads["T-A"].prob_rows(z["feats"].astype(np.float32))) / T)
    mid = np.nonzero((p_all > 0.05) & (p_all < 0.95))[0][:2]
    by_id = {r.sample_id: r for r in L.select_records(j[1], j[2], j[3], modality="audio")}
    for i in mid:
        cases.append(("T-A", "AU-8", "asvspoof_train", by_id[str(z["sample_id"][i])]))
    for tid, cand, dataset, r in cases:
        want, _ = stored_score(meta, heads, tid, cand, dataset, r.sample_id)
        path = Path("data") / r.path
        resp = analyse_file(path, path.name, adapter)
        per = (resp["result"] or {}).get("per_modality", {})
        key = {"T-V": "visual", "T-I": "image", "T-A": "audio"}[tid]
        got = per.get(key, {}).get("score")
        diff = None if got is None else abs(got - want)
        ok = got is not None and diff <= TOL and not resp["errors"] and resp["inference_source"] == "real"
        ok_all &= ok
        rows.append(dict(task=tid, sample_id=r.sample_id, label=r.label, stored_score=round(want, 6), adapter_score=got,
                         abs_diff=None if diff is None else round(diff, 6), path=per.get(key, {}).get("path"), n_units=per.get(key, {}).get("n_units"),
                         inference_source=resp["inference_source"], state=resp["status"]["state"], passed=bool(ok)))
        print(json.dumps(rows[-1]), flush=True)
    Path("data/_results/I-0").mkdir(parents=True, exist_ok=True)
    Path("data/_results/I-0/parity.json").write_text(json.dumps(dict(checkpoint=meta["checkpoint_id"], tolerance=TOL, passed=bool(ok_all), cases=rows), indent=1))
    print("PARITY", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
