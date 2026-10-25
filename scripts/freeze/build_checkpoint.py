"""Build a version 2 checkpoint of the frozen system so far (CPU only, no GPU, no git command).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/freeze/build_checkpoint.py [--out checkpoints/I-5-freeze-candidate] [--iteration I-5]

Tasks and sources (all seed 0, fixed before the build; no choice among seeds is made with dev_x):
  T-V  the fine-tuned IMG-17 of E-005 (learning rate 1e-5, seed 0): checkpoints/I-1/IMG-17_lr1e-05_s0/weights.pt. Temperature and threshold are fitted on
       the FF++ validation clip scores of that run (data/_results/I-1/IMG-17_lr1e-05_s0/scores.npz, mean of the frame probabilities, D-024).
  T-I  unchanged from the I-0 checkpoint (IMG-13 frozen probe): its head file and its metadata block are copied from checkpoints/I-0-mvp.
  T-A  the logit mean of AU-8, AU-4 and AU-6 with the seed 0 heads of E-009 (data/_results/I-4/head_<cand>_s0.npz, E-007 window, E-008 augmentation in
       training). Temperature and threshold are fitted on the ASVspoof 2019 LA development clip scores of the ensemble (features data/_features_trimnorm).
Only training and validation material is read. Writes the folder with save_checkpoint_v2 (SHA-256 of every file in the metadata).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dfd.eval import calibration as C
from dfd.eval.threshold import choose_threshold_max_balanced_accuracy
from dfd.infer.checkpoint import _sha256, save_checkpoint_v2
from dfd.prep import config as P

SEED = 0
I0 = Path("checkpoints/I-0-mvp")
FT_RUN = "IMG-17_lr1e-05_s0"
MEMBERS = ["AU-8", "AU-4", "AU-6"]


def git_head() -> str:
    """Short commit id read from .git without running git."""
    g = Path(".git")
    head = (g / "HEAD").read_text().strip()
    if head.startswith("ref:"):
        ref = head.split(" ", 1)[1]
        f = g / ref
        if f.is_file():
            return f.read_text().strip()[:7]
        for line in (g / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0][:7]
    return head[:7]


def calibrate(logits: np.ndarray, y: np.ndarray) -> tuple[float, float, dict]:
    """Temperature on logits and the maximum-balanced-accuracy threshold on the calibrated probabilities (D-024)."""
    ok = np.isfinite(logits)
    T = float(C.fit_temperature(logits[ok], y[ok].astype(int)).temperature)
    pc = C.sigmoid(logits[ok] / T)
    thr = float(choose_threshold_max_balanced_accuracy(y[ok].astype(int), pc).threshold)
    return T, thr, dict(n=int(ok.sum()), n_excluded=int((~ok).sum()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints/I-5-freeze-candidate")
    ap.add_argument("--iteration", default="I-5")
    a = ap.parse_args(argv)
    i0 = json.loads((I0 / "metadata.json").read_text())
    tasks, art, sources = {}, {}, {}

    # ---- T-V: fine-tuned encoder
    side = json.loads(Path("data/_features/IMG-17/T-V/ffpp_val.json").read_text())
    z = np.load(f"data/_results/I-1/{FT_RUN}/scores.npz", allow_pickle=True)
    p = np.clip(z["val_scores"].astype(np.float64), 1e-6, 1 - 1e-6)
    T_v, thr_v, nv = calibrate(np.log(p / (1 - p)), z["val_y"])
    wsrc = Path("checkpoints/I-1") / FT_RUN / "weights.pt"
    art["weights_T-V.pt"] = wsrc
    sources["T-V"] = {"weights": str(wsrc), "weights_sha256": _sha256(wsrc), "scores": f"data/_results/I-1/{FT_RUN}/scores.npz"}
    tasks["T-V"] = dict(
        encoder=dict(cid="IMG-17", name=side["encoder"], revision=side["revision"], dim=int(side["dim"]), input_size=int(side["input_size"])),
        preprocess_id=P.IN_USE["T-V"](), preprocess_config=P.video_config(), model={"kind": "finetuned_encoder", "weights_file": "weights_T-V.pt"},
        calibration=dict(method="temperature scaling, one scalar T, p = sigmoid(logit(p) / T)", temperature=T_v,
                         fitted_on=f"FF++ validation split, clip score of the fine-tuned model, seed {SEED} ({nv['n']} clips)"),
        threshold=dict(value=thr_v, how="maximum balanced accuracy on the FF++ validation split, on calibrated probabilities (D-024)"),
        train=dict(dataset="ffpp_train", validation="ffpp_val", evidence="E-005", recipe="full fine-tuning, AdamW 1e-5, wd 0.05, 1,800 steps, batch 32, class-balanced BCE (E-005)",
                   seed=SEED, manifests=i0["tasks"]["T-V"]["train"]["manifests"], splits=i0["tasks"]["T-V"]["train"]["splits"]))

    # ---- T-I: copied from I-0
    art["head_T-I.npz"] = I0 / i0["tasks"]["T-I"]["head_file"]
    sources["T-I"] = {"head": str(art["head_T-I.npz"]), "head_sha256": _sha256(art["head_T-I.npz"]), "checkpoint": i0["checkpoint_id"]}
    ti = json.loads(json.dumps(i0["tasks"]["T-I"]))
    ti.pop("head_sha256", None)
    ti["head_file"] = "head_T-I.npz"
    ti["model"] = {"kind": "linear_probe"}
    ti["train"]["evidence"] = "E-002, E-004 (unchanged from I-0)"
    tasks["T-I"] = ti

    # ---- T-A: ensemble
    members, zs, y = [], [], None
    for c in MEMBERS:
        side = json.loads(Path(f"data/_features_trimnorm/{c}/T-A/asvspoof_dev.json").read_text())
        h = Path("data/_results/I-4") / f"head_{c}_s{SEED}.npz"
        art[f"head_T-A_{c}.npz"] = h
        sources[f"T-A {c}"] = {"head": str(h), "head_sha256": _sha256(h)}
        members.append(dict(cid=c, name=side["encoder"], revision=side["revision"], dim=int(side["dim"]), input_size=int(side["input_size"]), head_file=f"head_T-A_{c}.npz"))
        f = np.load(f"data/_features_trimnorm/{c}/T-A/asvspoof_dev.npz", allow_pickle=False)
        hz = np.load(h)
        zs.append(((f["feats"].astype(np.float32) - hz["mean"]) / hz["std"]) @ hz["w"] + float(hz["b"]))
        if y is None:
            y, sid = f["label"].astype(int), f["sample_id"]
        else:
            assert np.array_equal(sid, f["sample_id"])
    T_a, thr_a, na = calibrate(np.mean(zs, axis=0), y)
    tasks["T-A"] = dict(
        encoder=dict(cid="+".join(MEMBERS), name="logit mean of " + ", ".join(m["name"] for m in members), revision="see model.members", dim=int(members[0]["dim"]), input_size=members[0]["input_size"]),
        preprocess_id=P.IN_USE["T-A"](), preprocess_config=P.audio_config(),
        model={"kind": "ensemble_logit_mean", "members": members},
        calibration=dict(method="temperature scaling, one scalar T, p = sigmoid(logit(p) / T)", temperature=T_a,
                         fitted_on=f"ASVspoof 2019 LA development split, ensemble clip score, seed {SEED} ({na['n']} clips)"),
        threshold=dict(value=thr_a, how="maximum balanced accuracy on the ASVspoof 2019 LA development split, on calibrated probabilities (D-024)"),
        train=dict(dataset="asvspoof_train", validation="asvspoof_dev", evidence="E-007, E-008, E-009", head="linear, E-009 recipe on the E-007 window with three augmented copies (E-008)",
                   seed=SEED, manifests=["asvspoof2019_la.manifest.csv"], splits=i0["tasks"]["T-A"]["train"]["splits"]))

    meta = dict(checkpoint_id=f"{Path(a.out).name}-" + time.strftime("%Y%m%d-%H%M"), iteration=a.iteration, git_commit=git_head(),
                protocol="docs/PROTOCOL.md; D-024, D-028, D-044, D-054, D-055, D-056", created=time.strftime("%Y-%m-%d %H:%M:%S"), seed=SEED,
                library_versions=dict(torch=torch.__version__, numpy=np.__version__),
                manifest_ids=sorted({m for t in tasks.values() for m in t["train"]["manifests"]}),
                split_ids=sorted({s for t in tasks.values() for s in t["train"]["splits"]}), tasks=tasks, fusion=None,
                evidence=dict(experiments=["E-005", "E-007", "E-008", "E-009"], note="dev_x verdicts are exploratory (D-041); no test split was read to build this checkpoint"),
                sources=sources)
    path = save_checkpoint_v2(a.out, meta, art)
    print("checkpoint", path, "T-V T", round(T_v, 3), "thr", round(thr_v, 3), "| T-A T", round(T_a, 3), "thr", round(thr_a, 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
