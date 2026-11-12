"""I-0 MVP (D-040): linear heads of the three sweep winners, temperature and threshold on validation only, the checkpoint folder.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/mvp/train_mvp.py [--out checkpoints/I-0-mvp]

Winners (D-049): T-V IMG-07 (DINOv2 small), T-I IMG-13 (OpenAI CLIP ViT-L/14), T-A AU-8 (HuBERT base). CPU only: the heads are trained on the stored
frozen features of the sweep (data/_features), with the head recipe of dfd.sweep.heads (class-balanced BCE, AdamW 1e-3, weight decay 1e-4, batch 1,024,
early stopping on validation AUC) and seeds 0, 1, 2. The shipped head is the seed 0 head (fixed before the run, decided autonomously: no choice among
seeds is made with dev_x). Temperature (D-024) and the decision threshold (maximum balanced accuracy, D-024) are fitted on the validation split of the
task with the clip score (mean of frame probabilities) and stored in calibrated-probability space. Only train, validation and dev_x are read.
Writes the checkpoint (src/dfd/infer/checkpoint.py, D-044) and data/_results/I-0/<task>/scores.npz and data/_results/I-0/metrics.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from dfd.eval import bootstrap as B
from dfd.eval import calibration as C
from dfd.eval import metrics as M
from dfd.eval.threshold import choose_threshold_max_balanced_accuracy
from dfd.infer.checkpoint import Head, save_checkpoint
from dfd.prep import config as P
from dfd.sweep import extract_audio, extract_images, extract_video, heads
from dfd.sweep.rank import load_split
from dfd.sweep.tasks import RESULTS, TASKS

WINNERS = {"T-V": "IMG-07", "T-I": "IMG-13", "T-A": "AU-8"}
SEEDS = (0, 1, 2)
SHIPPED_SEED = 0
OUT_RESULTS = RESULTS / "I-0"


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()


def task_files(tid: str):
    jobs = {"T-V": [(j[0], j[1], j[2]) for j in extract_video.JOBS],
            "T-I": [(j[0], j[1], j[2]) for j in extract_images.JOBS],
            "T-A": [(j[0], j[1], j[2]) for j in extract_audio.JOBS]}[tid]
    return sorted({Path(m).name for _, m, _ in jobs}), sorted({Path(s).stem for _, _, s in jobs})


def prob_clip(h, sp):
    return heads.clip_scores(h.prob_rows(sp.X), sp.clip_of, sp.y.size)


def run_task(tid: str):
    task, cand = TASKS[tid], WINNERS[tid]
    tr, _ = load_split(cand, task, task.train)
    va, vmeta = load_split(cand, task, task.val)
    dv, _ = load_split(cand, task, task.devx)
    hs = [heads.train_head(tr, va, s) for s in SEEDS]
    S = {name: np.array([prob_clip(h, sp) for h in hs]) for name, sp in (("train", tr), ("val", va), ("dev", dv))}
    ys = {"train": tr.y, "val": va.y, "dev": dv.y}
    gs = {"train": tr.groups, "val": va.groups, "dev": dv.groups}
    ok = {k: ~np.isnan(S[k]).any(0) for k in S}
    out = dict(task=tid, candidate=cand, seeds=list(SEEDS), shipped_seed=SHIPPED_SEED, n={k: int(ok[k].sum()) for k in S},
               n_excluded_no_score={k: int((~ok[k]).sum()) for k in S}, head_best_epochs=[h.best_epoch for h in hs])
    for k in S:
        b = B.cluster_bootstrap(ys[k][ok[k]], S[k][:, ok[k]], gs[k][ok[k]], "auc", n_boot=B.N_BOOT)
        out[f"{k}_auc"] = b.as_dict()
    yv, yd = ys["val"][ok["val"]], ys["dev"][ok["dev"]]
    per = []
    for i in range(len(SEEDS)):
        pv, pd = S["val"][i][ok["val"]], S["dev"][i][ok["dev"]]
        T = C.fit_temperature(C.logit(pv), yv).temperature
        pvc, pdc = C.sigmoid(C.logit(pv) / T), C.sigmoid(C.logit(pd) / T)
        thr = choose_threshold_max_balanced_accuracy(yv, pvc).threshold
        per.append(dict(seed=SEEDS[i], temperature=float(T), threshold_cal=float(thr),
                        val=dict(ece_before=M.ece(yv, pv), ece_after=M.ece(yv, pvc), brier_before=M.brier(yv, pv), brier_after=M.brier(yv, pvc),
                                 balanced_accuracy=M.balanced_accuracy(yv, pvc, thr)),
                        dev=dict(ece_before=M.ece(yd, pd), ece_after=M.ece(yd, pdc), brier_before=M.brier(yd, pd), brier_after=M.brier(yd, pdc),
                                 eer=M.eer(yd, pd), balanced_accuracy=M.balanced_accuracy(yd, pdc, thr))))
    out["per_seed"] = per
    out["mean"] = {split: {k: float(np.mean([p[split][k] for p in per])) for k in per[0][split]} for split in ("val", "dev")}
    out["temperature_mean"] = float(np.mean([p["temperature"] for p in per]))
    gap = [M.roc_auc(ys["train"][ok["train"]], S["train"][i][ok["train"]]) - M.roc_auc(yv, S["val"][i][ok["val"]]) for i in range(len(SEEDS))]
    out["gap_train_minus_val"] = float(np.mean(gap))
    out["gap_val_minus_dev"] = out["val_auc"]["estimate"] - out["dev_auc"]["estimate"]
    out["seed_sd_dev_auc"] = out["dev_auc"]["seed_sd"]
    out["dev_auc_half_width"] = (out["dev_auc"]["high"] - out["dev_auc"]["low"]) / 2
    h0 = hs[SHIPPED_SEED]
    head = Head(h0.mean.astype(np.float32), h0.std.astype(np.float32), h0.w.astype(np.float32), float(h0.b))
    np.savez(OUT_RESULTS / tid / "scores.npz", **{f"{k}_scores": S[k] for k in S}, **{f"{k}_y": ys[k] for k in S},
             **{f"{k}_group": gs[k] for k in S}, train_sample_id=tr.sample_id, val_sample_id=va.sample_id, dev_sample_id=dv.sample_id)
    return out, vmeta, per[SHIPPED_SEED], head


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints/I-0-mvp")
    a = ap.parse_args(argv)
    torch.set_num_threads(8)
    heads_out, tasks_meta, report = {}, {}, {}
    t0 = time.perf_counter()
    for tid in TASKS:
        (OUT_RESULTS / tid).mkdir(parents=True, exist_ok=True)
        rep, vm, ship, head = run_task(tid)
        report[tid] = rep
        heads_out[tid] = head
        manifests, splits = task_files(tid)
        cfg = {"T-V": P.video_config, "T-I": P.image_config, "T-A": P.audio_config}[tid]()
        tasks_meta[tid] = dict(
            encoder=dict(cid=WINNERS[tid], name=vm["encoder"], revision=vm["revision"], dim=int(vm["dim"]), input_size=int(vm["input_size"])),
            preprocess_id=P.IN_USE[tid](), preprocess_config=cfg,
            calibration=dict(method="temperature scaling, one scalar T, p = sigmoid(logit(p) / T)", temperature=ship["temperature"],
                             fitted_on=f"validation split {TASKS[tid].val}, clip score, seed {SHIPPED_SEED} head"),
            threshold=dict(value=ship["threshold_cal"], how="maximum balanced accuracy on the validation split, on calibrated probabilities (D-024)"),
            train=dict(dataset=TASKS[tid].train, validation=TASKS[tid].val, head="linear, class-balanced BCE, AdamW 1e-3, wd 1e-4, batch 1024, patience 10",
                       seed=SHIPPED_SEED, seeds_reported=list(SEEDS), n_train=rep["n"]["train"], manifests=manifests, splits=splits))
        print(tid, WINNERS[tid], "val", round(rep["val_auc"]["estimate"], 4), "dev", round(rep["dev_auc"]["estimate"], 4),
              "T", round(ship["temperature"], 3), flush=True)
    cid = "I-0-mvp-" + time.strftime("%Y%m%d-%H%M")
    meta = dict(checkpoint_id=cid, iteration="I-0", git_commit=git("rev-parse", "--short", "HEAD"),
                protocol="docs/PROTOCOL.md; D-024, D-028, D-036, D-040, D-049", created=time.strftime("%Y-%m-%d %H:%M:%S"), seed=SHIPPED_SEED,
                library_versions=dict(torch=torch.__version__, numpy=np.__version__),
                manifest_ids=sorted({m for t in tasks_meta.values() for m in t["train"]["manifests"]}),
                split_ids=sorted({s for t in tasks_meta.values() for s in t["train"]["splits"]}), tasks=tasks_meta)
    import transformers
    meta["library_versions"]["transformers"] = transformers.__version__
    path = save_checkpoint(a.out, meta, heads_out)
    (OUT_RESULTS / "metrics.json").write_text(json.dumps(dict(checkpoint_id=cid, seconds=round(time.perf_counter() - t0, 1), tasks=report), indent=1))
    print("checkpoint", path, "seconds", round(time.perf_counter() - t0, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
