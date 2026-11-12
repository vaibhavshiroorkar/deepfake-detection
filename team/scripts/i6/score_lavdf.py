"""Score the LAV-DF subset with the frozen unimodal heads (D-057, E-010): no fitting of any kind happens here.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/i6/score_lavdf.py --splits train val [--final]

Visual: the fine-tuned IMG-17 of E-005 (learning rate 1e-5, seeds 0, 1, 2; weights in checkpoints/I-1), frame probabilities averaged per clip
(D-024), saved as raw clip probabilities per seed. Audio: the E-009 members AU-8, AU-4, AU-6 (heads in data/_results/I-4/head_*.npz, the
E-007 window of the audio track of the video), saved as raw clip probabilities per member and seed. Temperatures are fitted later, in the
analysis, on the source-domain validation splits and on LAV-DF validation (reported both ways). The test split is refused unless --final.
Output: data/_results/I-6/scores_<split>.npz
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dfd.data import loaders as L
from dfd.prep.pipeline import folder_name
from dfd.sweep import encoders, extract_audio
from dfd.train import finetune as FT

MANIFEST = "data/_manifests/lavdf_subset.manifest.csv"
SPLIT_CSV = "data/_splits/lavdf_subset-e9d12decaf1f.csv"
CROPS = "data/_crops/lavdf_subset_yunet_n16"
OUT = Path("data/_results/I-6")
MEMBERS = ["AU-8", "AU-4", "AU-6"]
SEEDS = (0, 1, 2)


def frames_of(recs):
    paths, clip_of = [], []
    for i, r in enumerate(recs):
        with open(Path(CROPS) / folder_name(r.sample_id) / "index.csv", "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row["detected"] == "1" and row["crop_path"] and int(row["frame_index"]) < 16:
                    paths.append(str(Path(CROPS) / row["crop_path"]))
                    clip_of.append(i)
    return np.array(paths), np.array(clip_of)


def visual(recs, workers):
    paths, clip_of = frames_of(recs)
    y = np.array([r.label for r in recs])
    out = np.full((len(SEEDS), len(recs)), np.nan, np.float32)
    for k, s in enumerate(SEEDS):
        enc = encoders.load("IMG-17")
        w = torch.load(f"checkpoints/I-1/IMG-17_lr1e-05_s{s}/weights.pt", map_location="cpu")
        names = dict(enc.model.named_parameters())
        with torch.no_grad():
            for n, v in w["params"].items():
                names[n].copy_(v.to(names[n].device, names[n].dtype))
        head = torch.nn.Linear(enc.dim, 1).cuda()
        head.load_state_dict(w["head"])
        p = FT.score_frames(enc, head, FT.Frames(paths, clip_of, y), workers)
        out[k] = FT.clip_mean(p, clip_of, len(recs))
        print("visual seed", s, "done; clips without a face frame:", int(np.isnan(out[k]).sum()), flush=True)
        del enc, head
        torch.cuda.empty_cache()
    return out


def audio(recs, workers):
    out = np.zeros((len(MEMBERS), len(SEEDS), len(recs)), np.float32)
    for m, c in enumerate(MEMBERS):
        enc = encoders.load_audio(c)
        ds = extract_audio.AudioSet(recs, cache=None, variant="trim_norm")
        dl = DataLoader(ds, batch_size=32, num_workers=workers, shuffle=False)
        feats = np.zeros((len(recs), 2 * enc.dim), np.float32)
        for x, idx in dl:
            feats[idx.numpy()] = enc(x.numpy()).cpu().numpy().astype(np.float16).astype(np.float32)   # float16 as in the stored features
        for k, s in enumerate(SEEDS):
            h = np.load(Path("data/_results/I-4") / f"head_{c}_s{s}.npz")
            z = ((feats - h["mean"]) / h["std"]) @ h["w"] + float(h["b"])
            out[m, k] = 1.0 / (1.0 + np.exp(-z))
        print("audio", c, "done", flush=True)
        del enc
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for split in a.splits:
        recs = L.select_records(MANIFEST, SPLIT_CSV, split, modality="video", final=a.final)
        t0 = time.time()
        notes = [dict(kv.split("=") for kv in r.notes.split(";")) for r in recs]
        vis, aud = visual(recs, a.workers), audio(recs, a.workers)
        np.savez(OUT / f"scores_{split}.npz", sample_id=np.array([r.sample_id for r in recs]), y_fused=np.array([r.label for r in recs]),
                 y_video=np.array([int(n["modify_video"]) for n in notes]), y_audio=np.array([int(n["modify_audio"]) for n in notes]),
                 group=np.array([r.group_id for r in recs]), cell=np.array([r.method for r in recs]), vis_p=vis, aud_p=aud, members=np.array(MEMBERS))
        print(split, len(recs), "scored in", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
