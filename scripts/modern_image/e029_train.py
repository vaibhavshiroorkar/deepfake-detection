"""E-029 training and validation (preregistered; amendment A1), then the freeze. Eval is not read.

Usage (repo root, PYTHONPATH=src): python scripts/modern_image/e029_train.py [--reduced | --dry SCRATCH_DIR]
For IMG-13 and IMG-15 and seeds 0, 1, 2: dfd.sweep.heads.train_head (the T-I recipe) on the training rows of split.csv, early stopping on the
validation rows; temperature (dfd.eval.calibration.fit_temperature) and the maximum-balanced-accuracy threshold
(dfd.eval.threshold.choose_threshold_max_balanced_accuracy) on validation. Validation AUC with a cluster bootstrap (fakes clustered by
generator, reals by image; dfd.eval.bootstrap). Encoder choice: higher seed-mean validation AUC, IMG-13 if the paired interval contains 0.
Writes checkpoints/modern-E029/ (six heads, meta.json, freeze.json) and data/_results/modern/e029/val_results.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
import _env  # noqa: E402

CK = lib.ROOT / "checkpoints" / "modern-E029"
SEEDS = [0, 1, 2]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    from dfd.eval import bootstrap as B
    from dfd.eval import calibration as C
    from dfd.eval import metrics as M
    from dfd.eval.threshold import choose_threshold_max_balanced_accuracy
    from dfd.infer.checkpoint import Head, save_head
    from dfd.sweep.heads import Split, train_head
    global CK
    reduced = "--reduced" in sys.argv or "--dry" in sys.argv
    out = _env.RES / "e029"
    res_dir = out
    if "--dry" in sys.argv:  # code test on the shards present: writes to a scratch folder, never the checkpoint folder
        CK = Path(sys.argv[sys.argv.index("--dry") + 1])
        res_dir = CK
    if (CK / "freeze.json").exists():
        sys.exit("already frozen; not retrained")
    split = list(csv.DictReader(open(out / "split.csv", encoding="utf-8")))
    cache = out / "cache_small"
    need = sorted({r["shard"] for r in split})
    have = {f"data/{f.stem}.parquet": f for f in cache.glob("*.npz")}
    missing = [s for s in need if s not in have]
    if missing and not reduced:
        sys.exit(f"{len(missing)} Small shards not extracted: {missing}")
    feat = {}
    for s, f in have.items():
        z = np.load(f)
        f13, f15 = z["f13"], z["f15"]   # load each array once: z["f13"][k] inside the loop re-reads the whole array on every access
        for k, rw in enumerate(z["row"]):
            feat[(s, int(rw))] = (f13[k], f15[k])
    rows = [r for r in split if r["split"] in ("train", "val") and (r["shard"], int(r["row"])) in feat]
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    ytr = np.array([int(r["label"]) for r in tr])
    yva = np.array([int(r["label"]) for r in va])
    gva = np.array([("g:" + r["model_name"]) if r["label"] == "1" else ("r:" + r["image_name"]) for r in va])
    res = {"experiment": "E-029", "stage": "train_validate", "run_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "reduced": bool(missing),
           "missing_shards": missing, "n_train": {"fake": int(ytr.sum()), "real": int((1 - ytr).sum())},
           "n_val": {"fake": int(yva.sum()), "real": int((1 - yva).sum())}, "n_val_generators": len({r["model_name"] for r in va if r["label"] == "1"}),
           "encoders": {}}
    CK.mkdir(parents=True, exist_ok=True)
    val_scores = {}
    heads_meta = {}
    for ei, enc in enumerate(["IMG-13", "IMG-15"]):
        Xtr = np.stack([feat[(r["shard"], int(r["row"]))][ei] for r in tr]).astype(np.float32)
        Xva = np.stack([feat[(r["shard"], int(r["row"]))][ei] for r in va]).astype(np.float32)
        trs = Split(Xtr, np.arange(len(tr)), ytr, np.array([r["model_name"] for r in tr]), np.arange(len(tr)).astype(str))
        vas = Split(Xva, np.arange(len(va)), yva, gva, np.arange(len(va)).astype(str))
        per = []
        S = []
        for seed in SEEDS:
            h = train_head(trs, vas, seed)
            head = Head(h.mean.astype(np.float32), h.std.astype(np.float32), h.w.astype(np.float32), float(h.b))
            pv = head.prob_rows(Xva)
            T = C.fit_temperature(C.logit(pv), yva).temperature
            pvc = C.sigmoid(C.logit(pv) / T)
            thr = choose_threshold_max_balanced_accuracy(yva, pvc).threshold
            name = f"head_{enc}_seed{seed}.npz"
            save_head(CK / name, head)
            S.append(pv)
            ptr = head.prob_rows(Xtr)
            per.append({"seed": seed, "file": name, "epochs": h.epochs, "best_epoch": h.best_epoch, "val_auc": M.roc_auc(yva, pv),
                        "train_auc": M.roc_auc(ytr, ptr), "temperature": float(T), "threshold": float(thr),
                        "val_balanced_accuracy": M.balanced_accuracy(yva, pvc, thr), "val_ece_before": M.ece(yva, pv), "val_ece_after": M.ece(yva, pvc)})
            heads_meta[name] = {"encoder": enc, "seed": seed, "temperature": float(T), "threshold": float(thr)}
            print(enc, seed, per[-1], flush=True)
        S = np.array(S)
        val_scores[enc] = S
        b = B.cluster_bootstrap(yva, S, gva, "auc", n_boot=lib.N_BOOT, seed=lib.SEED)
        res["encoders"][enc] = {"per_seed": per, "val_auc_seed_mean": b.as_dict()}
    diff, a, bb = B.paired_bootstrap(yva, val_scores["IMG-15"], val_scores["IMG-13"], gva, "auc", n_boot=lib.N_BOOT, seed=lib.SEED)
    dd = diff.as_dict()
    if dd["low"] > 0:
        chosen = "IMG-15"
    elif dd["high"] < 0:
        chosen = "IMG-13"
    else:
        chosen = "IMG-13"
    res["val_diff_IMG15_minus_IMG13"] = dd
    res["chosen_encoder"] = chosen
    res["chosen_rule"] = "higher seed-mean validation AUC; IMG-13 if the paired interval contains 0 (preregistered)"
    res["frozen_candidate"] = f"head_{chosen}_seed0.npz"
    json.dump(res, open(res_dir / "val_results.json", "w"), indent=1, default=float)
    meta = {"experiment": "E-029", "created": time.strftime("%Y-%m-%d %H:%M:%S %z"), "heads": heads_meta, "chosen_encoder": chosen,
            "frozen_candidate": res["frozen_candidate"],
            "encoders": {"IMG-13": {"name": "openai/clip-vit-large-patch14", "revision": "32bd64288804d66eefd0ccbe215aa642df71cc41"},
                         "IMG-15": {"name": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K", "revision": "1627032197142fbe2a7cfec626f4ced3ae60d07a",
                                    "input": "IMG-13 processor output re-normalised to mean 0.5, std 0.5 (same resize and crop)"}},
            "data": {"train": "CommunityForensics-Small subset I6-S (DATA_LOG), rev 6c539a534c07917307c381f5af4053c6091b5278",
                     "split_csv_sha256": sha(out / "split.csv"), "n_train": res["n_train"], "n_val": res["n_val"]},
            "recipe": "dfd.sweep.heads.train_head; temperature and max-balanced-accuracy threshold on validation (scripts/mvp/train_mvp.py rule)",
            "status": "research head, not installed; APPROVED_HEADS unchanged"}
    json.dump(meta, open(CK / "meta.json", "w"), indent=1)
    files = sorted([p.name for p in CK.iterdir() if p.suffix in (".npz", ".json") and p.name != "freeze.json"])
    json.dump({"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "sha256": {f: sha(CK / f) for f in files}}, open(CK / "freeze.json", "w"), indent=1)
    print("chosen", chosen, dd)


if __name__ == "__main__":
    main()
