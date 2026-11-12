"""I-1 runner (E-005): stage 1 (3 encoders x 3 learning rates, seed 0), selection on validation, stage 2 (seeds 1 and 2 of the winner).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/i1/run_i1.py            (resumable: finished runs are skipped)

One GPU job at a time. Selection uses the validation split only (CLAUDE.md rule 3): within an encoder the learning rate with the highest
validation AUC; across encoders the D-036 tie rule on validation (paired group bootstrap against the leader, cheapest in the tie set
wins; cost = measured full fine-tuning time per 1,000 frames, data/_logs/step_time_alone.json, batch 32). Output: data/_results/I-1/.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np

from dfd.sweep import rank
from dfd.train import finetune as FT

OUT = Path("data/_results/I-1")
CANDS = ["IMG-07", "IMG-17", "IMG-08"]      # D-049 item 2: the I-1 candidates in the order of the ranking
LRS = [1e-5, 3e-5, 1e-4]
N_BOOT = 2000


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def safe_run(c, lr, seed):
    try:
        return FT.run(c, lr, seed, OUT, workers=10)
    except Exception as e:  # recorded, not hidden
        log(f"FAILED {c} {lr} {seed}: {type(e).__name__}: {e}")
        d = OUT / f"{c}_lr{lr:g}_s{seed}"
        d.mkdir(parents=True, exist_ok=True)
        r = dict(name=d.name, cand=c, lr=lr, seed=seed, status="error", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])
        (d / "result.json").write_text(json.dumps(r, indent=1))
        import torch
        torch.cuda.empty_cache()
        return r


def costs():
    d = json.loads(Path("data/_logs/step_time_alone.json").read_text())
    return {r["cid"]: dict(s_per_1000=r["step_ms"] / 32.0, params=r["trainable_params"]) for r in d["results"] if r["batch"] == 32 and not r.get("oom")}


def select_stage1():
    res = {c: {lr: json.loads((OUT / f"{c}_lr{lr:g}_s0" / "result.json").read_text()) for lr in LRS} for c in CANDS}
    best = {}
    for c in CANDS:
        ok = {lr: r for lr, r in res[c].items() if r.get("status") == "ok" and r.get("val_auc") == r.get("val_auc")}
        if ok:
            lr = max(ok, key=lambda k: ok[k]["val_auc"])
            best[c] = lr
    scores, y, g = {}, None, None
    for c, lr in best.items():
        z = np.load(OUT / f"{c}_lr{lr:g}_s0" / "scores.npz", allow_pickle=True)
        scores[c] = z["val_scores"][None, :]
        y, g = z["val_y"], z["val_group"]
    cs = costs()
    sel = rank.select(list(best), scores, y, g, {c: cs[c] for c in best}, N_BOOT)
    out = dict(rule="D-036 on validation; lr by validation AUC within an encoder", best_lr=best, leader=sel["leader"], tie_set=sel["tie_set"],
               winner=sel["winner"], winner_lr=best[sel["winner"]], cost_decided=sel["cost_decided"],
               val_auc={c: sel["boot"][c].as_dict() for c in best}, diffs_vs_leader={c: d.as_dict() for c, d in sel["diffs"].items()},
               permutation={c: p for c, p in sel["perm"].items()}, costs={c: cs[c] for c in best},
               stage1={c: {f"{lr:g}": {k: r.get(k) for k in ("status", "val_auc", "devx_auc", "train_subset_auc", "train_seconds", "wall_seconds", "peak_vram_gb")}
                           for lr, r in res[c].items()} for c in CANDS},
               date=time.strftime("%Y-%m-%d %H:%M"))
    (OUT / "selection.json").write_text(json.dumps(out, indent=1))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for c in CANDS:
        for lr in LRS:
            log(f"stage 1: {c} lr {lr:g} seed 0")
            safe_run(c, lr, 0)
    sel = select_stage1()
    log(f"selection: winner {sel['winner']} lr {sel['winner_lr']:g}, tie set {sel['tie_set']}, leader {sel['leader']}")
    for seed in (1, 2):
        log(f"stage 2: {sel['winner']} lr {sel['winner_lr']:g} seed {seed}")
        safe_run(sel["winner"], sel["winner_lr"], seed)
    (OUT / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + f" total {round(time.time() - t0)} s\n")
    log("finished")


if __name__ == "__main__":
    main()
