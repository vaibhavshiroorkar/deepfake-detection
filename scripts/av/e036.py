"""E-036: trained T-AV heads on frozen AV-HuBERT features, I-8b (real only) and I-8c (supervised). Preregistered in docs/EXPERIMENTS.md.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/av/e036.py train
    PYTHONPATH=src .venv/Scripts/python.exe scripts/av/e036.py evaluate

train     three seeds of each head on the cached train features; epoch chosen on val by the preregistered rule;
          checkpoints/T-AV/<I-8b|I-8c>-seed<k>.pt with metadata.
evaluate  val scores of every seed, the shuffled-audio controls, the E-035 blocks per head and the paired comparison against
          I-8a (data/_results/av/E-035/scores_val.csv) on the windows both scored.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("data")
FEATURES = ROOT / "_features_av"
OUT = ROOT / "_results" / "av" / "E-036"
CKPT = Path("checkpoints") / "T-AV"
MANIFEST = ROOT / "_manifests" / "av_lavdf_windows.manifest.csv"
SPLIT = ROOT / "_splits" / "av_lavdf_windows-8c8203b7d252.csv"
SEEDS = (0, 1, 2)
MIN_FACE = 0.8
SEED = 20261102
CELLS = ("video_fake", "audio_fake", "both_fake")


def load(split: str, with_shuffled: bool = False) -> tuple[list[dict], dict[str, np.ndarray]]:
    from dfd.av.features import cache_path

    assignment = {r["sample_id"]: r["split"] for r in csv.DictReader(open(SPLIT, encoding="utf-8"))}
    index = {r["sample_id"]: r for r in csv.DictReader(open(FEATURES / split / "index.csv", encoding="utf-8"))}
    rows, arrays = [], {"fused": [], "lips": [], "audio": []}
    shuffled, donors = [], []
    for r in csv.DictReader(open(MANIFEST, encoding="utf-8")):
        if assignment[r["sample_id"]] != split:
            continue
        entry = index.get(r["sample_id"])
        if entry is None or entry["status"] != "ok" or float(entry["face"]) < MIN_FACE:
            continue
        notes = dict(kv.split("=", 1) for kv in r["notes"].split(";"))
        stored = np.load(cache_path(FEATURES / split, r["sample_id"]))
        rows.append({"sample_id": r["sample_id"], "group_id": r["group_id"], "label": int(r["label"]),
                     "kind": notes["kind"], "cell": notes["cell"]})
        for key in arrays:
            arrays[key].append(stored[key])
        if with_shuffled:
            shuffled.append(stored["fused_shuffled"] if "fused_shuffled" in stored else None)
            donors.append(str(stored["donor"]) if "donor" in stored else None)
    out = {k: np.stack(v) for k, v in arrays.items()}
    if with_shuffled:
        out["fused_shuffled"] = shuffled
        out["donor"] = donors
    return rows, out


def _t(x: np.ndarray, device: str = "cuda") -> torch.Tensor:
    return torch.from_numpy(x).to(device).float()


def _auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


# ---------------------------------------------------------------- I-8b, real only

def alignment_loss(head, lips: torch.Tensor, audio: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Aligned frame pairs against the same lips with audio 5 to 20 frames away, and with another window's audio."""
    batch, frames, _ = lips.shape
    shift = int(torch.randint(5, 21, (1,), generator=generator, device="cpu"))
    shifted = torch.roll(audio, shifts=shift, dims=1)
    other = audio[torch.roll(torch.arange(batch, device=audio.device), 1)]
    positive = head(lips, audio)
    negative = torch.cat([head(lips, shifted), head(lips, other)], dim=1)
    return 0.5 * (F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive))
                  + F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative)))


def train_alignment(seed: int, train: dict, train_rows: list[dict], val: dict, val_rows: list[dict]):
    from dfd.av.heads import AlignmentHead

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    real = np.array([r["kind"] == "original" for r in train_rows])
    lips, audio = train["lips"][real], train["audio"][real]
    clean_val = np.array([r["kind"] != "forged" for r in val_rows])
    val_lips, val_audio = val["lips"][clean_val], val["audio"][clean_val]
    head = AlignmentHead().cuda()
    optimiser = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    best = (float("inf"), None, -1)
    for epoch in range(10):
        head.train()
        order = torch.randperm(len(lips), generator=generator)
        for start in range(0, len(order), 64):
            pick = order[start : start + 64].numpy()
            loss = alignment_loss(head, _t(lips[pick]), _t(audio[pick]), generator)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        head.eval()
        with torch.no_grad():
            val_loss = float(np.mean([alignment_loss(head, _t(val_lips[i : i + 256]), _t(val_audio[i : i + 256]),
                                                     torch.Generator().manual_seed(SEED)).item()
                                      for i in range(0, len(val_lips), 256)]))
        if val_loss < best[0]:
            best = (val_loss, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, epoch)
    head.load_state_dict(best[1])
    return head, {"selection": "lowest val alignment loss on genuine and original windows", "val_loss": best[0], "epoch": best[2]}


# ---------------------------------------------------------------- I-8c, supervised

def train_window(seed: int, train: dict, train_rows: list[dict], val: dict, val_rows: list[dict]):
    from dfd.av.heads import WindowHead

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    x = [train[k] for k in ("fused", "lips", "audio")]
    y = torch.tensor([r["label"] for r in train_rows], dtype=torch.float32, device="cuda")
    val_x = [val[k] for k in ("fused", "lips", "audio")]
    val_y = [r["label"] for r in val_rows]
    head = WindowHead().cuda()
    optimiser = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_weight = (y == 0).sum() / (y == 1).sum()
    best, waited = (-1.0, None, -1), 0
    for epoch in range(20):
        head.train()
        order = torch.randperm(len(y), generator=generator)
        for start in range(0, len(order), 128):
            pick = order[start : start + 128]
            logits = head(*(_t(a[pick.numpy()]) for a in x))
            loss = F.binary_cross_entropy_with_logits(logits, y[pick.cuda()], pos_weight=pos_weight)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        head.eval()
        with torch.no_grad():
            scores = torch.cat([head(*(_t(a[i : i + 512]) for a in val_x)) for i in range(0, len(val_y), 512)]).cpu().numpy()
        value = _auc(val_y, scores)
        if value > best[0]:
            best, waited = (value, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, epoch), 0
        else:
            waited += 1
            if waited >= 5:
                break
    head.load_state_dict(best[1])
    return head, {"selection": "highest val AUROC over all val windows, patience 5", "val_auc": best[0], "epoch": best[2]}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cmd_train() -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    train_rows, train = load("train")
    val_rows, val = load("val")
    print(f"train windows {len(train_rows)}, val windows {len(val_rows)}", flush=True)
    log = {}
    for name, fit in (("I-8b", train_alignment), ("I-8c", train_window)):
        for seed in SEEDS:
            head, info = fit(seed, train, train_rows, val, val_rows)
            meta = {"iteration": name, "seed": seed, **info, "manifest_sha256": _sha(MANIFEST), "split": SPLIT.name,
                    "features": "AV-HuBERT Base base_vox_iter5.pt, last layer, eye_px 64 (E-035 part A)",
                    "train_windows": len(train_rows), "val_windows": len(val_rows)}
            torch.save({"state_dict": head.state_dict(), "metadata": meta}, CKPT / f"{name}-seed{seed}.pt")
            log[f"{name}-seed{seed}"] = meta
            print(name, seed, info, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "training.json").write_text(json.dumps(log, indent=1))


def _score(name: str, head, data: dict, fused_key: str = "fused", audio: np.ndarray | None = None) -> np.ndarray:
    audio = data["audio"] if audio is None else audio
    out = []
    with torch.no_grad():
        for i in range(0, len(audio), 512):
            lips, sound = _t(data["lips"][i : i + 512]), _t(audio[i : i + 512])
            if name == "I-8b":
                out.append(head.window_score(lips, sound).cpu().numpy())
            else:
                fused = _t(np.stack(data[fused_key][i : i + 512])) if isinstance(data[fused_key], list) else _t(data[fused_key][i : i + 512])
                out.append(head(fused, lips, sound).cpu().numpy())
    return np.concatenate(out)


def cmd_evaluate() -> None:
    from dfd.av.heads import AlignmentHead, WindowHead
    from dfd.eval.bootstrap import cluster_bootstrap, paired_bootstrap

    rows, val = load("val", with_shuffled=True)
    position = {r["sample_id"]: i for i, r in enumerate(rows)}
    groups = np.array([r["group_id"] for r in rows])
    y = np.array([r["label"] for r in rows])

    def boot(mask, labels, scores):
        r = cluster_bootstrap(labels[mask], scores[:, mask], groups[mask], "auc", n_boot=2000, seed=SEED)
        return {"auc": round(r.estimate, 4), "low": round(r.low, 4), "high": round(r.high, 4),
                "seed_sd": None if r.seed_sd is None else round(r.seed_sd, 4), "n": int(mask.sum())}

    produced = np.array([r["kind"] in ("forged", "genuine") for r in rows])
    clean = np.array([r["kind"] != "forged" for r in rows])
    genuine = np.array([int(r["kind"] == "genuine") for r in rows])
    a_rows = {r["sample_id"]: float(r["s"]) for r in csv.DictReader(open(ROOT / "_results" / "av" / "E-035" / "scores_val.csv", encoding="utf-8"))}
    results, scores_all = {}, {}
    for name, cls in (("I-8b", AlignmentHead), ("I-8c", WindowHead)):
        per_seed = []
        for seed in SEEDS:
            head = cls().cuda().eval()
            head.load_state_dict(torch.load(CKPT / f"{name}-seed{seed}.pt")["state_dict"])
            per_seed.append(_score(name, head, val))
        scores = np.stack(per_seed)
        scores_all[name] = scores
        block = {"within_produced_all": boot(produced, y, scores)}
        for cell in CELLS:
            block[f"within_produced_{cell}"] = boot(produced & np.array([r["cell"] == cell for r in rows]), y, scores)
        block["genuine_vs_original"] = boot(clean, genuine, scores)
        block["all_val"] = boot(np.ones(len(rows), bool), y, scores)

        # Control: the clean windows with a donor's audio (and, for I-8c, the fused pass of that pair).
        has = [i for i in np.flatnonzero(clean) if val["donor"][i] is not None and val["donor"][i] in position]
        donor_audio = val["audio"][[position[val["donor"][i]] for i in has]]
        sub = {"lips": val["lips"][has], "audio": val["audio"][has], "fused": [val["fused_shuffled"][i] for i in has]}
        shuffled = []
        for seed in SEEDS:
            head = cls().cuda().eval()
            head.load_state_dict(torch.load(CKPT / f"{name}-seed{seed}.pt")["state_dict"])
            shuffled.append(_score(name, head, sub, audio=donor_audio))
        control_scores = np.concatenate([scores[:, has], np.stack(shuffled)], axis=1)
        control_y = np.array([0] * len(has) + [1] * len(has))
        control_groups = np.concatenate([groups[has], groups[has]])
        c = cluster_bootstrap(control_y, control_scores, control_groups, "auc", n_boot=2000, seed=SEED)
        block["shuffled_audio_control"] = {"auc": round(c.estimate, 4), "low": round(c.low, 4), "high": round(c.high, 4), "n": len(has)}

        # Paired against I-8a on the windows both scored.
        both = np.array([r["sample_id"] in a_rows for r in rows])
        base = np.array([a_rows.get(r["sample_id"], np.nan) for r in rows])
        for label, mask in (("within_produced_all", produced & both), ("all_val", both)):
            labels = y[mask]
            diff, _, _ = paired_bootstrap(labels, scores[:, mask], np.tile(base[mask], (len(SEEDS), 1)), groups[mask], "auc",
                                          n_boot=2000, seed=SEED)
            block[f"minus_I-8a_{label}"] = {"diff": round(diff.estimate, 4), "low": round(diff.low, 4), "high": round(diff.high, 4),
                                            "n": int(mask.sum())}
        results[name] = block
        print(name, json.dumps(block, indent=1), flush=True)
    with open(OUT / "scores_val.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "group_id", "label", "kind", "cell"] + [f"{n}_seed{s}" for n in scores_all for s in SEEDS])
        for i, r in enumerate(rows):
            w.writerow([r["sample_id"], r["group_id"], r["label"], r["kind"], r["cell"]]
                       + [round(float(scores_all[n][s, i]), 6) for n in scores_all for s in range(len(SEEDS))])
    (OUT / "results.json").write_text(json.dumps(results, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "evaluate"))
    command = parser.parse_args().command
    if not (OUT / "prereg.json").exists():
        raise SystemExit("data/_results/av/E-036/prereg.json missing: run scripts/av/prereg.py E-036 first")
    {"train": cmd_train, "evaluate": cmd_evaluate}[command]()


if __name__ == "__main__":
    main()
