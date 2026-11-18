"""P-001 amendment A19 (exploratory): AVH-Align's head, loss and score, retrained on our protocol.

    .venv/Scripts/python.exe scripts/run_a19.py

AVH-Align (Smeu et al., CVPR 2025; github.com/bit-ml/AVH-Align at 55401de, CC BY-NC-SA 4.0) trains
an MLP that scores a lips frame against the audio frames 15 either side and learns, with a softmax
over those 31 offsets, which one is in sync. Here it is retrained, unchanged except for the input
width (768, our frozen AV-HuBERT Base passes), on exactly the real FakeAVCeleb windows the P-001 sync
path used, at seeds 17, 29 and 43, then compared with the sync path on the same clips. The published
system (fine-tuned AV-HuBERT Large, trained on AV-Deepfake1M) is not evaluated; see amendment A19.
Writes results/p001-a19/; checkpoints go to runs/p001/a19/.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from deepfake_detection.evaluation.bootstrap import cluster_bootstrap_interval
from deepfake_detection.fusion.h2 import SeedPair, mean_seed_difference

ROOT = Path(__file__).resolve().parents[1]
OUTPUT, CHECKPOINTS = ROOT / "results" / "p001-a19", ROOT / "runs" / "p001" / "a19"
SEEDS = (17, 29, 43)
TAU, WIDTH = 15, 768
BATCH, LR, MAX_EPOCHS, STOP_PATIENCE, LR_PATIENCE = 1024, 1e-5, 500, 10, 5
FAKEAVCELEB = ROOT / "runs" / "avhubert-fakeavceleb-neutral"
SETS = {  # name: (feature cache, committed sync predictions with {seed})
    "test": (FAKEAVCELEB, "p001-sync/sync-test-seed{seed}"),
    "test-identity-strict": (
        FAKEAVCELEB,
        "p001-sync/sync-test-identity-strict-seed{seed}",
    ),
    "lavdf-test": (
        ROOT / "runs/avhubert-lavdf-test-neutral-a10",
        "p001-sync-targets/sync-lavdf-test-seed{seed}",
    ),
    "lavdf-window": (
        ROOT / "runs/avhubert-lavdf-test-neutral-a10",
        "p001-sync-targets/sync-lavdf-window-seed{seed}",
    ),
    "dfdc": (
        ROOT / "runs/avhubert-dfdc-neutral-a10",
        "p001-sync-targets/sync-dfdc-seed{seed}",
    ),
    "hifiavdf": (
        ROOT / "runs/avhubert-hifiavdf-neutral-a10",
        "p001-sync-targets/sync-hifiavdf-seed{seed}",
    ),
}


class FusionModel(nn.Module):
    """AVH-Align's model.py, with the input width as a parameter."""

    def __init__(
        self, visual_dim: int = WIDTH, audio_dim: int = WIDTH, hidden_dim: int = 1024
    ) -> None:
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim // 2)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, visual: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        return self.mlp(
            torch.cat((self.visual_proj(visual), self.audio_proj(audio)), dim=-1)
        )


def normalize(features: np.ndarray) -> torch.Tensor:
    """Float16 cache to float32, each frame scaled to unit L2 norm, as their loader does."""
    x = torch.from_numpy(np.asarray(features, dtype=np.float32))
    return x / torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)


def audio_windows(
    audio: torch.Tensor, window: torch.Tensor, frame: torch.Tensor
) -> torch.Tensor:
    """For each (window, frame) pair, the 2*TAU+1 audio frames around it, zeros past the window's edges.

    `audio` [N, T, D] already normalized; returns [B, 2*TAU+1, D].
    """
    padded = nn.functional.pad(
        audio, (0, 0, TAU, TAU)
    )  # zeros, as their out-of-range frames
    offsets = torch.arange(2 * TAU + 1, device=audio.device)
    return padded[window[:, None], frame[:, None] + offsets[None, :]]


def epoch_loss(model, lips, audio, optimizer=None) -> float:
    """Their run_epoch: frames in order, minus the log-softmax at the true offset, summed."""
    n, t, _ = lips.shape
    window = torch.arange(n, device=lips.device).repeat_interleave(t)
    frame = torch.arange(t, device=lips.device).repeat(n)
    total = 0.0
    model.train(optimizer is not None)
    with torch.set_grad_enabled(optimizer is not None):
        for start in range(0, len(window), BATCH):
            w, f = window[start : start + BATCH], frame[start : start + BATCH]
            visual = lips[w, f][:, None, :].expand(-1, 2 * TAU + 1, -1)
            outputs = model(visual, audio_windows(audio, w, f)).squeeze(-1)
            loss = -torch.log_softmax(outputs, dim=1)[:, TAU].sum()
            total += float(loss)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    return total / len(window)


@torch.no_grad()
def window_scores(model, lips, audio) -> np.ndarray:
    """Their process_video, per window: log-sum-exp over frames of minus the aligned-pair output."""
    model.eval()
    out = []
    for start in range(0, lips.shape[0], 256):
        output = model(lips[start : start + 256], audio[start : start + 256]).squeeze(
            -1
        )  # [b, T]
        out.append(torch.logsumexp(-output, dim=1).cpu().numpy())
    return np.concatenate(out)


def train(
    seed: int, lips, audio, val_lips, val_audio, device: str
) -> tuple[FusionModel, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FusionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=LR_PATIENCE
    )
    best, best_state, best_epoch, waited, history = float("inf"), None, 0, 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = epoch_loss(model, lips, audio, optimizer)
        val_loss = epoch_loss(model, val_lips, val_audio)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss})
        if val_loss < best:
            best, best_epoch, waited = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= STOP_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_val_loss": best,
        "epochs_run": len(history),
        "history": history,
    }


def _sync_module():
    spec = importlib.util.spec_from_file_location(
        "train_sync_path", ROOT / "scripts" / "train_sync_path.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_sync_path"] = module
    spec.loader.exec_module(module)
    return module


def _rows(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return {r["clip_id"]: r for r in csv.DictReader(f)}


class _Clip:
    def __init__(self, source: str, label: int, scores: list[float]) -> None:
        self.source_identity, self.label, self.scores = source, label, scores


def _mean_auc(clips) -> float:
    labels = [c.label for c in clips]
    return float(
        np.mean(
            [
                roc_auc_score(labels, [c.scores[s] for c in clips])
                for s in range(len(SEEDS))
            ]
        )
    )


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sync = _sync_module()
    split = ROOT / "runs" / "p001" / "split"
    real = {
        name: {
            k: v
            for k, v in sync.load_split(split / f"{name}.csv").items()
            if v["clip_fake"] == "0"
        }
        for name in ("train", "val")
    }
    _, lips, audio = sync.load_windows(FAKEAVCELEB, real["train"])
    _, val_lips, val_audio = sync.load_windows(FAKEAVCELEB, real["val"])
    lips, audio, val_lips, val_audio = (
        normalize(x).to(device) for x in (lips, audio, val_lips, val_audio)
    )
    print(
        f"real training windows {len(lips)}, real validation windows {len(val_lips)}",
        flush=True,
    )

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    models, training = [], {}
    for seed in SEEDS:
        model, info = train(seed, lips, audio, val_lips, val_audio, device)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "seed": seed,
                "info": {k: v for k, v in info.items() if k != "history"},
            },
            CHECKPOINTS / f"avh-align-seed{seed}.pt",
        )
        models.append(model)
        training[str(seed)] = info
        print(
            f"seed {seed}: best epoch {info['best_epoch']} of {info['epochs_run']}, "
            f"val loss {info['best_val_loss']:.4f}",
            flush=True,
        )
    (OUTPUT / "training.json").write_text(
        json.dumps(training, indent=1) + "\n", encoding="utf-8"
    )

    report = {"amendment": "A19", "exploratory": True, "sets": {}}
    for name, (features, stem) in SETS.items():
        tables = [
            _rows(ROOT / "results" / (stem.format(seed=s) + "-predictions.csv"))
            for s in SEEDS
        ]
        scored = [
            c for c, r in tables[0].items() if r["probability"] not in ("", "nan")
        ]
        owners, set_lips, set_audio = sync.load_windows(
            features, {c: tables[0][c] for c in scored}
        )
        set_lips, set_audio = (
            normalize(set_lips).to(device),
            normalize(set_audio).to(device),
        )
        per_seed = [window_scores(m, set_lips, set_audio) for m in models]
        clip_scores: dict[str, list[float]] = {}
        for s, values in enumerate(per_seed):
            sums: dict[str, list[float]] = {}
            for owner, value in zip(owners, values, strict=True):
                sums.setdefault(owner, []).append(float(value))
            for clip, v in sums.items():
                clip_scores.setdefault(clip, [None] * len(SEEDS))[s] = float(np.mean(v))
        clips = sorted(
            c
            for c in clip_scores
            if all(
                tables[k][c]["probability"] not in ("", "nan")
                for k in range(len(SEEDS))
            )
        )
        labels = np.array([int(tables[0][c]["label"]) for c in clips])
        groups = np.array([tables[0][c]["source"] for c in clips])
        items = [
            _Clip(g, int(y), clip_scores[c])
            for c, g, y in zip(clips, groups, labels, strict=True)
        ]
        interval = cluster_bootstrap_interval(items, _mean_auc, samples=1000, seed=17)
        pairs = [
            SeedPair(
                labels=labels,
                groups=groups,
                fusion=np.array([clip_scores[c][k] for c in clips]),
                visual=np.array([float(tables[k][c]["probability"]) for c in clips]),
            )
            for k in range(len(SEEDS))
        ]
        report["sets"][name] = {
            "clips": len(clips),
            "auc": {
                str(s): float(roc_auc_score(labels, [clip_scores[c][k] for c in clips]))
                for k, s in enumerate(SEEDS)
            },
            "auc_sync": {
                str(s): float(
                    roc_auc_score(
                        labels, [float(tables[k][c]["probability"]) for c in clips]
                    )
                )
                for k, s in enumerate(SEEDS)
            },
            "mean_auc": interval.estimate,
            "lower": interval.lower,
            "upper": interval.upper,
            "minus_sync": mean_seed_difference(pairs, samples=1000, seed=17),
        }
        with open(
            OUTPUT / f"avh-align-{name}-predictions.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(
                ["clip_id", "label", "source", *[f"score_seed{s}" for s in SEEDS]]
            )
            for c, g, y in zip(clips, groups, labels, strict=True):
                writer.writerow([c, int(y), g, *[f"{v:.6f}" for v in clip_scores[c]]])
        r = report["sets"][name]
        d = r["minus_sync"]
        print(
            f"{name:22} {r['clips']:5} clips  AVH {r['mean_auc']:.3f} [{r['lower']:.3f}, {r['upper']:.3f}]  "
            f"minus sync {d['estimate']:+.3f} [{d['lower']:+.3f}, {d['upper']:+.3f}]",
            flush=True,
        )
    (OUTPUT / "a19.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
