"""P-001 amendment A16 (exploratory): does the sync score move when the audio of a real clip is moved?

    .venv/Scripts/python.exe scripts/run_a16.py

Every real clip of the neutral FakeAVCeleb test partition with a face in at least 80% of the 50
frames of its 0 to 2 s window. Lips always come from 0 to 2 s. Audio comes from 0 to 2 s (original),
from later by 0.04, 0.08, 0.20, 0.40 or 0.80 s, or from another real test clip of a different
person (a derangement drawn with seed 17). Frozen AV-HuBERT Base and the three P-001 sync heads,
features cast to float16 as in the feature cache. Writes results/p001-a16/a16.json and windows.csv.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from deepfake_detection.avsync.features import (
    WINDOW_S,
    encode,
    init_worker,
    prepare_window,
)
from deepfake_detection.avsync.heads import AlignmentHead
from deepfake_detection.avsync.mouth import read_window
from deepfake_detection.evaluation.bootstrap import cluster_bootstrap_interval
from deepfake_detection.streams.avhubert import load_avhubert

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "p001-a16"
SEEDS = (17, 29, 43)
SHIFTS = (0.04, 0.08, 0.20, 0.40, 0.80)
MIN_FACE, FRAMES = 0.8, 50


def derangement(sources: list[str], seed: int = 17) -> list[int]:
    """For each clip, the index of another clip whose source differs. Redrawn until every pair differs."""
    rng = np.random.default_rng(seed)
    for _ in range(10_000):
        order = rng.permutation(len(sources))
        if all(sources[i] != sources[j] for i, j in enumerate(order)):
            return [int(j) for j in order]
    raise ValueError("no derangement with different sources found")


def real_test_clips() -> list[dict]:
    with open(ROOT / "runs/p001/split/test.csv", encoding="utf-8", newline="") as f:
        test = {r["clip_id"] for r in csv.DictReader(f)}
    with open(
        ROOT / "data/manifests/fakeavceleb-neutral.csv", encoding="utf-8", newline=""
    ) as f:
        return [
            r
            for r in csv.DictReader(f)
            if r["clip_id"] in test and r["manipulation_type"] == "RealVideo-RealAudio"
        ]


class _Pair:
    """One clip's original and manipulated window, as two rows of an AUC, with its person."""

    def __init__(self, source: str, original: list[float], moved: list[float]) -> None:
        self.source_identity, self.original, self.moved = source, original, moved


def mean_auc(pairs) -> float:
    """Mean over seeds of the AUC of moved windows against original windows."""
    values = []
    for s in range(len(SEEDS)):
        scores = [p.original[s] for p in pairs] + [p.moved[s] for p in pairs]
        values.append(roc_auc_score([0] * len(pairs) + [1] * len(pairs), scores))
    return float(np.mean(values))


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    init_worker(str(ROOT / "models/mediapipe/face_landmarker.task"), 64.0)
    model = load_avhubert(ROOT / "models/avhubert/base_vox_iter5.pt").to(device).eval()
    heads = []
    for seed in SEEDS:
        state = torch.load(
            ROOT / f"runs/p001/sync/sync-seed{seed}.pt",
            map_location="cpu",
            weights_only=True,
        )
        head = AlignmentHead(width=768)
        head.load_state_dict(state["state_dict"])
        heads.append(head.to(device).eval())

    usable = []
    for row in real_test_clips():
        path = str(ROOT / "data" / row["video_path"])
        face, crops, audio = prepare_window((path, 0.0))
        if (
            isinstance(face, str)
            or crops is None
            or face < MIN_FACE
            or len(crops) != FRAMES
        ):
            continue
        usable.append(
            {
                "clip_id": row["clip_id"],
                "source": row["source"],
                "path": path,
                "crops": crops,
                "audio": audio,
            }
        )
    swap = derangement([u["source"] for u in usable])

    def misfit(crops: np.ndarray, audio: np.ndarray) -> list[float]:
        features = encode(model, crops, audio, device=device)
        lips = torch.from_numpy(features["lips"].astype(np.float32))[None].to(device)
        sound = torch.from_numpy(features["audio"].astype(np.float32))[None].to(device)
        if lips.shape[1] != FRAMES or sound.shape[1] != FRAMES:
            raise ValueError("window is not 50 frames")
        with torch.no_grad():
            return [
                float(1 - math.exp(-h.window_score(lips, sound).item())) for h in heads
            ]

    conditions = ["original", *[f"shift {s:.2f} s" for s in SHIFTS], "swap"]
    scores: dict[str, list[list[float]]] = {c: [] for c in conditions}
    for i, clip in enumerate(usable):
        scores["original"].append(misfit(clip["crops"], clip["audio"]))
        for s in SHIFTS:
            moved = read_window(clip["path"], s, WINDOW_S)[1]
            scores[f"shift {s:.2f} s"].append(misfit(clip["crops"], moved))
        scores["swap"].append(misfit(clip["crops"], usable[swap[i]]["audio"]))

    report = {
        "amendment": "A16",
        "exploratory": True,
        "clips": len(usable),
        "identities": len({u["source"] for u in usable}),
        "conditions": {},
    }
    for c in conditions[1:]:
        pairs = [
            _Pair(u["source"], o, m)
            for u, o, m in zip(usable, scores["original"], scores[c], strict=True)
        ]
        interval = cluster_bootstrap_interval(pairs, mean_auc, samples=1000, seed=17)
        per_seed = {}
        for k, seed in enumerate(SEEDS):
            per_seed[str(seed)] = {
                "auc": float(
                    roc_auc_score(
                        [0] * len(pairs) + [1] * len(pairs),
                        [p.original[k] for p in pairs] + [p.moved[k] for p in pairs],
                    )
                ),
                "share_rising": float(
                    np.mean([p.moved[k] > p.original[k] for p in pairs])
                ),
            }
        report["conditions"][c] = {
            "mean_auc": interval.estimate,
            "lower": interval.lower,
            "upper": interval.upper,
            "per_seed": per_seed,
        }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "a16.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="utf-8"
    )
    with open(OUTPUT / "windows.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["clip_id", "source", "condition", *[f"misfit_seed{s}" for s in SEEDS]]
        )
        for c in conditions:
            for u, values in zip(usable, scores[c], strict=True):
                writer.writerow(
                    [u["clip_id"], u["source"], c, *[f"{v:.6f}" for v in values]]
                )
    for c, r in report["conditions"].items():
        print(
            f"{c:15} AUC {r['mean_auc']:.3f} [{r['lower']:.3f}, {r['upper']:.3f}]  "
            f"rising {np.mean([v['share_rising'] for v in r['per_seed'].values()]):.2f}"
        )
    print(f"{report['clips']} clips, {report['identities']} identities")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
