"""P-001 amendment A14 (exploratory): how much of the audio path's FakeAVCeleb score is a file cue?

The frozen P-001 audio models rescore the cached `audio_view` under three perturbations of the two
audio cues the shortcut audit names: loudness levelled to -23 dBFS RMS, 250 ms cut from both edges,
and both. "none" reruns the unperturbed audio through the same loop, as a check against the
official predictions. Writes one predictions file per seed, part and perturbation, in the format
`ddf evaluate predictions` reads, then scores each with identity-bootstrap intervals.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PERTURBATIONS = ("none", "loudness", "edges", "both")
SAMPLE_RATE = 16000
CUT = SAMPLE_RATE // 4                       # 250 ms
MIN_LEFT = SAMPLE_RATE                       # keep at least 1 s after cutting
TARGET_RMS = 10 ** (-23 / 20)
SEEDS = (17, 29, 43)
PARTS = ("test-identity-strict", "test")
COLUMNS = ("clip_id", "label", "probability", "predicted", "source", "manipulation_type", "method")


def too_short(valid: int) -> bool:
    return valid < 2 * CUT + MIN_LEFT


def perturb(wave: np.ndarray, valid: int, kind: str) -> tuple[np.ndarray, int]:
    """A copy of `wave` under one perturbation, and its new count of valid samples."""
    out = wave.copy()
    if kind in ("edges", "both") and not too_short(valid):
        kept = wave[CUT:valid - CUT].copy()
        out[:] = 0
        out[:kept.size] = kept
        valid = kept.size
    if kind in ("loudness", "both"):
        rms = float(np.sqrt(np.mean(out[:valid].astype(np.float64) ** 2)))
        if rms > 0:
            out[:valid] = out[:valid] * (TARGET_RMS / rms)
    return out, valid


def score_part(model, records, cache_index, store, device: str, threshold: float, batch: int = 16) -> dict:
    """Probabilities under every perturbation, and how many clips were too short to cut."""
    import torch

    from deepfake_detection.training.input_contract import audio_forward

    rows = {kind: [] for kind in PERTURBATIONS}
    short = 0
    for start in range(0, len(records), batch):
        chunk = records[start:start + batch]
        prepared = [store.load(cache_index[r.clip_id], views=("audio_view",)) for r in chunk]
        short += sum(too_short(int(p.audio_valid_samples)) for p in prepared)
        for kind in PERTURBATIONS:
            pairs = [perturb(p.audio_view, int(p.audio_valid_samples), kind) for p in prepared]
            values = torch.from_numpy(np.stack([w for w, _ in pairs])).float().to(device)
            counts = torch.tensor([n for _, n in pairs], dtype=torch.long)
            with torch.inference_mode():
                probabilities = torch.sigmoid(audio_forward(model, values, counts).logits).cpu().tolist()
            for record, probability in zip(chunk, probabilities, strict=True):
                rows[kind].append({
                    "clip_id": record.clip_id, "label": int(record.audio_fake), "probability": probability,
                    "predicted": int(probability >= threshold), "source": record.source,
                    "manipulation_type": record.manipulation_type, "method": record.method,
                })
    return {"rows": rows, "too_short": short}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, default=Path("runs/p001"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from deepfake_detection.branches.registry import load_branch
    from deepfake_detection.data.cache_index import read_cache_index
    from deepfake_detection.data.manifest import load_manifest
    from deepfake_detection.views.cache_store import CacheStore

    args.output.mkdir(parents=True, exist_ok=True)
    cache_index = read_cache_index(args.run / "cache-index.csv")
    store = CacheStore(args.run / "cache")
    summary = {"amendment": "A14", "exploratory": True, "too_short": {}}
    for seed in SEEDS:
        model, _ = load_branch(args.run / "checkpoints" / f"audio-seed{seed}.pt", "audio")
        model.to(args.device).eval()
        threshold_file = args.run / "evaluation" / f"audio-seed{seed}-threshold.json"
        threshold = json.loads(threshold_file.read_text(encoding="utf-8"))["threshold"]
        for part in PARTS:
            records = load_manifest(args.run / "usable" / f"{part}-audio.csv", dataset="FakeAVCeleb").records
            scored = score_part(model, records, cache_index, store, args.device, threshold)
            summary["too_short"][part] = scored["too_short"]
            for kind, rows in scored["rows"].items():
                stem = args.output / f"audio-seed{seed}-{part}-{kind}"
                with (stem.parent / f"{stem.name}-predictions.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=COLUMNS)
                    writer.writeheader()
                    writer.writerows(rows)
                subprocess.run([sys.executable, "-m", "deepfake_detection.cli", "evaluate", "predictions",
                                "--predictions", str(stem.parent / f"{stem.name}-predictions.csv"),
                                "--threshold-file", str(threshold_file),
                                "--output", str(stem.parent / f"{stem.name}-metrics.json"),
                                "--bootstrap-samples", "1000", "--seed", "17"],
                               check=True, stdout=subprocess.DEVNULL)
                metrics = json.loads((stem.parent / f"{stem.name}-metrics.json").read_text(encoding="utf-8"))
                print(f"seed {seed} {part} {kind}: AUC {metrics['overall']['metrics']['roc_auc']:.3f}", flush=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
