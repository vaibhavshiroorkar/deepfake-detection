"""P-001 amendment A15 (exploratory): H3 in one direction, FakeAVCeleb to LAV-DF.

    .venv/Scripts/python.exe scripts/run_a15.py

The two binary heads are the P-001 visual path (video label) and audio path (audio label) at their
FakeAVCeleb validation thresholds, as committed in results/p001-branches (`predicted`). The
baseline marks both streams fake exactly when the clip is fake, using the true clip label.
Macro-F1 is the mean of the two streams' F1. The difference heads minus baseline is averaged over
the three seeds, with a bootstrap over LAV-DF original recordings. Writes results/p001-a15/a15.json.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from deepfake_detection.evaluation.bootstrap import cluster_bootstrap_interval

ROOT = Path(__file__).resolve().parents[1]

SEEDS = (17, 29, 43)
BRANCHES = ROOT / "results" / "p001-branches"
LABELS = {
    "window": ROOT / "data/manifests/lavdf-test-window.csv",
    "file": ROOT / "data/manifests/lavdf-test-neutral.csv",
}
OUTPUT = ROOT / "results" / "p001-a15"


@dataclass(frozen=True)
class Clip:
    clip_id: str
    source_identity: str  # the LAV-DF original recording
    video_true: int
    audio_true: int
    clip_true: int
    video_pred: tuple[int, ...]  # one per seed
    audio_pred: tuple[int, ...]


def macro_f1(clips: list[Clip], seed: int | None) -> float:
    """Heads at one seed index, or the clip-label baseline when `seed` is None."""
    video = [c.clip_true if seed is None else c.video_pred[seed] for c in clips]
    audio = [c.clip_true if seed is None else c.audio_pred[seed] for c in clips]
    return float(
        np.mean(
            [
                f1_score([c.video_true for c in clips], video, zero_division=0),
                f1_score([c.audio_true for c in clips], audio, zero_division=0),
            ]
        )
    )


def difference(clips: list[Clip]) -> float:
    """Mean over seeds of macro-F1(heads) minus macro-F1(baseline)."""
    baseline = macro_f1(clips, None)
    return float(
        np.mean(
            [macro_f1(clips, s) - baseline for s in range(len(clips[0].video_pred))]
        )
    )


def bootstrap(clips: list[Clip], samples: int = 1000, seed: int = 17) -> dict:
    values: list[float] = []

    def statistic(items):
        value = difference(list(items))
        values.append(value)
        return value

    interval = cluster_bootstrap_interval(clips, statistic, samples=samples, seed=seed)
    resampled = np.asarray(values[:-1])  # the last call is the estimate on all clips
    return {
        "estimate": interval.estimate,
        "lower": interval.lower,
        "upper": interval.upper,
        "p_one_sided": float(np.mean(resampled <= 0)),
        "successful_samples": interval.successful_samples,
    }


def _rows(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return {r["clip_id"]: r for r in csv.DictReader(f)}


def load(label: str) -> list[Clip]:
    truth = _rows(LABELS[label])
    preds = {
        b: [_rows(BRANCHES / f"{b}-seed{s}-lavdf-test-predictions.csv") for s in SEEDS]
        for b in ("visual", "audio")
    }
    common = sorted(
        set(truth)
        & set.intersection(*(set(t) for tables in preds.values() for t in tables))
    )
    return [
        Clip(
            clip,
            truth[clip]["source"],
            int(truth[clip]["video_fake"]),
            int(truth[clip]["audio_fake"]),
            int(truth[clip]["clip_fake"]),
            tuple(int(t[clip]["predicted"]) for t in preds["visual"]),
            tuple(int(t[clip]["predicted"]) for t in preds["audio"]),
        )
        for clip in common
    ]


def per_stream(clips: list[Clip]) -> dict:
    out = {}
    for stream in ("video", "audio"):
        true = [getattr(c, f"{stream}_true") for c in clips]
        rows = {}
        for i, seed in enumerate(SEEDS):
            pred = [getattr(c, f"{stream}_pred")[i] for c in clips]
            rows[str(seed)] = {
                "f1": f1_score(true, pred, zero_division=0),
                "precision": precision_score(true, pred, zero_division=0),
                "recall": recall_score(true, pred, zero_division=0),
            }
        base = [c.clip_true for c in clips]
        rows["baseline"] = {
            "f1": f1_score(true, base, zero_division=0),
            "precision": precision_score(true, base, zero_division=0),
            "recall": recall_score(true, base, zero_division=0),
        }
        out[stream] = rows
    return out


def main() -> int:
    report = {
        "amendment": "A15",
        "exploratory": True,
        "direction": "FakeAVCeleb to LAV-DF",
        "labels": {},
    }
    for label in LABELS:
        clips = load(label)
        report["labels"][label] = {
            "clips": len(clips),
            "identities": len({c.source_identity for c in clips}),
            "macro_f1_heads": {str(s): macro_f1(clips, i) for i, s in enumerate(SEEDS)},
            "macro_f1_baseline": macro_f1(clips, None),
            "difference": bootstrap(clips),
            "per_stream": per_stream(clips),
        }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "a15.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="utf-8"
    )
    for label, r in report["labels"].items():
        d = r["difference"]
        print(
            f"{label}: {r['clips']} clips, heads {r['macro_f1_heads']}, baseline {r['macro_f1_baseline']:.3f}, "
            f"difference {d['estimate']:+.3f} [{d['lower']:+.3f}, {d['upper']:+.3f}], p {d['p_one_sided']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
