"""Every number the app shows, read from the evaluation files under runs/.

Nothing here is typed in. A row whose file is missing is left out, and the
source path travels with each row so a reader can open the file behind it.
"""

from __future__ import annotations

import functools
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deepfake_detection.paths import DATA_DIR, RUNS_DIR

PROGRAM = "program-20260906/evaluation"
GATE = "design-b-20260910/evaluation"
FAVC_ROOT = DATA_DIR / "FakeAVCeleb_v1.2" / "FakeAVCeleb_v1.2"


def _read(relative: str) -> dict | None:
    try:
        return json.loads((RUNS_DIR / relative).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _metrics_block(report: dict) -> dict | None:
    if isinstance(report.get("metrics"), dict):
        return report["metrics"]
    overall = report.get("overall")
    if isinstance(overall, dict) and isinstance(overall.get("metrics"), dict):
        return overall["metrics"]
    return None


def _auc_interval(report: dict) -> list[float] | None:
    for key in ("intervals", "confidence_intervals"):
        block = report.get(key)
        if isinstance(block, dict) and isinstance(block.get("roc_auc"), dict):
            return [block["roc_auc"]["lower"], block["roc_auc"]["upper"]]
    return None


def _report_row(group: str, model: str, dataset: str, stem: str) -> dict | None:
    report = _read(f"{stem}-metrics.json")
    if report is None:
        return None
    metrics = _metrics_block(report)
    if metrics is None:
        return None
    rows = report.get("rows") or (report.get("overall") or {}).get("scored")
    return {
        "group": group,
        "model": model,
        "dataset": dataset,
        "roc_auc": metrics.get("roc_auc"),
        "ci": _auc_interval(report),
        "eer": metrics.get("eer"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "ece": metrics.get("expected_calibration_error"),
        "recall": metrics.get("recall"),
        "fpr": metrics.get("fpr"),
        "clips": rows,
        "source": f"runs/{stem}-metrics.json",
    }


def _auc_row(
    group: str,
    model: str,
    dataset: str,
    auc: float | None,
    ci: list[float] | None,
    clips: int | None,
    source: str,
) -> dict | None:
    if auc is None:
        return None
    return {
        "group": group,
        "model": model,
        "dataset": dataset,
        "roc_auc": auc,
        "ci": ci,
        "eer": None,
        "balanced_accuracy": None,
        "ece": None,
        "recall": None,
        "fpr": None,
        "clips": clips,
        "source": source,
    }


def metric_table() -> list[dict]:
    """One row per model and dataset, grouped by how the model was built."""
    rows: list[dict | None] = []
    design_a = "Trained on FakeAVCeleb, per-branch (Design A)"
    for dataset, split in (
        ("FakeAVCeleb test", "in-domain-test"),
        ("DFDC", "dfdc"),
        ("Celeb-DF-v2", "celebdf"),
    ):
        rows.append(
            _report_row(
                "Served in this app",
                "Visual ensemble (6 models)",
                dataset,
                f"{PROGRAM}/visual-ensemble-{split}",
            )
        )
    for dataset, partition in (("FakeAVCeleb test", "in-domain"), ("DFDC", "dfdc")):
        rows.append(
            _report_row(
                "Served in this app",
                "Audio-visual fusion (served head)",
                dataset,
                f"{GATE}/gate-fusion-{partition}",
            )
        )
    for dataset, split in (("FakeAVCeleb test", "in-domain-test"), ("DFDC", "dfdc")):
        for seed in (17, 29, 43):
            rows.append(
                _report_row(
                    design_a,
                    f"Visual EfficientNet-B0, seed {seed}",
                    dataset,
                    f"{PROGRAM}/visual-{split}-seed{seed}",
                )
            )
        for seed, suffix in ((17, ""), (29, "-seed29"), (43, "-seed43")):
            rows.append(
                _report_row(
                    design_a,
                    f"Visual ConvNeXt-Tiny, seed {seed}",
                    dataset,
                    f"{PROGRAM}/visual-convnext-{split}{suffix}",
                )
            )
    for seed, suffix in ((17, ""), (29, "-seed29"), (43, "-seed43")):
        rows.append(
            _report_row(
                design_a,
                f"Visual EfficientNet-B0, seed {seed}",
                "Celeb-DF-v2",
                f"{PROGRAM}/visual-celebdf{suffix}",
            )
        )
    for seed, suffix in ((17, ""), (29, "-seed29"), (43, "-seed43")):
        rows.append(
            _report_row(
                design_a,
                f"Visual ConvNeXt-Tiny, seed {seed}",
                "Celeb-DF-v2",
                f"{PROGRAM}/visual-convnext-celebdf{suffix}",
            )
        )
    for seed, suffix in ((17, ""), (29, "-seed29"), (43, "-seed43")):
        rows.append(
            _report_row(
                design_a,
                f"Audio Wav2Vec2 (audio label), seed {seed}",
                "FakeAVCeleb test",
                f"{PROGRAM}/audio-in-domain-test{suffix}",
            )
        )
    for seed in (17, 29, 43):
        rows.append(
            _report_row(
                design_a,
                f"Sync (mouth + audio offset), seed {seed}",
                "FakeAVCeleb test",
                f"{PROGRAM}/sync-final-sync-seed{seed}",
            )
        )
    spread = _read(f"{PROGRAM}/fusion-seed-spread.json") or {}
    for partition, dataset in (("in-domain", "FakeAVCeleb test"), ("DFDC", "DFDC")):
        for seed, entry in (
            spread.get("own_head_partitions", {}).get(partition) or {}
        ).items():
            rows.append(
                _auc_row(
                    design_a,
                    f"Logistic fusion (visual + audio + sync), seed {seed}",
                    dataset,
                    entry.get("roc_auc"),
                    [entry.get("roc_auc_lower"), entry.get("roc_auc_upper")],
                    entry.get("clips"),
                    f"runs/{PROGRAM}/fusion-seed-spread.json",
                )
            )

    design_b = "Trained on FakeAVCeleb, five streams (Design B, seed 17)"
    streams = _read("design-b-20260910/evaluation/stream-scores.json") or {}
    names = {
        "stream-emotion": "Emotion stream (cross-attention)",
        "visual-efficientnet": "Visual stream, EfficientNet",
        "visual-dinov3": "Visual stream, DINOv3",
        "final-audio-seed17": "Audio stream, Wav2Vec2 (clip label)",
        "stream-lipsync": "Lip-sync stream (cross-attention)",
    }
    for key, label in names.items():
        entry = streams.get(key) or {}
        for partition, dataset in (("in-domain", "FakeAVCeleb test"), ("dfdc", "DFDC")):
            rows.append(
                _auc_row(
                    design_b,
                    label,
                    dataset,
                    entry.get(partition),
                    entry.get(f"{partition}_ci"),
                    None,
                    "runs/design-b-20260910/evaluation/stream-scores.json",
                )
            )
    ablation = (_read("design-b-20260910/evaluation/fusion-ablations.json") or {}).get(
        "partitions", {}
    )
    for partition, dataset in (("in-domain", "FakeAVCeleb test"), ("dfdc", "DFDC")):
        block = ablation.get(partition) or {}
        for key, label in (
            ("deep_masking", "Deep fusion, all five streams"),
            ("late_fusion", "Late fusion, all five streams"),
        ):
            entry = block.get(key) or {}
            rows.append(
                _auc_row(
                    design_b,
                    label,
                    dataset,
                    entry.get("auc"),
                    entry.get("ci"),
                    block.get("clips"),
                    "runs/design-b-20260910/evaluation/fusion-ablations.json",
                )
            )

    ffpp = (_read("ffpp-20260913/evaluation/zero-shot.json") or {}).get(
        "partitions", {}
    )
    for key, dataset in (
        ("ffpp-test", "FF++ c23 test"),
        ("celebdf", "Celeb-DF-v2"),
        ("dfdc", "DFDC"),
        ("fakeavceleb-test", "FakeAVCeleb test"),
    ):
        entry = ffpp.get(key) or {}
        rows.append(
            _auc_row(
                "Trained on FaceForensics++ c23 (standard protocol)",
                "Visual EfficientNet-B0",
                dataset,
                entry.get("roc_auc"),
                entry.get("ci"),
                entry.get("clips"),
                "runs/ffpp-20260913/evaluation/zero-shot.json",
            )
        )

    probe = _read("probe-20260914/df26-norm-encoder-comparison.json") or {}
    reference = probe.get("reference") or {}
    rows.append(
        _auc_row(
            "Unseen generators, DF26 leave-one-generator-out (frozen probes)",
            "DINOv3 linear probe",
            "DF26 macro",
            reference.get("macro_mean"),
            None,
            None,
            "runs/probe-20260914/df26-norm-encoder-comparison.json",
        )
    )
    for key, label in (
        ("clip", "CLIP linear probe"),
        ("pe", "Perception Encoder linear probe"),
    ):
        entry = (probe.get("comparisons") or {}).get(key) or {}
        rows.append(
            _auc_row(
                "Unseen generators, DF26 leave-one-generator-out (frozen probes)",
                label,
                "DF26 macro",
                entry.get("macro_mean"),
                None,
                None,
                "runs/probe-20260914/df26-norm-encoder-comparison.json",
            )
        )
    return [row for row in rows if row is not None]


def _predictions(stem: str) -> pd.DataFrame | None:
    path = RUNS_DIR / f"{stem}-predictions.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    return frame[frame["probability"].notna()]


def _roc_points(labels: np.ndarray, scores: np.ndarray, points: int = 120) -> list:
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    grid = np.linspace(0, 1, points)
    sampled = np.interp(grid, fpr, tpr)
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "fpr": [round(float(x), 4) for x in grid],
        "tpr": [round(float(y), 4) for y in sampled],
    }


def roc_curves() -> dict[str, list[dict]]:
    curves: dict[str, list[dict]] = {}
    for dataset, split, fusion in (
        ("FakeAVCeleb test", "in-domain-test", "fusion-test"),
        ("DFDC", "dfdc", "fusion-dfdc"),
    ):
        entries = []
        for label, stem in (
            ("Visual ensemble", f"{PROGRAM}/visual-ensemble-{split}"),
            ("Single face model (seed 17)", f"{PROGRAM}/visual-{split}-seed17"),
            ("Logistic fusion (seed 17)", f"{PROGRAM}/{fusion}"),
        ):
            frame = _predictions(stem)
            if frame is None or frame["label"].nunique() < 2:
                continue
            curve = _roc_points(
                frame["label"].to_numpy(), frame["probability"].to_numpy()
            )
            entries.append({"label": label, **curve})
        curves[dataset] = entries
    return curves


# Log-spaced on both sides of 0.5, plus the app's preset values exactly, so a
# preset reads its own row rather than the nearest grid point.
THRESHOLDS = tuple(
    float(x)
    for x in np.unique(
        np.concatenate(
            [
                np.geomspace(0.001, 0.5, 40),
                1 - np.geomspace(0.5, 0.001, 40),
                [0.02, 0.05, 0.1, 0.5],
            ]
        ).round(4)
    )
)


def operating_points() -> dict[str, dict[str, Any]]:
    """Measured detection and false-alarm rates at each threshold.

    This is a description of saved test predictions, not a fit: the app shows
    a reader what a threshold has done on known footage, and the default stays
    at 0.5.
    """
    out: dict[str, dict[str, Any]] = {}
    for model, stems in (
        (
            "ensemble",
            {
                "FakeAVCeleb test": f"{PROGRAM}/visual-ensemble-in-domain-test",
                "DFDC": f"{PROGRAM}/visual-ensemble-dfdc",
                "Celeb-DF-v2": f"{PROGRAM}/visual-ensemble-celebdf",
            },
        ),
        (
            "visual-best",
            {
                "FakeAVCeleb test": f"{PROGRAM}/visual-in-domain-test-seed17",
                "DFDC": f"{PROGRAM}/visual-dfdc-seed17",
                "Celeb-DF-v2": f"{PROGRAM}/visual-celebdf",
            },
        ),
        (
            "multimodal",
            {
                "FakeAVCeleb test": f"{GATE}/gate-fusion-in-domain",
                "DFDC": f"{GATE}/gate-fusion-dfdc",
            },
        ),
    ):
        datasets = {}
        for dataset, stem in stems.items():
            frame = _predictions(stem)
            if frame is None:
                continue
            labels = frame["label"].to_numpy()
            scores = frame["probability"].to_numpy()
            fakes, reals = scores[labels == 1], scores[labels == 0]
            datasets[dataset] = {
                "fakes": int(len(fakes)),
                "reals": int(len(reals)),
                "detection": [
                    round(float((fakes >= t).mean()), 4) if len(fakes) else None
                    for t in THRESHOLDS
                ],
                "false_alarm": [
                    round(float((reals >= t).mean()), 4) if len(reals) else None
                    for t in THRESHOLDS
                ],
            }
        out[model] = {"thresholds": list(THRESHOLDS), "datasets": datasets}
    return out


def model_metrics(reports: dict[str, str]) -> dict[str, dict]:
    found = {}
    for dataset, stem in reports.items():
        report = _read(f"{stem}-metrics.json")
        metrics = _metrics_block(report) if report else None
        if metrics:
            found[dataset] = {
                "roc_auc": metrics.get("roc_auc"),
                "ci": _auc_interval(report),
            }
    return found


def seed_spread() -> dict[str, list[dict]]:
    """DFDC ROC-AUC per seed, the spread F21 and F27 measure."""
    out: dict[str, list[dict]] = {}
    for label, stems in (
        (
            "Visual EfficientNet-B0",
            [(s, f"{PROGRAM}/visual-dfdc-seed{s}") for s in (17, 29, 43)],
        ),
        (
            "Visual ConvNeXt-Tiny",
            [
                (17, f"{PROGRAM}/visual-convnext-dfdc"),
                (29, f"{PROGRAM}/visual-convnext-dfdc-seed29"),
                (43, f"{PROGRAM}/visual-convnext-dfdc-seed43"),
            ],
        ),
    ):
        values = []
        for seed, stem in stems:
            report = _read(f"{stem}-metrics.json")
            metrics = _metrics_block(report) if report else None
            if metrics:
                values.append({"seed": seed, "roc_auc": metrics["roc_auc"]})
        out[label] = values
    spread = _read(f"{PROGRAM}/fusion-seed-spread.json") or {}
    out["Logistic fusion"] = [
        {"seed": int(seed), "roc_auc": entry["roc_auc"]}
        for seed, entry in (
            spread.get("own_head_partitions", {}).get("DFDC") or {}
        ).items()
    ]
    ensemble = _read(f"{PROGRAM}/visual-ensemble-dfdc-metrics.json")
    if ensemble:
        out["Visual ensemble (all six)"] = [
            {"seed": "all", "roc_auc": ensemble["metrics"]["roc_auc"]}
        ]
    return out


def ensemble_summary() -> dict | None:
    """DFDC, where the ensemble helps, beside Celeb-DF and MNW, where it does not."""
    report = _read(f"{PROGRAM}/visual-ensemble-dfdc-metrics.json")
    if report is None:
        return None
    summary = {
        "roc_auc": report["metrics"]["roc_auc"],
        "ci": _auc_interval(report),
        "paired": report.get("paired_roc_auc_difference"),
        "members": report.get("member_roc_auc"),
        "celebdf": None,
        "mnw": None,
    }
    celebdf = _read(f"{PROGRAM}/visual-ensemble-celebdf-metrics.json")
    if celebdf is not None:
        summary["celebdf"] = {
            "roc_auc": celebdf["metrics"]["roc_auc"],
            "ci": _auc_interval(celebdf),
            "paired": celebdf.get("paired_roc_auc_difference"),
        }
    mnw = _read(f"{PROGRAM}/visual-ensemble-mnw-metrics.json")
    if mnw is not None:
        summary["mnw"] = {
            "fakes": mnw["fakes"],
            "ensemble": mnw["fakes_detected"]["ensemble"],
            "best_single": mnw["fakes_detected"]["efficientnet-b0-seed17"],
        }
    return summary


# Hand-picked to show what each kind of forgery does to the face models,
# including the cases they miss. Every one sits in a test partition the models
# never trained on.
SAMPLE_CLIPS: tuple[dict[str, str], ...] = (
    {
        "id": "favc-real",
        "title": "Genuine speaker",
        "dataset": "FakeAVCeleb test",
        "truth": "real",
        "kind": "Unaltered video and voice",
        "glob": "RealVideo-RealAudio/*/*/id00032/00028.mp4",
    },
    {
        "id": "favc-faceswap",
        "title": "Face swap",
        "dataset": "FakeAVCeleb test",
        "truth": "fake",
        "kind": "Face replaced, voice real",
        "glob": "FakeVideo-RealAudio/*/*/id00032/00028_00.mp4",
    },
    {
        "id": "favc-lipsync",
        "title": "Lip-sync forgery",
        "dataset": "FakeAVCeleb test",
        "truth": "fake",
        "kind": "Mouth re-animated to a cloned voice (Wav2Lip)",
        "csv_method": "wav2lip",
        "csv_type": "FakeVideo-FakeAudio",
    },
    {
        "id": "favc-voiceclone",
        "title": "Voice clone only",
        "dataset": "FakeAVCeleb test",
        "truth": "fake",
        "kind": "Real video, cloned voice. Face models cannot see this.",
        "csv_method": "rtvc",
        "csv_type": "RealVideo-FakeAudio",
    },
    {
        "id": "dfdc-fake-caught",
        "title": "DFDC swap, caught",
        "dataset": "DFDC",
        "truth": "fake",
        "kind": "Unseen capture conditions",
        "dfdc": "caught",
    },
    {
        "id": "dfdc-fake-missed",
        "title": "DFDC swap, missed",
        "dataset": "DFDC",
        "truth": "fake",
        "kind": "Unseen capture conditions",
        "dfdc": "missed",
    },
    {
        "id": "dfdc-real",
        "title": "DFDC genuine",
        "dataset": "DFDC",
        "truth": "real",
        "kind": "Unseen capture conditions",
        "dfdc": "real",
    },
)


def _favc_path_from_clip_id(clip_id: str) -> Path | None:
    parts = clip_id.split("__")
    if len(parts) != 3:
        return None
    hits = glob.glob(str(FAVC_ROOT / "*" / "*" / "*" / parts[1] / f"{parts[2]}.mp4"))
    return Path(hits[0]) if len(hits) == 1 else None


@functools.cache
def samples() -> dict[str, dict]:
    """The sample clips that exist on this machine, keyed by id."""
    found: dict[str, dict] = {}
    favc = _predictions(f"{PROGRAM}/visual-ensemble-in-domain-test")
    dfdc = _predictions(f"{PROGRAM}/visual-ensemble-dfdc")
    for spec in SAMPLE_CLIPS:
        path: Path | None = None
        if "glob" in spec:
            hits = glob.glob(str(FAVC_ROOT / spec["glob"]))
            path = Path(hits[0]) if hits else None
        elif "csv_method" in spec and favc is not None:
            rows = favc[
                (favc["method"] == spec["csv_method"])
                & (favc["manipulation_type"] == spec["csv_type"])
            ].sort_values("clip_id")
            for clip_id in rows["clip_id"].head(5):
                path = _favc_path_from_clip_id(clip_id)
                if path is not None:
                    break
        elif "dfdc" in spec and dfdc is not None:
            if spec["dfdc"] == "caught":
                rows = dfdc[(dfdc["label"] == 1) & (dfdc["probability"] > 0.8)]
            elif spec["dfdc"] == "missed":
                rows = dfdc[(dfdc["label"] == 1) & (dfdc["probability"] < 0.02)]
            else:
                rows = dfdc[dfdc["label"] == 0]
            for clip_id in rows.sort_values("clip_id")["clip_id"].head(5):
                candidate = DATA_DIR / "DFDC" / f"{clip_id.split('__', 1)[1]}.mp4"
                if candidate.is_file():
                    path = candidate
                    break
        if path is not None and path.is_file():
            public = {
                key: spec[key] for key in ("id", "title", "dataset", "truth", "kind")
            }
            found[spec["id"]] = {**public, "path": path}
    return found
