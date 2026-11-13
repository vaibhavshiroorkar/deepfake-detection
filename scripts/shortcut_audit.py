"""How far can a classifier get without looking at a face or listening to a voice?

For every clip in a manifest this reads what the file says about itself
(container, codecs, bitrates, frame rate, size, duration) and three things the
audio says about its edges (leading and trailing digital silence, loudness).
None of these is evidence of manipulation. Each one is a way a generator's
pipeline differs from the camera's.

It then scores each cue alone, and all cues together, as a real-or-fake
classifier. Folds are grouped by source identity, so a cue cannot win by
recognising a person. The best of these is the dataset's shortcut ceiling: a
detector that does not beat it has not shown that it sees the manipulation.

The audit runs separately for the clip label and, where the manifest has them,
for the video and audio labels, because a cue can give away which modality was
edited without giving away whether the clip was.

    uv run python scripts/shortcut_audit.py --manifest data/manifests/fakeavceleb.csv \
        --dataset FakeAVCeleb --output results/shortcut-audit/fakeavceleb
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
# Digital silence: below -60 dBFS. Smeu et al. (CVPR 2025) found generated
# FakeAVCeleb and AV-Deepfake1M clips start with a few tens of milliseconds of it.
SILENCE = 10 ** (-60 / 20)
NUMERIC = (
    "duration",
    "overall_bitrate",
    "video_bitrate",
    "fps",
    "width",
    "height",
    "audio_bitrate",
    "audio_sample_rate",
    "leading_silence_ms",
    "trailing_silence_ms",
    "loudness_db",
)
CATEGORICAL = ("container", "video_codec", "pixel_format", "audio_codec", "has_audio")


def _number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _rate(text: str | None) -> float:
    if not text or "/" not in text:
        return _number(text)
    top, bottom = text.split("/", 1)
    return _number(top) / _number(bottom) if _number(bottom) else float("nan")


def probe(path: Path) -> dict[str, object]:
    run = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if run.returncode != 0:
        return {"error": run.stderr.strip()[:200]}
    info = json.loads(run.stdout)
    video = next((s for s in info["streams"] if s.get("codec_type") == "video"), {})
    audio = next((s for s in info["streams"] if s.get("codec_type") == "audio"), {})
    form = info.get("format", {})
    return {
        "container": form.get("format_name", ""),
        "duration": _number(form.get("duration")),
        "overall_bitrate": _number(form.get("bit_rate")),
        "video_codec": video.get("codec_name", ""),
        "pixel_format": video.get("pix_fmt", ""),
        "video_bitrate": _number(video.get("bit_rate")),
        "fps": _rate(video.get("avg_frame_rate")),
        "width": _number(video.get("width")),
        "height": _number(video.get("height")),
        "audio_codec": audio.get("codec_name", ""),
        "audio_bitrate": _number(audio.get("bit_rate")),
        "audio_sample_rate": _number(audio.get("sample_rate")),
        "has_audio": "yes" if audio else "no",
    }


def edges(path: Path) -> dict[str, float]:
    run = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0?", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
        capture_output=True,
        check=False,
    )
    wave = np.frombuffer(run.stdout, np.float32)
    if run.returncode != 0 or wave.size == 0:
        return {"leading_silence_ms": float("nan"), "trailing_silence_ms": float("nan"),
                "loudness_db": float("nan")}
    loud = np.flatnonzero(np.abs(wave) > SILENCE)
    if loud.size == 0:
        lead = trail = wave.size
    else:
        lead, trail = int(loud[0]), int(wave.size - 1 - loud[-1])
    rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
    return {
        "leading_silence_ms": 1000 * lead / SAMPLE_RATE,
        "trailing_silence_ms": 1000 * trail / SAMPLE_RATE,
        "loudness_db": 20 * np.log10(max(rms, 1e-10)),
    }


def describe(job: tuple[str, Path]) -> dict[str, object]:
    clip_id, path = job
    return {"clip_id": clip_id, **probe(path), **edges(path)}


def _cue_auc(values: np.ndarray, labels: np.ndarray, groups: np.ndarray, categorical: bool,
             folds: int, seed: int) -> float:
    """Out-of-fold AUC of one cue, folds grouped by source identity.

    A numeric cue is ranked as is, then flipped if it points the other way, so
    the score is the separation it carries whichever direction that runs. A
    categorical cue is turned into the fake rate of its category, learnt on the
    other folds, so a category seen only in one fold carries no information.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    if len(set(labels)) < 2:
        return float("nan")
    if not categorical:
        keep = np.isfinite(values.astype(float))
        if keep.sum() < 10 or len(set(labels[keep])) < 2:
            return float("nan")
        auc = roc_auc_score(labels[keep], values[keep].astype(float))
        return float(max(auc, 1 - auc))
    scores = np.full(len(labels), np.nan)
    prior = labels.mean()
    for train, test in GroupKFold(n_splits=folds).split(values, labels, groups):
        rates: dict[object, float] = {}
        for category in set(values[train]):
            mask = values[train] == category
            rates[category] = labels[train][mask].mean()
        scores[test] = [rates.get(v, prior) for v in values[test]]
    return float(roc_auc_score(labels, scores))


def _ceiling(rows: list[dict], labels: np.ndarray, groups: np.ndarray, folds: int, seed: int) -> float:
    """All cues together in a gradient-boosted model: the shortcut ceiling."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    if len(set(labels)) < 2:
        return float("nan")
    columns = []
    for name in NUMERIC:
        columns.append([_number(r.get(name)) for r in rows])
    for name in CATEGORICAL:
        vocabulary = {v: i for i, v in enumerate(sorted({str(r.get(name, "")) for r in rows}))}
        columns.append([vocabulary[str(r.get(name, ""))] for r in rows])
    features = np.array(columns, dtype=float).T
    categorical = [False] * len(NUMERIC) + [True] * len(CATEGORICAL)
    scores = np.zeros(len(labels))
    for train, test in GroupKFold(n_splits=folds).split(features, labels, groups):
        model = HistGradientBoostingClassifier(
            categorical_features=categorical, max_iter=200, random_state=seed
        )
        model.fit(features[train], labels[train])
        scores[test] = model.predict_proba(features[test])[:, 1]
    return float(roc_auc_score(labels, scores))


def summarise(rows: list[dict], manifest: dict[str, dict], folds: int, seed: int) -> dict:
    groups = np.array([manifest[r["clip_id"]]["source"] for r in rows])
    report: dict[str, object] = {}
    for label_name in ("clip_fake", "video_fake", "audio_fake"):
        if label_name not in next(iter(manifest.values())):
            continue
        labels = np.array([int(manifest[r["clip_id"]][label_name]) for r in rows])
        if len(set(labels)) < 2:
            continue
        cues = {}
        for name in NUMERIC:
            values = np.array([_number(r.get(name)) for r in rows])
            cues[name] = _cue_auc(values, labels, groups, False, folds, seed)
        for name in CATEGORICAL:
            values = np.array([str(r.get(name, "")) for r in rows], dtype=object)
            cues[name] = _cue_auc(values, labels, groups, True, folds, seed)
        report[label_name] = {
            "positives": int(labels.sum()),
            "negatives": int((1 - labels).sum()),
            "cue_auc": {k: round(v, 4) for k, v in sorted(cues.items(), key=lambda kv: -np.nan_to_num(kv[1]))},
            "shortcut_ceiling_auc": round(_ceiling(rows, labels, groups, folds, seed), 4),
        }
    by_class: dict[str, dict] = {}
    for row in rows:
        key = manifest[row["clip_id"]].get("manipulation_type", "unknown")
        entry = by_class.setdefault(key, {"clips": 0, "video_codec": {}, "container": {}, "audio_codec": {}})
        entry["clips"] += 1
        for name in ("video_codec", "container", "audio_codec"):
            value = str(row.get(name, ""))
            entry[name][value] = entry[name].get(value, 0) + 1
    report["codec_census"] = by_class
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Stratified sample size for very large sets")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    arguments = parser.parse_args(argv)

    with arguments.manifest.open(newline="", encoding="utf-8-sig") as handle:
        manifest = {row["clip_id"]: row for row in csv.DictReader(handle)}
    # Older manifests (DFDC, LAV-DF windows) carry only manipulation_type.
    for row in manifest.values():
        kind = row.get("manipulation_type", "")
        row.setdefault("video_fake", str(int(kind.startswith("FakeVideo-"))))
        row.setdefault("audio_fake", str(int(kind.endswith("-FakeAudio"))))
        row.setdefault("clip_fake", str(int(row["video_fake"] == "1" or row["audio_fake"] == "1")))
    ids = sorted(manifest)
    if arguments.limit and len(ids) > arguments.limit:
        generator = np.random.default_rng(arguments.seed)
        by_type: dict[str, list[str]] = {}
        for clip_id in ids:
            by_type.setdefault(manifest[clip_id].get("manipulation_type", ""), []).append(clip_id)
        share = arguments.limit / len(ids)
        ids = sorted(
            clip_id
            for members in by_type.values()
            for clip_id in generator.choice(members, max(1, round(len(members) * share)), replace=False)
        )
    jobs = [(i, arguments.data_dir / manifest[i]["video_path"].replace("\\", "/")) for i in ids]

    arguments.output.mkdir(parents=True, exist_ok=True)
    clips_path = arguments.output / "clips.csv"
    done: dict[str, dict] = {}
    if clips_path.exists():
        with clips_path.open(newline="", encoding="utf-8") as handle:
            done = {row["clip_id"]: row for row in csv.DictReader(handle)}
    pending = [job for job in jobs if job[0] not in done]
    fields = ("clip_id", "error", *CATEGORICAL, *NUMERIC)
    with clips_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not done:
            writer.writeheader()
        with ProcessPoolExecutor(arguments.workers) as pool:
            for index, row in enumerate(pool.map(describe, pending, chunksize=16), start=1):
                writer.writerow(row)
                done[row["clip_id"]] = row
                if index % 1000 == 0:
                    handle.flush()
                    print(f"{index}/{len(pending)}", flush=True)

    rows = [done[i] for i in ids if i in done and not done[i].get("error")]
    errors = sum(1 for i in ids if done.get(i, {}).get("error"))
    report = {
        "dataset": arguments.dataset,
        "manifest": str(arguments.manifest),
        "clips_requested": len(ids),
        "clips_read": len(rows),
        "clips_failed": errors,
        "silence_threshold_dbfs": -60,
        "folds": arguments.folds,
        "fold_grouping": "source identity",
        **summarise(rows, manifest, arguments.folds, arguments.seed),
    }
    (arguments.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for label_name in ("clip_fake", "video_fake", "audio_fake"):
        if label_name in report:
            entry = report[label_name]
            top = list(entry["cue_auc"].items())[:4]
            print(f"{label_name}: ceiling {entry['shortcut_ceiling_auc']}  top cues {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
