"""How much degradation does the detector tolerate before it stops working?

The threat model lists recompression, resizing and added noise as things an
adaptive black-box attacker will do, and names corruption tests as the check
against the codec shortcut: a model that learned a corpus's encoding rather than
its manipulations should fall apart when the encoding changes.

**What this measures, precisely.** The corruption is applied to the cached view,
so the face track is held fixed and only the pixels the model sees are degraded.
That isolates the classifier from the detector. It answers "given the face was
found and cropped correctly, how much degradation does the classifier tolerate",
which is the question about the classifier.

**What it does not measure.** Corrupting the source video and re-running the
pipeline is a different and harder experiment: heavy compression also makes
faces harder to find, so the real-world loss includes detector failures and
abstentions this cannot see. JPEG artifacts introduced at 224x224 are also not
the artifacts of compressing a 720p source and then cropping. Every number here
is therefore an optimistic bound on corruption robustness, not an estimate of
it, and it should be reported that way.

The view is de-normalized back through the ImageNet statistics the preprocessor
applied, corrupted as 8-bit pixels, then re-normalized, so the model receives a
tensor built exactly the way its training tensors were.

    uv run python scripts/run_corruption_eval.py --run-dir runs/program-20260906
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MEAN = (0.485, 0.456, 0.406)
STANDARD_DEVIATION = (0.229, 0.224, 0.225)


def _conditions(argv_branch: str) -> tuple[tuple[str, str, dict], ...]:
    """Name, kind and parameters for each arm, clean first.

    The levels are the ones the threat model implies rather than a sweep: a
    recompression a platform would apply, a resize to a small player, and noise
    at a level a phone microphone would produce.
    """
    if argv_branch == "visual":
        return (
            ("clean", "none", {}),
            ("jpeg-75", "jpeg", {"jpeg_quality": 75}),
            ("jpeg-50", "jpeg", {"jpeg_quality": 50}),
            ("jpeg-30", "jpeg", {"jpeg_quality": 30}),
            ("scale-0.50", "scale", {"scale": 0.5}),
            ("scale-0.25", "scale", {"scale": 0.25}),
        )
    return (
        ("clean", "none", {}),
        ("noise-20db", "noise", {"snr_db": 20.0}),
        ("noise-10db", "noise", {"snr_db": 10.0}),
        ("noise-5db", "noise", {"snr_db": 5.0}),
        ("silence-0.5s", "silence", {"seconds": 0.5}),
    )


def _corrupt_visual(view, kind: str, parameters: dict):
    """Round-trip the normalized crop through 8-bit pixels and back."""
    import numpy as np

    from deepfake_detection.evaluation.corruptions import (
        compress_video_frames,
        degrade_resolution,
    )

    if kind == "none":
        return view
    mean = np.asarray(MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    deviation = np.asarray(STANDARD_DEVIATION, dtype=np.float32).reshape(1, 3, 1, 1)
    rgb = np.clip(view * deviation + mean, 0.0, 1.0)
    # The corruption helpers take [time, height, width, 3]; the cache holds
    # [time, 3, height, width].
    frames = (rgb.transpose(0, 2, 3, 1) * 255.0).round().astype(np.uint8)
    if kind == "jpeg":
        frames = compress_video_frames(frames, **parameters)
    elif kind == "scale":
        frames = degrade_resolution(frames, **parameters)
    else:
        raise ValueError(f"Unsupported visual corruption {kind}")
    restored = frames.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    return (restored - mean) / deviation


def _corrupt_audio(view, kind: str, parameters: dict, *, sample_rate: int):
    from deepfake_detection.evaluation.corruptions import (
        add_audio_noise,
        prepend_silence,
    )

    if kind == "none":
        return view
    if kind == "noise":
        return add_audio_noise(view, seed=17, **parameters)
    if kind == "silence":
        return prepend_silence(
            view, silence_samples=round(parameters["seconds"] * sample_rate)
        )
    raise ValueError(f"Unsupported audio corruption {kind}")


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    from deepfake_detection.data.manifest import load_manifest
    from deepfake_detection.views.cache_store import CacheStore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--branch", choices=("visual", "audio"), default="visual")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dataset", default="FakeAVCeleb")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Score only the first N clips. 0 scores every clip.",
    )
    arguments = parser.parse_args(argv)
    run = arguments.run_dir
    branch = arguments.branch

    checkpoint = arguments.checkpoint or (
        run / "checkpoints" / f"final-{branch}-seed17.pt"
    )
    manifest = arguments.manifest or (
        run
        / "split"
        / ("test-usable.csv" if branch == "visual" else "test-audio-usable.csv")
    )
    if not checkpoint.is_file():
        print(f"No checkpoint at {checkpoint}", file=sys.stderr)
        return 1
    if not manifest.is_file():
        print(f"No manifest at {manifest}", file=sys.stderr)
        return 1

    from deepfake_detection.data.cache_index import read_cache_index

    index = read_cache_index(run / "cache-index.csv")
    records = [
        record
        for record in load_manifest(manifest, dataset=arguments.dataset).records
        if record.clip_id in index
    ]
    if arguments.limit:
        # Shuffled with a fixed seed before truncating. The manifests are
        # grouped by label, so a prefix of any size is one class and has no
        # ranking metric at all.
        order = np.random.default_rng(arguments.seed).permutation(len(records))
        records = [records[index] for index in order[: arguments.limit]]
    if not records:
        print("No manifest rows resolve to a cache entry", file=sys.stderr)
        return 1

    from deepfake_detection.branches.registry import load_branch

    # Through the registry, so the branch is rebuilt as it was trained and its
    # metadata is checked rather than skipped.
    model, _ = load_branch(checkpoint, branch)
    view_name = "visual_view" if branch == "visual" else "audio_view"
    model.to(arguments.device).eval()

    store = CacheStore(run)
    conditions = _conditions(branch)
    print(
        f"{branch} branch, {len(records):,} clips, "
        f"{len(conditions)} conditions, checkpoint {checkpoint.name}\n"
    )

    # Views are read once and corrupted in memory for every arm, so each arm
    # sees identical source pixels and the decompression cost is paid once.
    labels: list[int] = []
    views: list = []
    for record in records:
        prepared = store.load(index[record.clip_id], views=(view_name,))
        view = getattr(prepared, view_name)
        if view is None:
            continue
        views.append(view)
        labels.append(
            int(record.video_fake if branch == "visual" else record.audio_fake)
        )
    if len(set(labels)) < 2:
        print("Scored set has a single class, no ranking metric exists")
        return 1
    print(
        f"loaded {len(views):,} views (fake {sum(labels)}, real "
        f"{len(labels) - sum(labels)})\n"
    )

    report: dict[str, dict] = {}
    for name, kind, parameters in conditions:
        scores: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(views), arguments.batch_size):
                chunk = views[start : start + arguments.batch_size]
                corrupted = [
                    _corrupt_visual(view, kind, parameters)
                    if branch == "visual"
                    else _corrupt_audio(
                        view, kind, parameters, sample_rate=arguments.sample_rate
                    )
                    for view in chunk
                ]
                batch = (
                    torch.from_numpy(np.stack(corrupted)).float().to(arguments.device)
                )
                scores.extend(
                    torch.sigmoid(model(batch).logits).detach().cpu().tolist()
                )
        auc = float(roc_auc_score(labels, scores))
        accused = sum(
            1
            for score, label in zip(scores, labels, strict=True)
            if label == 0 and score >= 0.5
        )
        missed = sum(
            1
            for score, label in zip(scores, labels, strict=True)
            if label == 1 and score < 0.5
        )
        report[name] = {
            "kind": kind,
            "parameters": parameters,
            "roc_auc": auc,
            "delta_against_clean": auc - report["clean"]["roc_auc"]
            if "clean" in report
            else 0.0,
            "genuine_called_fake": accused,
            "fake_called_genuine": missed,
            "clips": len(labels),
        }
        entry = report[name]
        print(
            f"  {name:14} ROC-AUC {auc:.4f}  "
            f"delta {entry['delta_against_clean']:+.4f}  "
            f"accused {accused:>4}/{len(labels) - sum(labels):<4}  "
            f"missed {missed:>5}/{sum(labels)}"
        )

    worst = min(report.values(), key=lambda entry: entry["roc_auc"])
    print(
        f"\nworst arm falls {worst['delta_against_clean']:+.4f} "
        f"to {worst['roc_auc']:.4f}"
    )

    payload = {
        "branch": branch,
        "checkpoint": str(checkpoint).replace("\\", "/"),
        "manifest": str(manifest).replace("\\", "/"),
        "clips": len(labels),
        "scope": (
            "Corruption applied to the cached view, so the face track is held "
            "fixed. This is an optimistic bound: corrupting the source and "
            "re-running the pipeline would also cause detector failures and "
            "abstentions that this cannot see."
        ),
        "conditions": report,
    }
    out = arguments.output or (run / "evaluation" / f"corruption-{branch}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
