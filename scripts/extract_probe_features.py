"""Frozen-encoder features over contiguous windows, cached per clip.

The first half of the leave-one-generator-out pipeline. It reads a folder laid
out as `real/` and `fake/<generator>/`, decodes contiguous windows of full
frames, pushes them through a frozen encoder, and writes one feature vector per
clip. The probe in `train_logo_probe.py` then trains on those in seconds, which
is what makes seven leave-one-out arms cheap: the expensive part happens once.

Two choices here are the measured ones rather than the conventional ones.

**Contiguous windows, not frames spread across the clip.** Sampling 16 frames
across a whole video is what taught the existing system that motion means
authenticity, correlation -0.316 among manipulated clips. Windows preserve local
temporal structure and give a per-window score that doubles as crude
localization.

**Full frames, not face crops.** A face-first pipeline cannot answer on
generated video without a person in it, and that is not hypothetical: on 314
Veo 3 clips the face-dependent branches answered on 52.5 percent against the
audio branch's 100 percent.

The encoder is never fine-tuned. Frozen frontier encoders with light probes are
what generalize to unseen generators, and fine-tuning is precisely what converts
a general representation into a generator-specific one.

    uv run python scripts/extract_probe_features.py --folder data/df26-scored
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from research_artifacts import sha256

# Share preprocessing with serving. Historical PE/CLIP results require reruns.
from deepfake_detection.inference.frozen_features import (  # noqa: E402
    ENCODERS,
    embed_windows,
    load_encoder,
    normalisation,
    read_windows,
)


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    from deepfake_detection import media_kind
    from deepfake_detection.inference import frozen_features

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--encoder", choices=sorted(ENCODERS), default="pe")
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--window-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only complete features with matching inputs and configuration.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing artifact without loading encoder weights.",
    )
    arguments = parser.parse_args(argv)
    output = arguments.output or arguments.folder / f"features-{arguments.encoder}.npz"
    meta = output.with_suffix(".json")
    previous = None
    if output.exists() or meta.exists():
        if not (arguments.resume or arguments.validate_only):
            raise FileExistsError(f"Evidence exists at {output}; choose a new output")
        if not (output.is_file() and meta.is_file()):
            raise ValueError(
                f"Incomplete feature artifact at {output}; choose a new output"
            )
        previous = json.loads(meta.read_text(encoding="utf-8"))
        if not previous.get("normalisation"):
            raise ValueError(
                "Legacy features have no normalisation metadata; re-extract to a new path"
            )
    elif arguments.validate_only:
        raise FileNotFoundError(output)

    files: list[tuple[Path, str, int]] = []
    real_root = arguments.folder / "real"
    if real_root.is_dir():
        found = [
            p
            for p in sorted(real_root.rglob("*"))
            if p.is_file() and media_kind.classify(p) == media_kind.VIDEO
        ]
        files += [
            (p, "real", 0)
            for p in (found[: arguments.limit] if arguments.limit else found)
        ]
    fake_root = arguments.folder / "fake"
    if fake_root.is_dir():
        for source in sorted(p for p in fake_root.iterdir() if p.is_dir()):
            found = [
                p
                for p in sorted(source.rglob("*"))
                if p.is_file() and media_kind.classify(p) == media_kind.VIDEO
            ]
            chosen = found[: arguments.limit] if arguments.limit else found
            files += [(p, source.name, 1) for p in chosen]
    if not files:
        print(f"No video under {arguments.folder}, expected real/ and fake/<gen>/")
        return 1

    configuration = {
        "encoder": arguments.encoder,
        "encoder_model": ENCODERS[arguments.encoder][0],
        "normalisation": normalisation(arguments.encoder),
        "size": ENCODERS[arguments.encoder][1],
        "windows": arguments.windows,
        "window_length": arguments.window_length,
        "limit": arguments.limit,
        "batch_size": arguments.batch_size,
        "folder": str(arguments.folder.resolve()).replace("\\", "/"),
        "implementation_sha256": {
            "extractor": sha256(Path(__file__)),
            "frozen_features": sha256(Path(frozen_features.__file__)),
        },
        "input_sha256": {
            str(path.relative_to(arguments.folder)).replace("\\", "/"): sha256(path)
            for path, _, _ in files
        },
    }
    # JSON normalizes tuples from model configuration into lists.
    configuration = json.loads(json.dumps(configuration))
    if previous is not None:
        if previous.get("configuration") != configuration:
            raise ValueError(
                "Feature configuration or input digest mismatch; use a new output"
            )
        if previous.get("feature_sha256") != sha256(output):
            raise ValueError("Feature digest mismatch; use a new output")
        print(f"verified complete features: {output}")
        return 0

    # Through the shared loader, so a timm name and an open_clip name behave the
    # same and this script cannot drift from the served gate's embedding.
    model, size = load_encoder(arguments.encoder, device=arguments.device)
    print(
        f"{len(files)} clips, {arguments.encoder} at {size}px, "
        f"{arguments.windows} windows of {arguments.window_length}",
        flush=True,
    )

    vectors: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    names: list[str] = []
    skipped = 0
    for index, (path, group, label) in enumerate(files, start=1):
        windows = read_windows(path, arguments.windows, arguments.window_length, size)
        if windows is None:
            skipped += 1
            continue
        # Mean over frames within a window, then over windows, and the spread
        # across windows beside it: a generated clip is often steadier across
        # time than a real one, and averaging alone would discard that.
        vector = embed_windows(
            model, windows, arguments.device, batch_size=arguments.batch_size
        )
        vectors.append(vector)
        labels.append(label)
        groups.append(group)
        names.append(str(path.relative_to(arguments.folder)).replace("\\", "/"))
        if index % 25 == 0:
            print(f"  {index} of {len(files)}", flush=True)

    if not vectors:
        print("Nothing could be decoded.")
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        np.savez_compressed(
            handle,
            features=np.stack(vectors).astype(np.float32),
            labels=np.asarray(labels, dtype=np.int64),
            groups=np.asarray(groups),
            files=np.asarray(names),
        )
    counts: dict[str, int] = {}
    for group in groups:
        counts[group] = counts.get(group, 0) + 1
    print(f"\n{len(vectors)} clips embedded, {skipped} undecodable")
    for group, count in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"  {group:28} {count:5}")
    print(f"wrote {output}")
    metadata = (
        json.dumps(
            {
                "encoder": arguments.encoder,
                "encoder_model": ENCODERS[arguments.encoder][0],
                # Recorded because features extracted before 2026-10-22 used
                # ImageNet statistics for every encoder, which was wrong for
                # PE and CLIP. A file without this key is one of those.
                "normalisation": configuration["normalisation"],
                "configuration": configuration,
                "feature_sha256": sha256(output),
                "size": size,
                "windows": arguments.windows,
                "window_length": arguments.window_length,
                "dimension": int(vectors[0].shape[0]),
                "counts": counts,
                "skipped": skipped,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with meta.open("x", encoding="utf-8") as handle:
        handle.write(metadata)
    return 0


if __name__ == "__main__":
    sys.exit(main())
