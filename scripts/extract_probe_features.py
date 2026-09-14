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

# Cached locally and therefore usable offline. Perception Encoder is the better
# choice on this task by about 15 AUROC points, and needs `open_clip_torch`,
# which is not installed here. Pass --encoder to switch once it is.
ENCODERS = {
    "dinov3": ("vit_small_patch16_dinov3.lvd1689m", 224),
    "dinov3-b": ("vit_base_patch16_dinov3.lvd1689m", 224),
}


def _windows(total: int, count: int, length: int) -> list[int]:
    """Start indices for `count` contiguous windows spread across the clip."""
    if total <= length:
        return [0]
    if count == 1:
        return [max(0, (total - length) // 2)]
    span = total - length
    return [round(span * i / (count - 1)) for i in range(count)]


def _read_windows(path: Path, count: int, length: int, size: int):
    """Decoded windows as uint8 arrays, [window, frame, H, W, 3]."""
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = length * count
    starts = _windows(total, count, length)
    wanted = {index for start in starts for index in range(start, start + length)}

    frames: dict[int, np.ndarray] = {}
    index = 0
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        if index in wanted:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames[index] = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        index += 1
        if index > max(wanted, default=0):
            break
    capture.release()
    if not frames:
        return None

    ordered = sorted(frames)
    stacked = []
    for start in starts:
        picked = [frames[i] for i in range(start, start + length) if i in frames]
        if not picked:
            picked = [frames[ordered[0]]]
        while len(picked) < length:
            picked.append(picked[-1])
        stacked.append(np.stack(picked))
    return np.stack(stacked)


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    import timm
    import torch

    from deepfake_detection.dashboard.lib import media_kind
    from deepfake_detection.preprocessing.ops.constants import (
        IMAGENET_MEAN,
        IMAGENET_STD,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--encoder", choices=sorted(ENCODERS), default="dinov3")
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--window-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)

    files: list[tuple[Path, str, int]] = []
    real_root = arguments.folder / "real"
    if real_root.is_dir():
        found = [
            p for p in sorted(real_root.rglob("*"))
            if p.is_file() and media_kind.classify(p) == media_kind.VIDEO
        ]
        files += [(p, "real", 0) for p in (found[: arguments.limit] if arguments.limit else found)]
    fake_root = arguments.folder / "fake"
    if fake_root.is_dir():
        for source in sorted(p for p in fake_root.iterdir() if p.is_dir()):
            found = [
                p for p in sorted(source.rglob("*"))
                if p.is_file() and media_kind.classify(p) == media_kind.VIDEO
            ]
            chosen = found[: arguments.limit] if arguments.limit else found
            files += [(p, source.name, 1) for p in chosen]
    if not files:
        print(f"No video under {arguments.folder}, expected real/ and fake/<gen>/")
        return 1

    name, size = ENCODERS[arguments.encoder]
    model = timm.create_model(name, pretrained=True, num_classes=0)
    model = model.to(arguments.device).eval()
    mean = torch.tensor(IMAGENET_MEAN, device=arguments.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=arguments.device).view(1, 3, 1, 1)
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
        windows = _read_windows(
            path, arguments.windows, arguments.window_length, size
        )
        if windows is None:
            skipped += 1
            continue
        flat = torch.from_numpy(windows.reshape(-1, size, size, 3))
        flat = flat.permute(0, 3, 1, 2).float().div_(255.0)
        pieces = []
        with torch.inference_mode():
            for start in range(0, flat.shape[0], arguments.batch_size):
                batch = flat[start : start + arguments.batch_size].to(arguments.device)
                batch = (batch - mean) / std
                pieces.append(model(batch).float().cpu())
        embedded = torch.cat(pieces).reshape(windows.shape[0], windows.shape[1], -1)
        # Mean over frames within a window, then over windows, and the standard
        # deviation across windows beside it: a generated clip is often steadier
        # across time than a real one, and averaging alone would discard that.
        per_window = embedded.mean(dim=1)
        vector = torch.cat([per_window.mean(dim=0), per_window.std(dim=0)])
        vectors.append(vector.numpy())
        labels.append(label)
        groups.append(group)
        names.append(str(path.relative_to(arguments.folder)).replace("\\", "/"))
        if index % 25 == 0:
            print(f"  {index} of {len(files)}", flush=True)

    if not vectors:
        print("Nothing could be decoded.")
        return 1
    output = arguments.output or arguments.folder / f"features-{arguments.encoder}.npz"
    np.savez_compressed(
        output,
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
    meta = output.with_suffix(".json")
    meta.write_text(
        json.dumps(
            {
                "encoder": name,
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
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
