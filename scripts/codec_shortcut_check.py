"""Does the visual model read the video codec instead of the face?

FakeAVCeleb does not encode its categories the same way. Genuine video is H.264
at about 150 kbps. Every FakeVideo-FakeAudio clip (face swap plus wav2lip) is
MPEG-4 Part 2 at about 330 kbps, because that generator wrote its output with a
different encoder. A classifier trained on those labels can separate the
largest fake category by its compression artefacts alone.

The test re-encodes test clips and scores them again with every served visual
checkpoint. Re-encoding genuine clips as H.264 is the control: it changes the
pixels without changing the codec. If the score moves only when the codec
changes, the model is reading the codec.

    uv run python scripts/codec_shortcut_check.py
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

H264 = ("-c:v", "libx264", "-b:v", "160k")
MPEG4 = ("-c:v", "mpeg4", "-b:v", "330k")
ARMS = {
    "genuine": ("RealVideo-RealAudio", None),
    "genuine-as-h264": ("RealVideo-RealAudio", H264),
    "genuine-as-mpeg4": ("RealVideo-RealAudio", MPEG4),
    "faceswap-wav2lip": ("FakeVideo-FakeAudio", None),
    "faceswap-wav2lip-as-h264": ("FakeVideo-FakeAudio", H264),
    "faceswap": ("FakeVideo-RealAudio", None),
    "faceswap-as-mpeg4": ("FakeVideo-RealAudio", MPEG4),
}


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    from deepfake_detection.inference.ensemble import (
        EnsembleMember,
        load_visual_ensemble,
        program_members,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/program-20260906"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data"))
    parser.add_argument("--clips", type=int, default=75)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        default=None,
        help="Visual checkpoints to average. Defaults to the six served ones.",
    )
    arguments = parser.parse_args(argv)

    with (arguments.run_dir / "split" / "test-usable.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_kind: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_kind.setdefault(row["manipulation_type"], []).append(row)
    members = (
        program_members(arguments.run_dir / "checkpoints")
        if arguments.checkpoints is None
        else tuple(
            EnsembleMember(name=path.stem, checkpoint=path)
            for path in arguments.checkpoints
        )
    )
    engine = load_visual_ensemble(
        members, code_version="program-v1", device=arguments.device
    )

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is required for the codec check")
    results = {}
    with tempfile.TemporaryDirectory() as scratch:
        for arm, (kind, codec) in ARMS.items():
            logits = []
            for index, row in enumerate(by_kind[kind][: arguments.clips]):
                path = arguments.dataset_root / row["video_path"].replace("\\", "/")
                if codec is not None:
                    target = Path(scratch) / f"{arm}-{index}.mp4"
                    subprocess.run(  # noqa: S603
                        # Fixed codec options and dataset paths, passed without a shell.
                        [
                            ffmpeg,
                            "-v",
                            "error",
                            "-y",
                            "-i",
                            str(path),
                            *codec,
                            "-c:a",
                            "copy",
                            str(target),
                        ],
                        check=True,
                    )
                    path = target
                prediction = engine.predict(path)
                if prediction.result.probability is not None:
                    logits.append(prediction.result.branch_logits)
            names = list(logits[0])
            matrix = np.array([[item[name] for name in names] for item in logits])
            ensemble = 1 / (1 + np.exp(-matrix.mean(axis=1)))
            results[arm] = {
                "category": kind,
                "reencoded_with": " ".join(codec) if codec else None,
                "clips": len(logits),
                "ensemble_called_fake": float((ensemble >= 0.5).mean()),
                "members_called_fake": {
                    name: float((matrix[:, i] >= 0).mean())
                    for i, name in enumerate(names)
                },
            }
            print(arm, results[arm]["ensemble_called_fake"], flush=True)

    output = arguments.output or (
        arguments.run_dir / "evaluation" / "codec-shortcut-check.json"
    )
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
