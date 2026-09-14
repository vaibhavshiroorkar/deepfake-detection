"""Re-encode every clip identically, real and generated alike.

This is the step that decides whether a collected dataset teaches manipulation or
teaches post-processing, and it is easy to skip because the data looks fine
without it.

The failure it prevents is measured, not hypothetical. The detector in this
repository finds re-encoded in-the-wild clips at 86 percent [49, 97] and raw
generator output at 23 percent [15, 34], intervals that do not overlap. It
learned the encoder. Generated clips arrive as tool downloads at one resolution
and bitrate; real clips arrive from a phone at another. A classifier separates
those two pipelines perfectly and collapses the moment someone re-encodes.

So both sides go through one ffmpeg command with identical settings. The output
is not prettier than the input and is usually worse. That is the point: after
this, resolution, frame rate, codec, bitrate, pixel format and container carry
no information about the label, and metadata is stripped so the tool name cannot
leak either.

    uv run python scripts/normalize_media.py --source data/raw --target data/demo

Layout is preserved, so `raw/fake/veo/x.mp4` becomes `demo/fake/veo/x.mp4`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Deliberately modest. The target is what a clip looks like after a phone
# recorded it and a platform re-encoded it, which is the state the demo will
# meet, not the state a generator downloads in.
DEFAULTS = {
    "width": 640,
    "height": 360,
    "fps": 25,
    "crf": 26,
    "preset": "medium",
    "audio_rate": 16000,
    "audio_bitrate": "64k",
}


def _probe(path: Path) -> dict:
    """Stream summary, so the report can show what actually changed."""
    try:
        # ffprobe from PATH, arguments built here rather than from user input.
        probe_command = [  # noqa: S607
            "ffprobe", "-v", "error", "-show_entries",
            "stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate",
            "-of", "json", str(path),
        ]
        output = subprocess.run(  # noqa: S603
            probe_command,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout
        return json.loads(output)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return {}


def _has_audio(probe: dict) -> bool:
    return any(s.get("codec_type") == "audio" for s in probe.get("streams", []))


def normalize(source: Path, target: Path, settings: dict) -> bool:
    """One clip through the shared pipeline. True when it produced output."""
    target.parent.mkdir(parents=True, exist_ok=True)
    probe = _probe(source)
    # Scale to fit inside the box and pad, rather than stretch: a generator that
    # only emits 16:9 and a phone that only emits 9:16 would otherwise be
    # separable by aspect ratio alone.
    video_filter = (
        f"scale={settings['width']}:{settings['height']}"
        ":force_original_aspect_ratio=decrease,"
        f"pad={settings['width']}:{settings['height']}:(ow-iw)/2:(oh-ih)/2,"
        f"fps={settings['fps']},format=yuv420p"
    )
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
        "-map_metadata", "-1",
        "-vf", video_filter,
        "-c:v", "libx264", "-preset", settings["preset"],
        "-crf", str(settings["crf"]), "-pix_fmt", "yuv420p",
    ]
    if _has_audio(probe):
        command += [
            "-c:a", "aac", "-b:a", settings["audio_bitrate"],
            "-ar", str(settings["audio_rate"]), "-ac", "1",
        ]
    else:
        command += ["-an"]
    command.append(str(target))

    try:
        # ffmpeg from PATH, arguments built here rather than from user input.
        subprocess.run(command, check=True, capture_output=True, timeout=900)  # noqa: S603, S607
    except subprocess.CalledProcessError as error:
        print(f"  {source.name}: ffmpeg failed, {error.stderr.decode()[:120]}")
        return False
    except (OSError, subprocess.SubprocessError) as error:
        print(f"  {source.name}: {type(error).__name__} {error}")
        return False
    return target.is_file()


def main(argv: list[str] | None = None) -> int:
    from deepfake_detection.dashboard.lib import media_kind

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--width", type=int, default=DEFAULTS["width"])
    parser.add_argument("--height", type=int, default=DEFAULTS["height"])
    parser.add_argument("--fps", type=int, default=DEFAULTS["fps"])
    parser.add_argument("--crf", type=int, default=DEFAULTS["crf"])
    parser.add_argument("--skip-existing", action="store_true")
    arguments = parser.parse_args(argv)

    settings = dict(DEFAULTS)
    settings.update(
        width=arguments.width,
        height=arguments.height,
        fps=arguments.fps,
        crf=arguments.crf,
    )

    files = [
        path
        for path in sorted(arguments.source.rglob("*"))
        if path.is_file() and media_kind.classify(path) == media_kind.VIDEO
    ]
    if not files:
        print(f"No video under {arguments.source}")
        return 1
    print(
        f"{len(files)} clips -> {settings['width']}x{settings['height']} "
        f"@{settings['fps']}fps crf{settings['crf']}, metadata stripped\n"
    )

    written = skipped = failed = 0
    for path in files:
        destination = arguments.target / path.relative_to(arguments.source)
        destination = destination.with_suffix(".mp4")
        if arguments.skip_existing and destination.is_file():
            skipped += 1
            continue
        if normalize(path, destination, settings):
            written += 1
        else:
            failed += 1
        if (written + failed) % 25 == 0 and written + failed:
            print(f"  {written + failed} of {len(files)}")

    report = arguments.target / "normalization.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {"settings": settings, "written": written, "skipped": skipped,
             "failed": failed, "source": str(arguments.source).replace("\\", "/")},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {written}, skipped {skipped}, failed {failed}")
    print(f"settings recorded in {report}")
    if failed:
        print("Failed clips are excluded, so check them before training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
