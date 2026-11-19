"""The website's dataset samples: a few clips per dataset and kind of fake, from the held-out test sets.

    .venv/Scripts/python.exe scripts/build_samples.py [--per-kind 8]

Clips come from the test sets the site's lines and rates are measured on (scripts/build_web_calibration.py),
so a sample shows how the site does on footage it never trained on, plus our own Veo 3 clips. DF26 is left
out: its licence forbids publishing any part. Vaibhav chose (11-20) to show FakeAVCeleb, DFDC, LAV-DF and
HiFi-AVDF clips on the public site.

Each sample keeps its original file for the check (so a file's own AI label still counts) and gets a browser
playable H.264 preview (some FakeAVCeleb clips use a codec browsers cannot play). Writes data/samples/
catalogue.json and data/samples/previews/<id>.mp4 (data/ is git-ignored). Fixed seed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
DATA = ROOT / "data"
OUT = DATA / "samples"

FAV_KIND = {
    ("RealVideo-RealAudio", "real"): "Real",
    ("FakeVideo-RealAudio", "faceswap"): "Face swap",
    ("FakeVideo-RealAudio", "fsgan"): "Face swap",
    ("FakeVideo-RealAudio", "wav2lip"): "Lip sync",
    ("RealVideo-FakeAudio", "rtvc"): "Voice clone",
    ("FakeVideo-FakeAudio", "fsgan-wav2lip"): "Face swap and voice clone",
    ("FakeVideo-FakeAudio", "faceswap-wav2lip"): "Face swap and voice clone",
    ("FakeVideo-FakeAudio", "wav2lip"): "Lip sync and voice clone",
}
LAVDF_KIND = {
    "RealVideo-RealAudio": "Real",
    "FakeVideo-RealAudio": "Word edit (picture)",
    "RealVideo-FakeAudio": "Word edit (voice)",
    "FakeVideo-FakeAudio": "Word edit (picture and voice)",
}
GENERATORS = {
    "kling2.5": "Kling 2.5",
    "pixverseV5": "PixVerse V5",
    "seedance1.0": "Seedance 1.0",
    "sora2": "Sora 2",
    "veo3.1": "Veo 3.1",
    "wan2.5": "Wan 2.5",
}


def manifest(name: str) -> dict[str, dict]:
    with open(DATA / "manifests" / f"{name}.csv", encoding="utf-8", newline="") as f:
        return {r["clip_id"]: r for r in csv.DictReader(f)}


def candidates() -> list[dict]:
    """Every test-set clip with its dataset, kind, label and original file."""
    from build_web_calibration import CORPORA, rows_for

    test = {
        c: {r["clip_id"] for r in rows_for(c, b, s, {})}
        for c, (b, s) in CORPORA.items()
    }
    out = []
    fav = manifest("fakeavceleb")
    for clip in test["FakeAVCeleb"]:
        r = fav.get(clip)
        kind = r and FAV_KIND.get((r["manipulation_type"], r["method"]))
        if kind:
            out.append(
                {
                    "dataset": "FakeAVCeleb",
                    "kind": kind,
                    "path": DATA / r["video_path"].replace("\\", "/"),
                }
            )
    dfdc = manifest("dfdc-neutral")
    for clip in test["DFDC"]:
        r = dfdc.get(clip)
        if r:
            kind = (
                "Real"
                if r["manipulation_type"] == "RealVideo-RealAudio"
                else "Face swap"
            )
            out.append(
                {
                    "dataset": "DFDC",
                    "kind": kind,
                    "path": DATA / "DFDC" / Path(r["video_path"]).name,
                }
            )
    lav = manifest("lavdf-files")
    for clip in test["LAV-DF"]:
        r = lav.get(clip)
        if r:
            out.append(
                {
                    "dataset": "LAV-DF",
                    "kind": LAVDF_KIND[r["manipulation_type"]],
                    "path": DATA / r["video_path"],
                }
            )
    hifi = manifest("hifiavdf")
    for clip in test["HiFi-AVDF"]:
        r = hifi.get(clip)
        if r:
            gen = clip.split("__")[1]
            kind = (
                "Real"
                if r["manipulation_type"] == "RealVideo-RealAudio"
                else f"AI-generated ({GENERATORS[gen]})"
            )
            out.append(
                {"dataset": "HiFi-AVDF", "kind": kind, "path": DATA / r["video_path"]}
            )
    for path in sorted((DATA / "generated" / "raw" / "fake" / "veo3").glob("*.mp4")):
        out.append(
            {"dataset": "Our Veo 3 clips", "kind": "AI-generated (Veo 3)", "path": path}
        )
    return [c for c in out if c["path"].is_file()]


def preview(source: Path, target: Path) -> bool:
    run = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-t",
            "60",
            "-vf",
            "scale=-2:'min(720,ih)'",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "26",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-movflags",
            "+faststart",
            str(target),
        ],
        capture_output=True,
        timeout=300,
        check=False,
    )
    return run.returncode == 0 and target.is_file()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-kind", type=int, default=8)
    args = parser.parse_args()
    groups: dict = {}
    for c in candidates():
        groups.setdefault((c["dataset"], c["kind"]), []).append(c)
    (OUT / "previews").mkdir(parents=True, exist_ok=True)
    catalogue = []
    for (dataset, kind), clips in sorted(groups.items()):
        rng = random.Random(f"41-{dataset}-{kind}")  # noqa: S311 - a seeded sample, not security
        clips = sorted(clips, key=lambda c: str(c["path"]))
        rng.shuffle(clips)
        for n, clip in enumerate(clips[: args.per_kind], 1):
            sid = hashlib.sha256(
                str(clip["path"].relative_to(DATA)).encode()
            ).hexdigest()[:12]
            target = OUT / "previews" / f"{sid}.mp4"
            if not target.is_file() and not preview(clip["path"], target):
                print("skipped, no preview:", clip["path"].name, flush=True)
                continue
            catalogue.append(
                {
                    "id": sid,
                    "dataset": dataset,
                    "kind": kind,
                    "label": "real" if kind == "Real" else "fake",
                    "title": f"{kind} {n}",
                    "source": str(clip["path"].relative_to(DATA)).replace("\\", "/"),
                }
            )
        print(dataset, kind, min(len(clips), args.per_kind), flush=True)
    (OUT / "catalogue.json").write_text(
        json.dumps(catalogue, indent=1) + "\n", encoding="utf-8"
    )
    print(f"{len(catalogue)} samples -> {(OUT / 'catalogue.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
