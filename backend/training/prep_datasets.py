"""Organise FaceForensics++, Celeb-DF, or DFDC raw downloads into the
--data-real / --data-fake directory structure expected by the training scripts.

Usage
-----
python -m training.prep_datasets \\
    --dataset dfdc \\
    --input /path/to/dfdc_train_part_0/ \\
    --output-real ./data/real \\
    --output-fake ./data/fake

Supported --dataset values: dfdc, faceforensics, celebdf

Optional flags
  --copy      Copy files instead of creating symlinks (slower, portable).
  --frames N  Extract N evenly-spaced frames from each video as JPEG images
              instead of linking the video files. Requires ffmpeg on PATH.
              Useful when training the image head (DINOv2).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def _extract_frames(video: Path, out_dir: Path, n: int, stem: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / f"{stem}_%04d.jpg")
    duration_cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video),
    ]
    try:
        dur = float(subprocess.check_output(duration_cmd, stderr=subprocess.DEVNULL))
    except Exception:
        dur = None

    if dur and dur > 0:
        fps = n / dur
        cmd = ["ffmpeg", "-i", str(video), "-vf", f"fps={fps:.4f}",
               "-q:v", "2", "-frames:v", str(n), pattern, "-y"]
    else:
        cmd = ["ffmpeg", "-i", str(video), "-vf", f"select=not(mod(n\\,30))",
               "-vsync", "vfr", "-q:v", "2", "-frames:v", str(n), pattern, "-y"]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        return 0
    return len(list(out_dir.glob(f"{stem}_*.jpg")))


def prep_dfdc(input_dir: Path, real_dir: Path, fake_dir: Path,
              copy: bool, frames: int | None) -> tuple[int, int]:
    """DFDC: metadata.json maps filename -> {"label": "REAL"/"FAKE", ...}."""
    metadata_path = input_dir / "metadata.json"
    if not metadata_path.exists():
        sys.exit(f"ERROR: metadata.json not found in {input_dir}. "
                 "Make sure --input points to a single DFDC part directory.")

    with open(metadata_path) as f:
        meta: dict[str, dict] = json.load(f)

    n_real = n_fake = 0
    for filename, info in meta.items():
        src = input_dir / filename
        if not src.exists():
            continue
        label = str(info.get("label", "")).upper()
        out_base = real_dir if label == "REAL" else fake_dir
        if frames:
            out_sub = out_base / src.stem
            count = _extract_frames(src, out_sub, frames, src.stem)
            if label == "REAL":
                n_real += count
            else:
                n_fake += count
        else:
            dst = out_base / src.name
            _link_or_copy(src, dst, copy)
            if label == "REAL":
                n_real += 1
            else:
                n_fake += 1

    return n_real, n_fake


def prep_faceforensics(input_dir: Path, real_dir: Path, fake_dir: Path,
                       copy: bool, frames: int | None) -> tuple[int, int]:
    """FaceForensics++: original_sequences/ is real, manipulated_sequences/ is fake.

    The script descends into any compression level (c0/c23/c40) and any
    manipulation method (Deepfakes, Face2Face, FaceSwap, NeuralTextures).
    """
    exts = IMAGE_EXTS if frames is None else VIDEO_EXTS

    def walk(root: Path, out_dir: Path) -> int:
        count = 0
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() not in exts and p.suffix.lower() not in VIDEO_EXTS:
                continue
            if frames and p.suffix.lower() in VIDEO_EXTS:
                out_sub = out_dir / p.stem
                count += _extract_frames(p, out_sub, frames, p.stem)
            elif not frames and p.suffix.lower() in exts:
                _link_or_copy(p, out_dir / p.name, copy)
                count += 1
            elif not frames and p.suffix.lower() in VIDEO_EXTS:
                _link_or_copy(p, out_dir / p.name, copy)
                count += 1
        return count

    real_root = input_dir / "original_sequences"
    fake_root = input_dir / "manipulated_sequences"

    if not real_root.exists():
        sys.exit(f"ERROR: original_sequences/ not found in {input_dir}.")
    if not fake_root.exists():
        sys.exit(f"ERROR: manipulated_sequences/ not found in {input_dir}.")

    n_real = walk(real_root, real_dir)
    n_fake = walk(fake_root, fake_dir)
    return n_real, n_fake


def prep_celebdf(input_dir: Path, real_dir: Path, fake_dir: Path,
                 copy: bool, frames: int | None) -> tuple[int, int]:
    """Celeb-DF v2: Celeb-real/ and YouTube-real/ are real; Celeb-synthesis/ is fake."""
    real_sources = [input_dir / "Celeb-real", input_dir / "YouTube-real"]
    fake_sources = [input_dir / "Celeb-synthesis"]

    def walk(sources: list[Path], out_dir: Path) -> int:
        count = 0
        for src_root in sources:
            if not src_root.exists():
                continue
            for p in sorted(src_root.rglob("*")):
                if p.suffix.lower() not in VIDEO_EXTS:
                    continue
                if frames:
                    out_sub = out_dir / p.stem
                    count += _extract_frames(p, out_sub, frames, p.stem)
                else:
                    _link_or_copy(p, out_dir / p.name, copy)
                    count += 1
        return count

    n_real = walk(real_sources, real_dir)
    n_fake = walk(fake_sources, fake_dir)
    return n_real, n_fake


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Organise deepfake dataset downloads into real/fake directory pairs."
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=["dfdc", "faceforensics", "celebdf"],
        help="Which dataset to prepare.",
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="Path to the raw dataset download.")
    parser.add_argument("--output-real", required=True, type=Path,
                        dest="output_real", help="Destination for real files.")
    parser.add_argument("--output-fake", required=True, type=Path,
                        dest="output_fake", help="Destination for fake/synthetic files.")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of symlinking.")
    parser.add_argument("--frames", type=int, default=None,
                        help="Extract N frames per video instead of linking the video.")
    args = parser.parse_args()

    input_dir: Path = args.input.resolve()
    real_dir: Path = args.output_real.resolve()
    fake_dir: Path = args.output_fake.resolve()

    if not input_dir.exists():
        sys.exit(f"ERROR: input directory does not exist: {input_dir}")

    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    if args.frames is not None and args.frames < 1:
        sys.exit("ERROR: --frames must be >= 1")

    fn = {"dfdc": prep_dfdc, "faceforensics": prep_faceforensics, "celebdf": prep_celebdf}[
        args.dataset
    ]
    print(f"Preparing {args.dataset} from {input_dir} ...")
    n_real, n_fake = fn(input_dir, real_dir, fake_dir, args.copy, args.frames)

    print(f"Done.")
    print(f"  Real  → {real_dir}  ({n_real} files)")
    print(f"  Fake  → {fake_dir}  ({n_fake} files)")
    if n_real == 0 or n_fake == 0:
        print("WARNING: one split is empty — check --input path and dataset structure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
