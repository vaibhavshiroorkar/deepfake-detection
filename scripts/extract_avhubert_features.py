"""Cache frozen AV-HuBERT features for every clip in a manifest: the sync path's input.

Each clip gives up to `--windows` 2-second windows, starting at 0, 2, 4 ... seconds
(or at the manifest's `sync_start_sec` when it has one). Each window holds three
passes of the same encoder, each [50, width] at 25 steps per second: lips and
audio together, lips alone, and audio alone. They are stored as float16 in one
.npz per window, with the share of frames where a face was found.

    uv run python scripts/extract_avhubert_features.py --manifest data/manifests/fakeavceleb-neutral.csv \
        --output runs/avhubert-fakeavceleb-neutral

Resumable: windows whose .npz exists are skipped. The index records failures
(no face, no audio, decode errors) so they are counted, not silently dropped.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np


WINDOW_SECONDS = 2.0


def _duration(path: Path) -> float:
    run = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=False,
    )
    try:
        return float(json.loads(run.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return 0.0


def window_starts(duration: float, fixed: str, count: int, step: float = 2.0) -> list[float]:
    if fixed.strip():
        return [float(fixed)]
    starts = [step * i for i in range(count) if step * i + WINDOW_SECONDS <= duration + 1e-6]
    return starts or ([0.0] if duration >= 1.0 else [])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("models/avhubert/base_vox_iter5.pt"))
    parser.add_argument("--landmarker", type=Path, default=Path("models/mediapipe/face_landmarker.task"))
    parser.add_argument("--eye-px", type=float, default=64.0)
    parser.add_argument("--locator", type=Path, help="YuNet model for small faces (P-001 amendment A10)")
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--step", type=float, default=2.0, help="seconds between window starts")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cuts", type=Path, help="neutralizer sidecar with kept_sec per clip")
    arguments = parser.parse_args(argv)

    from deepfake_detection.avsync.features import cache_path, encode, init_worker, prepare_window
    from deepfake_detection.streams.avhubert import load_avhubert

    with arguments.manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if arguments.limit:
        rows = rows[: arguments.limit]

    arguments.output.mkdir(parents=True, exist_ok=True)
    index_path = arguments.output / "index.csv"
    done: set[str] = set()
    if index_path.exists():
        # Only final outcomes count as done. A decode error under memory
        # pressure is transient, and skipping it on resume would lose the window.
        with index_path.open(newline="", encoding="utf-8") as handle:
            done = {r["window_id"] for r in csv.DictReader(handle)
                    if r["status"] in {"ok", "no face", "no audio"}}

    paths = []
    for row in rows:
        path = Path(row["video_path"].replace("\\", "/"))
        paths.append(path if path.is_absolute() else arguments.data_dir / path)
    # The neutralizer's sidecar already holds every kept length. Probing 21,544
    # files one by one took most of an hour on the first run.
    lengths: dict[str, float] = {}
    if arguments.cuts and arguments.cuts.exists():
        with arguments.cuts.open(newline="", encoding="utf-8") as handle:
            lengths = {r["clip_id"]: float(r["kept_sec"]) for r in csv.DictReader(handle) if r.get("kept_sec")}
    missing = [i for i, row in enumerate(rows) if row["clip_id"] not in lengths]
    with ThreadPoolExecutor(16) as pool:
        for i, value in zip(missing, pool.map(_duration, [paths[i] for i in missing]), strict=True):
            lengths[rows[i]["clip_id"]] = value

    jobs, meta = [], []
    for row, path in zip(rows, paths, strict=True):
        for start in window_starts(lengths[row["clip_id"]], row.get("sync_start_sec", "") or "", arguments.windows, arguments.step):
            window_id = f"{row['clip_id']}@{start:.2f}"
            if window_id in done:
                continue
            jobs.append((str(path), start))
            meta.append((row["clip_id"], window_id, start))

    model = load_avhubert(arguments.weights).to("cuda").eval()
    fields = ("clip_id", "window_id", "start", "file", "face", "status")
    with index_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not done:
            writer.writeheader()
        with Pool(arguments.workers, initializer=init_worker,
                  initargs=(str(arguments.landmarker), arguments.eye_px,
                            str(arguments.locator) if arguments.locator else None)) as pool:
            for count, ((clip_id, window_id, start), (face, crops, audio)) in enumerate(
                zip(meta, pool.imap(prepare_window, jobs, chunksize=2), strict=True), start=1
            ):
                target = cache_path(arguments.output, window_id)
                if crops is None or audio is None or len(audio) == 0:
                    status = face if isinstance(face, str) else ("no audio" if crops is not None else "no face")
                    face_share = 0.0 if isinstance(face, str) else float(face)
                    file = ""
                else:
                    status, face_share = "ok", float(face)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(target, face=np.float32(face_share), **encode(model, crops, audio))
                    file = str(target.relative_to(arguments.output)).replace("\\", "/")
                writer.writerow({"clip_id": clip_id, "window_id": window_id, "start": start,
                                 "file": file, "face": round(face_share, 4), "status": status})
                if count % 500 == 0:
                    handle.flush()
                    print(f"{count}/{len(jobs)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
