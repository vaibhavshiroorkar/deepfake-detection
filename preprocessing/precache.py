"""
One-time pre-caching of face crops + aligned audio for the train+val splits.

Training's __getitem__ calls extract_clip(), which caches to
data/processed/<clip_id>/ on first access. Doing that lazily during epoch 1
makes the first epoch very slow. This script front-loads it in parallel so
every training epoch just reads the cache.

Parallelism: a process pool, each worker forcing the detector onto CPU
(device="cpu") so the workers don't fight over the single 6GB GPU. It's
idempotent -- already cached clips are skipped by extract_clip's fast path -- so
it's safe to re-run or resume after an interruption.

The detector is stamped into each cache, so switching detectors re-extracts
rather than reusing the other one's crops. Caching both means running this twice.

Usage:
    python -m preprocessing.precache                 # train+val, ~6 workers
    python -m preprocessing.precache --workers 1     # single process (uses GPU)
    python -m preprocessing.precache --splits train  # one split only
    python -m preprocessing.precache --detector yunet
"""
import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from preprocessing.extract_clip import extract_clip
from preprocessing.ops import detectors as D

DATA_DIR = _REPO_ROOT / "data"


def _worker(args) -> tuple:
    """
    Run in a child process. Returns (clip_id, status, message, detect_ms).
    Forces the detector onto CPU so N workers can run concurrently without GPU
    contention. Must be top-level (picklable) for ProcessPoolExecutor on Windows
    (spawn).
    """
    clip_id, video_path, device, detector = args
    try:
        res = extract_clip(Path(video_path), clip_id, device=device, detector=detector)
        if res.get("cached"):
            return (clip_id, "skipped", "", None)
        return (clip_id, "cached", "", res.get("detect_ms"))
    except Exception as e:
        return (clip_id, "failed", str(e), None)


def build_jobs(splits: list[str], device: str, detector: str) -> list[tuple]:
    jobs = []
    for split in splits:
        manifest = DATA_DIR / f"{split}.csv"
        if not manifest.exists():
            raise FileNotFoundError(f"{manifest} not found -- run build_splits.py first.")
        df = pd.read_csv(manifest)
        for _, row in df.iterrows():
            video_path = str(DATA_DIR / row["video_path"])
            jobs.append((row["clip_id"], video_path, device, detector))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="Cap #clips (for a quick test).")
    parser.add_argument("--detector", choices=D.DETECTOR_NAMES, default=D.DEFAULT_DETECTOR,
                        help="Face detector. Stamped into the cache, so switching re-extracts.")
    args = parser.parse_args()

    # workers>1 -> CPU detector (parallel-safe); workers==1 -> let extract_clip
    # auto-select (GPU if available), which is faster for a single process.
    device = "cpu" if args.workers > 1 else None
    jobs = build_jobs(args.splits, device, args.detector)
    if args.limit:
        jobs = jobs[:args.limit]
    # YuNet runs on CPU either way, so the device note would be misleading there.
    where = "cpu (always)" if args.detector == D.YUNET_NAME else (device or "auto")
    print(f"Pre-caching {len(jobs)} clips from splits={args.splits} "
          f"with {args.workers} worker(s), detector={args.detector} device={where}")

    counts = {"cached": 0, "skipped": 0, "failed": 0}
    failures = []
    detect_times = []

    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(jobs), desc="precache")
    except ImportError:
        pbar = None

    def record(result):
        clip_id, status, msg, detect_ms = result
        counts[status] = counts.get(status, 0) + 1
        if detect_ms is not None:
            detect_times.append(detect_ms)
        if status == "failed":
            failures.append({"clip_id": clip_id, "error": msg})
        if pbar:
            pbar.update(1)
            pbar.set_postfix(cached=counts["cached"], skipped=counts["skipped"], failed=counts["failed"])

    if args.workers <= 1:
        for job in jobs:
            record(_worker(job))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_worker, job) for job in jobs]
            for fut in as_completed(futures):
                record(fut.result())

    if pbar:
        pbar.close()

    print(f"\nDone. cached={counts['cached']} skipped(already)={counts['skipped']} failed={counts['failed']}")
    if detect_times:
        # Wall-clock per clip across all workers, so it reflects contention, not
        # a single detector's best case. Only freshly extracted clips count.
        mean_ms = sum(detect_times) / len(detect_times)
        print(f"Detection: {args.detector}, mean {mean_ms:.0f} ms/clip "
              f"over {len(detect_times)} extracted clips")
    if failures:
        out = DATA_DIR / "precache_failures.csv"
        pd.DataFrame(failures).to_csv(out, index=False)
        print(f"Logged {len(failures)} failures to {out}")


if __name__ == "__main__":
    main()
