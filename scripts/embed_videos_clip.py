"""CLIP face embeddings of raw videos through the website's own path, per 3-second part, for testing a probe.

    .venv/Scripts/python.exe scripts/embed_videos_clip.py --set NAME --list FILE.csv [--device cuda]

FILE.csv has columns path, label (0 real, 1 fake) and group (a generator or source name). Each video is
cut into the website's parts (plan_windows), each part is cleaned (neutralize_clip) and its face crops
prepared exactly as the site does, and the crops get the CLIP image embedding of embed_faces_clip.py.
Writes runs/gen-probe/videos-NAME.jsonl, one line per part, resuming by path. Test sets only: nothing
here is trained on.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "runs" / "gen-probe"


def face_batch(preprocessor, neutral: Path, cache_root: Path):
    """The face crops the website scores for one cleaned part, or None when no steady face is found."""
    from deepfake_detection.data.datasets import CachedBranchDataset, collate_branch_items
    from deepfake_detection.data.manifest import ClipRecord
    from deepfake_detection.views.cache_store import CacheStore
    from deepfake_detection.webapp.scorer import P001_PREPROCESSING_HASH

    record = ClipRecord(
        clip_id="upload",
        dataset="Upload",
        video_path=neutral,
        manipulation_type="RealVideo-RealAudio",
        method="upload",
        source="upload",
        targets=(),
        clip_fake=False,
        video_fake=False,
        audio_fake=False,
    )
    prepared = preprocessor.prepare(record, neutral)
    store = CacheStore(cache_root)
    path = store.save(prepared, dataset=record.dataset)
    if "visual_view" not in store.available_views(path):
        return None
    item = CachedBranchDataset(
        records=[record],
        cache_index={record.clip_id: path},
        cache_store=store,
        branch="visual",
        preprocessing_hash=P001_PREPROCESSING_HASH,
    )[0]
    return collate_branch_items([item]).values[0]


def main() -> int:
    from embed_faces_clip import ClipEmbedder

    from deepfake_detection.preprocessing.neutralize import NeutralSettings, plan_windows
    from deepfake_detection.webapp.scorer import Scorer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", required=True)
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    target = OUT / f"videos-{args.set}.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if target.exists():
        done = {json.loads(line)["path"] for line in target.read_text(encoding="utf-8").splitlines()}
    with open(args.list, encoding="utf-8", newline="") as f:
        videos = [r for r in csv.DictReader(f) if r["path"] not in done]
    preprocessor = Scorer(device=args.device).models.preprocessor
    embed = ClipEmbedder(args.device)
    began = time.time()
    for n, video in enumerate(videos, 1):
        source = Path(video["path"])
        parts = plan_windows(source, NeutralSettings())
        lines = []
        for i, (start, _) in enumerate(parts):
            with tempfile.TemporaryDirectory() as work:
                work = Path(work)
                neutral = Scorer._neutralize(source, work / "neutral.mp4", start)
                crops = face_batch(preprocessor, neutral, work / "cache")
            lines.append(
                {
                    "path": video["path"],
                    "label": int(video["label"]),
                    "group": video["group"],
                    "part": i,
                    "embedding": None if crops is None else [round(float(x), 6) for x in embed(crops)],
                }
            )
        if not parts:
            lines.append({"path": video["path"], "label": int(video["label"]), "group": video["group"], "part": None, "embedding": None})
        with open(target, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        if n % 20 == 0:
            print(f"{args.set} {n}/{len(videos)}, {(time.time() - began) / n:.1f} s/video", flush=True)
    print(f"done: {args.set} -> {target.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
