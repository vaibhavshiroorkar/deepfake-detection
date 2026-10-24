"""Face size and number of faces per video clip of the T-V validation split and dev_x (battery item 6, CPU only).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/mvp/face_stats.py

Three frames per clip (the 4th, 9th and 13th of the 16 uniformly spaced frames of the preprocessing), the same YuNet detector as the
preprocessing. Per clip: the largest number of faces in any of the three frames, the median height of the largest face divided by the frame
height, and the frame size. Reads only FF++ validation and Celeb-DF v2 dev_x videos. Output data/_results/I-0/T-V/face_stats.json.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from dfd.data import loaders as L
from dfd.prep import face as F
from dfd.prep import video as V
from dfd.sweep import extract_video

_DET = None


def _init():
    global _DET
    _DET = F.make_detector("yunet", F.DEFAULT_MODEL_DIR)


def one(args):
    sid, path = args
    try:
        info = V.probe(path)
        idx = V.sample_indices(info.n_frames, 16)
        pick = [idx[3], idx[8], idx[12]]
        n, hs = 0, []
        for _, fr in V.iter_frames(path, backend="cv_seq", indices=pick, info=info):
            faces = _DET.detect(fr)
            n = max(n, len(faces))
            big = F.largest(faces)
            if big is not None:
                hs.append(big.h / fr.shape[0])
        return sid, dict(n_faces=n, face_h_rel=float(np.median(hs)) if hs else None, width=info.width, height=info.height)
    except Exception as e:  # recorded, not hidden
        return sid, dict(error=f"{type(e).__name__}: {e}")


def main():
    jobs = [extract_video.JOBS[1], extract_video.JOBS[2]]        # ffpp val, celebdf dev_x
    items = []
    for name, manifest, split_csv, crops, split in jobs:
        for r in L.select_records(manifest, split_csv, split, modality="video"):
            items.append((r.sample_id, str(Path("data") / r.path)))
    t0 = time.perf_counter()
    with mp.Pool(10, initializer=_init) as pool:
        out = dict(pool.imap_unordered(one, items, chunksize=8))
    Path("data/_results/I-0/T-V").mkdir(parents=True, exist_ok=True)
    Path("data/_results/I-0/T-V/face_stats.json").write_text(json.dumps(out))
    err = sum("error" in v for v in out.values())
    print("clips", len(out), "errors", err, "seconds", round(time.perf_counter() - t0, 1))


if __name__ == "__main__":
    main()
