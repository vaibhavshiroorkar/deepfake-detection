"""Face-crop pipeline: videos listed in a manifest -> JPEG face crops + index CSV.

    PYTHONPATH=src .venv/Scripts/python.exe -m dfd.prep.pipeline \
        --manifest data/_manifests/<file>.csv --out data/_crops/<name> --workers 8

Manifest: a CSV with at least the columns `sample_id` and `path` (the data
manifests of src/dfd/data/manifest.py have more columns, which are ignored; rows whose
`modality` is not video are skipped). Relative paths are resolved against --root
(default `data`, because the data manifests store paths relative to data/ with forward slashes).
The folder of a sample is its sample_id with every character outside [A-Za-z0-9._-]
replaced by `_` (for example ffpp:Deepfakes:123_456 -> ffpp_Deepfakes_123_456).

Output under --out:
    params.json                     the settings of this run (a different setting needs a new --out)
    <folder>/<frame_index>.jpg      one crop per sampled frame with a detected face
    <folder>/index.csv              per-video index rows
    <folder>/_done.json             written last: marks the video finished (resume skips it)
    index.csv                       all finished videos: sample_id, frame_index, source_frame_index,
                                    detected, box, crop_path, score, n_faces, crop_box, note
    videos.csv                      one row per manifest video: status, counts, seconds per stage
    summary.json                    wall time, videos/s, frames/s, CPU use of this run

`frame_index` is the position among the sampled frames (0..N-1); `source_frame_index`
is the frame number in the source video. `box` is the detected face "x,y,w,h" in source
pixels; `crop_box` is the enlarged square "x0,y0,side" that was cut out; `crop_path` is
relative to --out with forward slashes. Frames with no detection have detected=0 and
empty box and crop_path: they are recorded, never silently dropped. A frame that could
not be decoded has note=not_decoded. A video that cannot be opened is listed in
videos.csv with status=error and is retried on the next run.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2  # noqa: E402

from dfd.prep import face as F  # noqa: E402
from dfd.prep import video as V  # noqa: E402

INDEX_COLUMNS = ["sample_id", "frame_index", "source_frame_index", "detected", "box",
                 "crop_path", "score", "n_faces", "crop_box", "note"]
VIDEO_COLUMNS = ["sample_id", "path", "status", "message", "n_requested", "n_decoded", "n_detected",
                 "width", "height", "container_frames", "decode_s", "detect_s", "crop_write_s", "total_s"]


@dataclass(frozen=True)
class Params:
    detector: str = "yunet"
    backend: str = "cv_seq"
    n_frames: int = V.DEFAULT_N_FRAMES
    enlarge: float = F.DEFAULT_ENLARGE
    size: int = F.DEFAULT_CROP_SIZE
    quality: int = F.DEFAULT_JPEG_QUALITY
    model_dir: str = str(F.DEFAULT_MODEL_DIR)
    min_age_s: float = 20.0


_DET = None
_DET_KEY = None


def _init_worker(p: Params) -> None:
    cv2.setNumThreads(1)  # one worker = one core; parallelism comes from the pool
    _get_detector(p)


def _get_detector(p: Params):
    global _DET, _DET_KEY
    key = (p.detector, p.model_dir)
    if _DET is None or _DET_KEY != key:
        _DET, _DET_KEY = F.make_detector(p.detector, p.model_dir), key
    return _DET


def folder_name(sample_id: str) -> str:
    """Folder name of a sample: every character outside [A-Za-z0-9._-] becomes '_'.

    Manifest ids such as ``ffpp:Deepfakes:123_456`` contain ':' which Windows does not allow in names.
    The index keeps the true sample_id; crop_path holds the folder. Two ids that map to the same folder
    are rejected when the manifest is read.
    """
    f = re.sub(r"[^A-Za-z0-9._-]", "_", sample_id)
    if not f.strip("."):
        raise ValueError(f"sample_id is not usable as a folder name: {sample_id!r}")
    return f


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="")
    os.replace(tmp, path)


def _is_done(vdir: Path, params_hash: str) -> bool:
    f = vdir / "_done.json"
    if not f.is_file():
        return False
    try:
        return json.loads(f.read_text(encoding="utf-8")).get("params") == params_hash
    except Exception:
        return False


def process_video(sample_id: str, path: str, out_dir: str, p: Params, params_hash: str) -> dict:
    """Process one video. Never raises: errors come back as status=error."""
    t_start = time.perf_counter()
    res = dict(sample_id=sample_id, path=path, status="ok", message="", n_requested=0, n_decoded=0,
               n_detected=0, width=0, height=0, container_frames=0,
               decode_s=0.0, detect_s=0.0, crop_write_s=0.0, total_s=0.0)
    folder = folder_name(sample_id)
    vdir = Path(out_dir) / folder
    try:
        if not V.is_stable(path, p.min_age_s):
            res.update(status="skipped_unstable", message="missing, .part or modified recently")
            return res
        det = _get_detector(p)
        info = V.probe(path)
        res.update(width=info.width, height=info.height, container_frames=info.n_frames)
        wanted = V.sample_indices(info.n_frames, p.n_frames)
        res["n_requested"] = len(wanted)
        if vdir.exists():  # stale files of an unfinished earlier attempt
            for old in vdir.iterdir():
                if old.is_file():
                    old.unlink()
        vdir.mkdir(parents=True, exist_ok=True)
        got: dict[int, dict] = {}
        it = V.iter_frames(path, backend=p.backend, indices=wanted, info=info)
        t_dec = t_det = t_wr = 0.0
        while True:
            t0 = time.perf_counter()
            nxt = next(it, None)
            t_dec += time.perf_counter() - t0
            if nxt is None:
                break
            src_idx, frame = nxt
            t0 = time.perf_counter()
            faces = det.detect(frame)
            t_det += time.perf_counter() - t0
            t0 = time.perf_counter()
            crop, f, sq = F.crop_largest_face(frame, faces, p.enlarge, p.size)
            k = wanted.index(src_idx)
            row = dict(sample_id=sample_id, frame_index=k, source_frame_index=src_idx,
                       detected=0, box="", crop_path="", score="", n_faces=len(faces),
                       crop_box="", note="")
            if crop is not None:
                rel = f"{folder}/{k}.jpg"
                if not cv2.imwrite(str(Path(out_dir) / rel), crop, [cv2.IMWRITE_JPEG_QUALITY, p.quality]):
                    raise OSError(f"cannot write {rel}")
                row.update(detected=1, box=f"{f.x:.1f},{f.y:.1f},{f.w:.1f},{f.h:.1f}", crop_path=rel,
                           score=f"{f.score:.4f}", crop_box=f"{sq[0]},{sq[1]},{sq[2]}")
            t_wr += time.perf_counter() - t0
            got[k] = row
        rows = []
        for k, src_idx in enumerate(wanted):
            rows.append(got.get(k) or dict(sample_id=sample_id, frame_index=k, source_frame_index=src_idx,
                                           detected=0, box="", crop_path="", score="", n_faces=0,
                                           crop_box="", note="not_decoded"))
        res.update(n_decoded=len(got), n_detected=sum(r["detected"] for r in rows),
                   decode_s=t_dec, detect_s=t_det, crop_write_s=t_wr)
        if not got:
            raise V.VideoOpenError("no frame could be decoded")
        buf = _csv_text(INDEX_COLUMNS, rows)
        _write_atomic(vdir / "index.csv", buf)
        res["total_s"] = time.perf_counter() - t_start
        _write_atomic(vdir / "_done.json", json.dumps({"params": params_hash, **{
            k: res[k] for k in ("n_requested", "n_decoded", "n_detected", "width", "height",
                                "container_frames", "decode_s", "detect_s", "crop_write_s", "total_s")}}))
    except Exception as e:  # noqa: BLE001  (a bad file must not stop the run)
        res.update(status="error", message=f"{type(e).__name__}: {e}"[:300])
    res["total_s"] = time.perf_counter() - t_start
    return res


def _csv_text(columns: list[str], rows: list[dict]) -> str:
    import io
    s = io.StringIO()
    w = csv.DictWriter(s, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return s.getvalue()


def read_manifest(manifest: str | Path, root: str | Path = "data") -> list[tuple[str, str]]:
    """(sample_id, path) for every video row.

    Relative paths are resolved against `root` (the data manifests store paths relative to data/).
    If the manifest has a `modality` column, rows that are not `video` are ignored.
    """
    root = Path(root)
    out, seen, folders = [], set(), {}
    with open(manifest, newline="", encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh)
        if not rd.fieldnames or "sample_id" not in rd.fieldnames or "path" not in rd.fieldnames:
            raise SystemExit("manifest needs the columns sample_id and path")
        for r in rd:
            if r.get("modality", "video") != "video":
                continue
            sid = r["sample_id"]
            if sid in seen:
                raise SystemExit(f"duplicate sample_id in manifest: {sid}")
            seen.add(sid)
            fo = folder_name(sid)
            if fo in folders:
                raise SystemExit(f"sample_ids {folders[fo]!r} and {sid!r} map to the same folder {fo!r}")
            folders[fo] = sid
            pth = Path(r["path"])
            out.append((sid, str(pth if pth.is_absolute() else root / pth)))
    return out


class _CpuMeter:
    """System-wide CPU percent sampled once per second (includes other processes)."""

    def __init__(self):
        try:
            import psutil
            self.psutil = psutil
        except ImportError:
            self.psutil = None
        self.samples: list[float] = []
        self._stop = False

    def start(self):
        if not self.psutil:
            return
        import threading
        self.psutil.cpu_percent(None)

        def run():
            while not self._stop:
                time.sleep(1.0)
                self.samples.append(self.psutil.cpu_percent(None))
        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()

    def stop(self) -> float | None:
        self._stop = True
        return round(sum(self.samples) / len(self.samples), 1) if self.samples else None


def run(manifest, out, workers=4, params: Params = Params(), root="data", limit=None, quiet=False) -> dict:
    items = read_manifest(manifest, root)
    if limit:
        items = items[:limit]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # min_age_s is a run-time filter, not an output setting: leave it out of the identity of the output
    ident = {k: v for k, v in asdict(params).items() if k not in ("min_age_s", "model_dir")}
    params_hash = json.dumps(ident, sort_keys=True)
    pf = out / "params.json"
    if pf.is_file():
        old = json.loads(pf.read_text(encoding="utf-8"))
        if old != ident:
            raise SystemExit(f"{pf} has different settings {old}; use a new --out for new settings")
    else:
        pf.write_text(params_hash, encoding="utf-8")
    todo = [(s, p) for s, p in items if not _is_done(out / folder_name(s), params_hash)]
    n_skipped_done = len(items) - len(todo)
    if not quiet:
        print(f"manifest videos {len(items)}, already finished {n_skipped_done}, to do {len(todo)}, workers {workers}", flush=True)
    meter = _CpuMeter()
    meter.start()
    t0 = time.perf_counter()
    results: dict[str, dict] = {}
    args = ([s for s, _ in todo], [p for _, p in todo], [str(out)] * len(todo),
            [params] * len(todo), [params_hash] * len(todo))
    if workers <= 0:
        it = map(process_video, *args)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(params,))
        it = pool.map(process_video, *args, chunksize=1)
    for i, r in enumerate(it, 1):
        results[r["sample_id"]] = r
        if not quiet and (i % 50 == 0 or i == len(todo)):
            print(f"  {i}/{len(todo)} done, {i / (time.perf_counter() - t0):.2f} videos/s", flush=True)
    if pool:
        pool.shutdown()
    wall = time.perf_counter() - t0
    cpu = meter.stop()

    # merge: videos.csv and index.csv cover every manifest video that is finished or was tried now
    vrows, irows = [], []
    for sid, pth in items:
        if sid in results:
            vrows.append(results[sid])
        elif _is_done(out / folder_name(sid), params_hash):
            d = json.loads((out / folder_name(sid) / "_done.json").read_text(encoding="utf-8"))
            vrows.append(dict(sample_id=sid, path=pth, status="ok_previous_run", message="", **{
                k: d.get(k, "") for k in VIDEO_COLUMNS if k in d}))
        else:
            continue
        if vrows[-1]["status"] in ("ok", "ok_previous_run"):
            with open(out / folder_name(sid) / "index.csv", newline="", encoding="utf-8") as fh:
                irows.extend(csv.DictReader(fh))
    _write_atomic(out / "videos.csv", _csv_text(VIDEO_COLUMNS, vrows))
    _write_atomic(out / "index.csv", _csv_text(INDEX_COLUMNS, irows))
    ok = [r for r in results.values() if r["status"] == "ok"]
    n_frames_done = sum(r["n_decoded"] for r in ok)
    summary = dict(
        params=ident, workers=workers, manifest_videos=len(items), skipped_already_done=n_skipped_done,
        processed_ok=len(ok), errors=sum(r["status"] == "error" for r in results.values()),
        skipped_unstable=sum(r["status"] == "skipped_unstable" for r in results.values()),
        frames_decoded=n_frames_done, frames_detected=sum(r["n_detected"] for r in ok),
        wall_s=round(wall, 3),
        videos_per_s=round(len(ok) / wall, 4) if wall > 0 else None,
        frames_per_s=round(n_frames_done / wall, 3) if wall > 0 else None,
        s_per_video_wall=round(wall / len(ok), 4) if ok else None,
        mean_cpu_s_per_video_in_worker=round(sum(r["total_s"] for r in ok) / len(ok), 4) if ok else None,
        mean_decode_s=round(sum(r["decode_s"] for r in ok) / len(ok), 4) if ok else None,
        mean_detect_s=round(sum(r["detect_s"] for r in ok) / len(ok), 4) if ok else None,
        mean_crop_write_s=round(sum(r["crop_write_s"] for r in ok) / len(ok), 4) if ok else None,
        system_cpu_percent_mean=cpu,
    )
    _write_atomic(out / "summary.json", json.dumps(summary, indent=1))
    if not quiet:
        print(json.dumps(summary), flush=True)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4, help="0 = run in this process")
    ap.add_argument("--root", default="data", help="base folder of the relative paths in the manifest (data manifests are relative to data/)")
    ap.add_argument("--limit", type=int, default=None, help="first K manifest rows only")
    d = Params()
    ap.add_argument("--detector", default=d.detector, choices=F.DETECTORS)
    ap.add_argument("--backend", default=d.backend, choices=V.BACKENDS)
    ap.add_argument("--n-frames", type=int, default=d.n_frames)
    ap.add_argument("--enlarge", type=float, default=d.enlarge)
    ap.add_argument("--size", type=int, default=d.size)
    ap.add_argument("--quality", type=int, default=d.quality)
    ap.add_argument("--model-dir", default=d.model_dir)
    ap.add_argument("--min-age", type=float, default=d.min_age_s, help="skip files modified in the last K seconds")
    a = ap.parse_args(argv)
    p = Params(a.detector, a.backend, a.n_frames, a.enlarge, a.size, a.quality, a.model_dir, a.min_age)
    run(a.manifest, a.out, a.workers, p, a.root, a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
