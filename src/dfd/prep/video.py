"""Video decoding and uniform frame sampling (data preparation only).

Three decode backends are provided so that their speed can be compared
(docs/_agents/S2C.md). None of them keeps a whole video in memory: frames
are yielded one at a time and only the sampled frames are decoded to pixels.

    cv_seek    OpenCV VideoCapture, jump to each wanted frame with CAP_PROP_POS_FRAMES.
    cv_seq     OpenCV VideoCapture, read from the start with grab(), retrieve() only
               the wanted frames (decoding is still sequential, colour conversion is not).
    ffmpeg     ffmpeg subprocess, select filter with the wanted frame numbers, raw BGR
               frames on a pipe, read one frame at a time.

All backends return BGR uint8 arrays (H, W, 3), the OpenCV convention.
The frame positions are chosen by `sample_indices`, which is a pure function of
the frame count, so the same video gives the same positions on every backend.
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2  # noqa: E402

BACKENDS = ("cv_seq", "cv_seek", "ffmpeg")
DEFAULT_N_FRAMES = 16  # a default only, not a decision (PROTOCOL.md section 9)


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    n_frames: int  # container's frame count (may be wrong for odd files)
    fps: float


class VideoOpenError(RuntimeError):
    pass


def is_stable(path: str | Path, min_age_s: float = 20.0) -> bool:
    """True if the file looks finished: no .part suffix and not modified recently."""
    p = Path(path)
    if p.suffix == ".part" or p.name.endswith(".part") or not p.is_file():
        return False
    return (time.time() - p.stat().st_mtime) >= min_age_s


def _open(path: str | Path, threads: int = 1) -> cv2.VideoCapture:
    # CAP_PROP_N_THREADS bounds the decoder threads, so that a pool of workers
    # does not oversubscribe the CPU.
    try:
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, int(threads)])
    except Exception:  # older/newer builds without open-params support
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        raise VideoOpenError(f"cannot open video: {path}")
    return cap


def probe(path: str | Path) -> VideoInfo:
    """Read the header only (no frame is decoded)."""
    cap = _open(path)
    try:
        info = VideoInfo(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
        )
    finally:
        cap.release()
    if info.n_frames <= 0 or info.width <= 0 or info.height <= 0:
        raise VideoOpenError(f"video has no frames or no size: {path}")
    return info


def sample_indices(n_frames: int, n: int = DEFAULT_N_FRAMES) -> list[int]:
    """n uniformly spaced frame indices in [0, n_frames).

    The video is cut into n equal segments and the frame at the centre of each
    segment is taken: idx_i = floor((i + 0.5) * n_frames / n). This avoids the
    first and last frame (the last is the least reliable to seek to) and gives
    the same spacing between all neighbours up to rounding. If n_frames <= n all
    frames are returned (fewer than n; the caller sees the shorter list).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if n_frames < 1:
        return []
    if n_frames <= n:
        return list(range(n_frames))
    return [int((i + 0.5) * n_frames // n) for i in range(n)]


def _iter_cv_seek(path, indices, threads) -> Iterator[tuple[int, np.ndarray]]:
    cap = _open(path, threads)
    try:
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                continue  # recorded by the caller as a missing frame
            yield i, frame
    finally:
        cap.release()


def _iter_cv_seq(path, indices, threads) -> Iterator[tuple[int, np.ndarray]]:
    cap = _open(path, threads)
    wanted = sorted(set(indices))
    try:
        pos = 0
        for i in wanted:
            while pos < i:
                if not cap.grab():
                    return
                pos += 1
            ok, frame = cap.read()  # grab + retrieve frame number i
            pos += 1
            if not ok:
                return
            yield i, frame
    finally:
        cap.release()


def _iter_ffmpeg(path, indices, info: VideoInfo, threads) -> Iterator[tuple[int, np.ndarray]]:
    wanted = sorted(set(indices))
    if not wanted:
        return
    sel = "+".join(f"eq(n,{i})" for i in wanted)
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-threads", str(threads),
        "-i", str(path), "-an", "-sn",
        "-vf", f"select='{sel}'", "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    nbytes = info.width * info.height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        k = 0
        while k < len(wanted):
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            yield wanted[k], np.frombuffer(buf, np.uint8).reshape(info.height, info.width, 3).copy()
            k += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def iter_frames(
    path: str | Path,
    n: int = DEFAULT_N_FRAMES,
    backend: str = "cv_seq",
    indices: Sequence[int] | None = None,
    threads: int = 1,
    info: VideoInfo | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (source_frame_index, BGR frame) for uniformly spaced frames.

    `indices` overrides the uniform rule. Frames that cannot be decoded are
    simply absent from the stream; compare with the requested list to see them.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}, choose from {BACKENDS}")
    if info is None:
        info = probe(path)
    idx = list(indices) if indices is not None else sample_indices(info.n_frames, n)
    if backend == "cv_seq":
        return _iter_cv_seq(path, idx, threads)
    if backend == "cv_seek":
        return _iter_cv_seek(path, idx, threads)
    return _iter_ffmpeg(path, idx, info, threads)


def read_frames(path, n: int = DEFAULT_N_FRAMES, backend: str = "cv_seq", **kw) -> list[tuple[int, np.ndarray]]:
    """Convenience: list of the sampled frames (n frames in memory, never the whole video)."""
    return list(iter_frames(path, n=n, backend=backend, **kw))
