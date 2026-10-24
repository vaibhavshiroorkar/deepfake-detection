"""Pulse (rPPG) signal-sanity check S-1: skin trace, POS, spectral SNR, shuffled control. CPU only.

Source of the POS method: Wang, den Brinker, Stuijk, de Haan, "Algorithmic Principles of Remote-PPG", IEEE TBME
64(7), 2017 (author's pre-review version; RESEARCH.md R-057): Algorithm 1 on page 7, equations (32) to (34).
Everything that the source does not fix is marked "implementer's choice" in the field comments of S1Params and in the
design spec (scratchpad s1_spec.md, copied to docs by the main agent). No parameter is tuned on results.

Pipeline for one clip:
    segment_plan        which contiguous frames (start, count) of the original video
    extract_trace       per frame: YuNet face, smoothed box, central ROI, YCrCb skin mask, mean RGB (NaN if no face)
    fill_missing        linear interpolation of the RGB trace over frames without a face
    pos                 POS with a sliding window and overlap-add (no band-pass, as in the source)
    spectrum_snr        Hann periodogram, peak in 0.7 to 3 Hz, SNR around the peak and its second harmonic
    shuffled_control    the same measure on independent time shuffles of the per-frame RGB rows
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2  # noqa: E402


@dataclass(frozen=True)
class S1Params:
    # segment (implementer's choice unless stated)
    seg_seconds: float = 10.0        # from the pre-declared design ("about 10 s")
    start_offset_s: float = 0.5      # skip the first 0.5 s, then take the segment
    min_usable_s: float = 9.0        # a clip that yields fewer frames than this is unusable
    # face and skin region (implementer's choice)
    min_face_px: int = 48            # faces with a smaller box side (original pixels) count as no face
    roi_lo: float = 0.2              # central ROI of the YuNet box: [0.2, 0.8] of width and of height
    roi_hi: float = 0.8
    cr_lo: int = 133                 # YCrCb skin rule (OpenCV YCrCb order Y, Cr, Cb)
    cr_hi: int = 173
    cb_lo: int = 77
    cb_hi: int = 127
    min_skin_frac: float = 0.2       # fewer skin pixels than this share of the ROI: frame counts as no face
    min_skin_px: int = 30
    box_ema: float = 0.3             # weight of the new detection in the causal box smoothing
    box_reset_iou: float = 0.3       # IoU with the smoothed box below this: take the new box as is
    min_face_share: float = 0.8      # at least this share of frames with a face, else the clip is unusable
    max_gap_s: float = 1.0           # longest run of frames without a face that is interpolated
    # POS (source: window l = 32 frames at 20 fps, i.e. 1.6 s)
    pos_window_s: float = 1.6
    # spectrum and SNR
    band_lo: float = 0.7             # pre-declared design
    band_hi: float = 3.0             # pre-declared design
    delta_hz: float = 0.2            # implementer's choice: half main lobe of a Hann window over 10 s (2/T)
    nfft_min_bins_per_hz: float = 100.0  # zero padding to at least 100 bins per Hz (spacing <= 0.01 Hz)
    # control and sampling
    n_shuffles: int = 20             # pre-declared: at least 10
    percentile: float = 95.0         # pre-declared
    pass_fraction: float = 0.6       # pre-declared
    seed: int = 0
    n_per_dataset: int = 150         # pre-declared target


PARAMS = S1Params()


def params_dict(p: S1Params = PARAMS) -> dict:
    return asdict(p)


# ---------------------------------------------------------------- segment

def segment_plan(n_total: int, fps: float, p: S1Params = PARAMS):
    """(start_frame, n_frames) of the contiguous segment, or None if the video has fewer than n_frames frames.

    n_frames = round(seg_seconds * fps). The segment starts after start_offset_s, moved earlier if the video is
    too short to leave room (down to frame 0), so every video with at least n_frames frames is eligible."""
    if not (fps and fps > 0 and n_total and n_total > 0):
        return None
    n = int(round(p.seg_seconds * fps))
    if n_total < n:
        return None
    start = min(int(round(p.start_offset_s * fps)), n_total - n)
    return start, n


# ---------------------------------------------------------------- skin trace

def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def skin_rgb(frame_bgr: np.ndarray, box, p: S1Params = PARAMS):
    """Mean RGB of the skin-coloured pixels in the central ROI of a face box, or None.

    box = (x, y, w, h) in pixels. The ROI is [roi_lo, roi_hi] of the box in both directions (clipped to the frame);
    a pixel is skin if Cr and Cb lie in the fixed ranges of S1Params. Returns a float64 array [R, G, B]."""
    H, W = frame_bgr.shape[:2]
    x, y, w, h = box
    x0 = int(round(x + p.roi_lo * w)); x1 = int(round(x + p.roi_hi * w))
    y0 = int(round(y + p.roi_lo * h)); y1 = int(round(y + p.roi_hi * h))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    roi = frame_bgr[y0:y1, x0:x1]
    ycc = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    m = ((ycc[..., 1] >= p.cr_lo) & (ycc[..., 1] <= p.cr_hi) & (ycc[..., 2] >= p.cb_lo) & (ycc[..., 2] <= p.cb_hi))
    n = int(m.sum())
    if n < max(p.min_skin_px, p.min_skin_frac * m.size):
        return None
    px = roi[m]  # [n, 3] BGR
    return px.mean(axis=0)[::-1].astype(np.float64)


def extract_trace(path: str, detector, start: int, n_frames: int, p: S1Params = PARAMS) -> dict:
    """Read frames [start, start + n_frames) sequentially and return the per-frame mean skin RGB.

    detector must offer detect(frame_bgr) -> list of faces with attributes x, y, w, h sorted by decreasing area
    (dfd.prep.face YuNet). The largest face is used; its box is smoothed causally (exponential average, reset when
    the new box overlaps the smoothed one by IoU below box_reset_iou). Frames without a usable face give NaN.
    Returns dict(rgb [n_read, 3] with NaN, fps, n_read, n_total, width, height)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return dict(rgb=np.zeros((0, 3)), fps=float("nan"), n_read=0, n_total=0, width=0, height=0, error="open_failed")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rgb = np.full((n_frames, 3), np.nan)
    k = 0
    box = None
    try:
        for _ in range(start):          # sequential decode, no seeking (seek accuracy depends on the backend)
            if not cap.grab():
                break
        while k < n_frames:
            ok, frame = cap.read()
            if not ok:
                break
            faces = detector.detect(frame)
            f = faces[0] if faces else None
            if f is not None and min(f.w, f.h) >= p.min_face_px:
                new = np.array([f.x, f.y, f.w, f.h], dtype=np.float64)
                if box is None or _iou(box, new) < p.box_reset_iou:
                    box = new
                else:
                    box = (1.0 - p.box_ema) * box + p.box_ema * new
                v = skin_rgb(frame, box, p)
                if v is not None:
                    rgb[k] = v
            k += 1
    finally:
        cap.release()
    return dict(rgb=rgb[:k], fps=fps, n_read=k, n_total=n_total, width=width, height=height)


def fill_missing(rgb: np.ndarray):
    """Linear interpolation of NaN rows (constant extrapolation at the ends).

    Returns (filled [N, 3], face_share, max_gap_frames, n_valid). If no row is valid, filled is all NaN."""
    N = len(rgb)
    valid = np.isfinite(rgb).all(axis=1)
    nv = int(valid.sum())
    if N == 0 or nv == 0:
        return np.full((N, 3), np.nan), 0.0, N, nv
    idx = np.arange(N)
    out = np.empty_like(rgb, dtype=np.float64)
    for c in range(3):
        out[:, c] = np.interp(idx, idx[valid], rgb[valid, c])
    gap = best = 0
    for v in valid:
        gap = 0 if v else gap + 1
        best = max(best, gap)
    return out, nv / N, best, nv


# ---------------------------------------------------------------- POS

_P = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])  # equation (33): S1 = Gn - Bn, S2 = -2 Rn + Gn + Bn


def pos(rgb: np.ndarray, fps: float, p: S1Params = PARAMS) -> np.ndarray:
    """POS pulse signal H [N] (Algorithm 1 of the source).

    For every window of l = round(pos_window_s * fps) frames: divide each channel by its mean over the window
    (temporal normalisation), project with (33), tune h = S1 + (std(S1) / std(S2)) S2 (34), make h zero-mean and
    add it into the output at the window position (overlap-add, window slides one frame). No band-pass."""
    rgb = np.asarray(rgb, dtype=np.float64)
    N = len(rgb)
    l = max(2, int(round(p.pos_window_s * fps)))
    if N < l:
        raise ValueError(f"trace of {N} frames is shorter than the POS window of {l} frames")
    W = sliding_window_view(rgb, l, axis=0)             # [M, 3, l]
    mu = W.mean(axis=2, keepdims=True)
    mu = np.where(mu == 0, 1.0, mu)
    Cn = W / mu
    S1 = Cn[:, 1] - Cn[:, 2]                            # same as _P[0] @ Cn
    S2 = -2.0 * Cn[:, 0] + Cn[:, 1] + Cn[:, 2]          # same as _P[1] @ Cn
    s1, s2 = S1.std(axis=1), S2.std(axis=1)
    alpha = np.divide(s1, s2, out=np.zeros_like(s1), where=s2 > 0)
    h = S1 + alpha[:, None] * S2
    h = h - h.mean(axis=1, keepdims=True)
    M = N - l + 1
    H = np.zeros(N)
    for k in range(l):
        H[k:k + M] += h[:, k]
    return H


# ---------------------------------------------------------------- spectrum and SNR

def spectrum(x: np.ndarray, fps: float, p: S1Params = PARAMS):
    """Periodogram of the linearly detrended, Hann-windowed signal, zero-padded. Returns (freqs, power)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    t = np.arange(n)
    slope, icpt = np.polyfit(t, x, 1)
    x = (x - (slope * t + icpt)) * np.hanning(n)
    nfft = 1 << int(math.ceil(math.log2(max(n, p.nfft_min_bins_per_hz * fps))))
    P = np.abs(np.fft.rfft(x, nfft)) ** 2
    return np.fft.rfftfreq(nfft, d=1.0 / fps), P


def spectrum_snr(x: np.ndarray, fps: float, p: S1Params = PARAMS) -> dict:
    """Peak frequency in [band_lo, band_hi] and its SNR in dB.

    signal = power within +-delta_hz of the peak f0 and of 2 f0; noise = all other power in the band;
    snr_db = 10 log10(signal / noise). Both bands are cut to [band_lo, band_hi]. Returns dict(peak_hz, bpm, snr_db)."""
    f, P = spectrum(x, fps, p)
    band = (f >= p.band_lo) & (f <= p.band_hi)
    if not band.any() or not np.isfinite(P[band]).all() or P[band].max() <= 0:
        return dict(peak_hz=float("nan"), bpm=float("nan"), snr_db=float("nan"))
    fb, Pb = f[band], P[band]
    f0 = float(fb[np.argmax(Pb)])
    sig = (np.abs(fb - f0) <= p.delta_hz) | (np.abs(fb - 2.0 * f0) <= p.delta_hz)
    s, nz = float(Pb[sig].sum()), float(Pb[~sig].sum())
    snr = 10.0 * math.log10(s / nz) if nz > 0 and s > 0 else float("nan")
    return dict(peak_hz=f0, bpm=60.0 * f0, snr_db=snr)


def pulse_snr(rgb: np.ndarray, fps: float, p: S1Params = PARAMS) -> dict:
    """POS then spectrum_snr on a filled RGB trace."""
    return spectrum_snr(pos(rgb, fps, p), fps, p)


def shuffled_control(rgb: np.ndarray, fps: float, rng: np.random.Generator, n_shuffles: int | None = None,
                     p: S1Params = PARAMS) -> np.ndarray:
    """SNR (dB) of n_shuffles independent time permutations of the per-frame RGB rows (same POS and SNR code)."""
    n = p.n_shuffles if n_shuffles is None else n_shuffles
    out = np.empty(n)
    N = len(rgb)
    for i in range(n):
        out[i] = pulse_snr(rgb[rng.permutation(N)], fps, p)["snr_db"]
    return out


# ---------------------------------------------------------------- one clip

def analyse_clip(trace: dict, rng: np.random.Generator, p: S1Params = PARAMS) -> dict:
    """Usability rules, real measurement and shuffled control for one extracted trace (dict of extract_trace)."""
    fps, n_read = trace["fps"], trace["n_read"]
    res = dict(fps=fps, n_frames=n_read, face_share=0.0, max_gap_frames=n_read, peak_hz=float("nan"),
               bpm=float("nan"), snr_db=float("nan"), usable=False, reason="", shuffled=np.full(p.n_shuffles, np.nan))
    if trace.get("error"):
        res["reason"] = trace["error"]
        return res
    if not (fps > 0) or n_read < int(round(p.min_usable_s * fps)):
        res["reason"] = "short_read"
        return res
    filled, share, gap, nv = fill_missing(trace["rgb"])
    res.update(face_share=share, max_gap_frames=gap)
    if share < p.min_face_share:
        res["reason"] = "face_share"
        return res
    if gap > int(round(p.max_gap_s * fps)):
        res["reason"] = "long_gap"
        return res
    m = pulse_snr(filled, fps, p)
    if not np.isfinite(m["snr_db"]):
        res["reason"] = "nan_snr"
        return res
    res.update(peak_hz=m["peak_hz"], bpm=m["bpm"], snr_db=m["snr_db"], usable=True, reason="ok",
               shuffled=shuffled_control(filled, fps, rng, p.n_shuffles, p))
    res["filled"] = filled
    return res


# ---------------------------------------------------------------- statistics

def wilson(k: int, n: int, z: float = 1.959963984540054):
    """Wilson score interval (lo, hi) for k successes in n trials (95 percent by default)."""
    if n <= 0:
        return float("nan"), float("nan")
    ph = k / n
    d = 1.0 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    r = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - r), min(1.0, c + r)
