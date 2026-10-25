"""Audio and video preprocessing for SyncNet v2, following syncnet_python (MIT, Copyright (c) 2016-present Joon Son Chung).

Every constant names its reference line. References: SI = SyncNetInstance.py, RP = run_pipeline.py, all of
github.com/joonson/syncnet_python at master, fetched 2026-10-26. The MFCC function is a port of ``mfcc`` / ``fbank`` / ``sigproc``
from python_speech_features (James Lyons, MIT licence, Copyright (c) 2013 James Lyons), which SI imports (line 117) and calls as
``python_speech_features.mfcc(audio, sample_rate)`` (SI line 201) with the library defaults. The package is not installed in this
environment, so the needed functions are ported with their default arguments.

Deviations from the reference, all deliberate and listed in docs/LIPSYNC_IMPLEMENTATION_NOTE.md:
  * faces come from the project's YuNet detector instead of S3FD, so the box differs from the one the crop rule was designed for;
  * tracking is a nearest-box rule on one face track instead of the reference's IOU tracker with shot detection;
  * video is decoded by one ffmpeg pipe (``-r 25``) instead of jpg files (RP lines 267-273, ``-qscale:v 2``), so there is no jpg
    compression step; audio is decoded by dfd.prep.audio (ffmpeg, mono, 16 kHz) and rounded to int16 as ``-acodec pcm_s16le``.
"""
from __future__ import annotations

import decimal
import math
import subprocess
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.fftpack import dct

FRAME_RATE = 25            # SI line 2 "Video 25 FPS, Audio 16000HZ"; RP line 31 (--frame_rate default 25) and line 267 (ffmpeg -r 25)
SAMPLE_RATE = 16000        # SI line 2; SI line 173 (-ar 16000)
SAMPLES_PER_FRAME = 640    # SI line 214 (floor(len(audio)/640)): 16000 / 25
MFCC_PER_FRAME = 4         # SI line 232 (vframe*4): MFCC step 0.01 s = 100 per second, 4 per 25 fps frame
MFCC_WINDOW = 20           # SI line 232 (vframe*4 : vframe*4+20): 0.2 s of MFCC for one video window
VIDEO_WINDOW = 5           # SI line 227 (vframe : vframe+5): 5 frames per video window
FACE_SIZE = 224            # RP lines 117 and 148 (cv2.resize(face, (224, 224)))
CROP_SCALE = 0.40          # RP line 29 (--crop_scale default 0.40)
CROP_MEDFILT = 13          # RP lines 129-131 (signal.medfilt(..., kernel_size=13))
CROP_FILL = 110            # RP line 142 (np.pad constant_values=(110, 110))
MFCC_NUMCEP, MFCC_NFILT = 13, 26                 # python_speech_features.mfcc defaults (base.py lines 25-27)
MFCC_WINLEN, MFCC_WINSTEP = 0.025, 0.01
MFCC_PREEMPH, MFCC_CEPLIFTER = 0.97, 22


# ---- python_speech_features port (defaults only; winfunc = ones, appendEnergy = True, nfft from calculate_nfft) -----------------

def _round_half_up(x) -> int:                                    # sigproc.py line 10
    return int(decimal.Decimal(x).quantize(decimal.Decimal("1"), rounding=decimal.ROUND_HALF_UP))


def _framesig(sig: np.ndarray, frame_len: float, frame_step: float) -> np.ndarray:    # sigproc.py lines 21-53, no window
    slen = len(sig)
    frame_len, frame_step = _round_half_up(frame_len), _round_half_up(frame_step)
    numframes = 1 if slen <= frame_len else 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))
    padlen = int((numframes - 1) * frame_step + frame_len)
    pad = np.concatenate((sig, np.zeros((padlen - slen,))))
    shape = pad.shape[:-1] + (pad.shape[-1] - frame_len + 1, frame_len)
    return np.lib.stride_tricks.as_strided(pad, shape=shape, strides=pad.strides + (pad.strides[-1],))[::frame_step]


def _hz2mel(hz):                                                 # base.py line 139
    return 2595 * np.log10(1 + hz / 700.0)


def _mel2hz(mel):                                                # base.py line 147
    return 700 * (10 ** (mel / 2595.0) - 1)


def _filterbanks(nfilt: int, nfft: int, samplerate: int) -> np.ndarray:          # base.py lines 149-177 (lowfreq 0, highfreq sr/2)
    melpoints = np.linspace(_hz2mel(0), _hz2mel(samplerate / 2), nfilt + 2)
    b = np.floor((nfft + 1) * _mel2hz(melpoints) / samplerate)
    fb = np.zeros([nfilt, nfft // 2 + 1])
    for j in range(nfilt):
        for i in range(int(b[j]), int(b[j + 1])):
            fb[j, i] = (i - b[j]) / (b[j + 1] - b[j])
        for i in range(int(b[j + 1]), int(b[j + 2])):
            fb[j, i] = (b[j + 2] - i) / (b[j + 2] - b[j + 1])
    return fb


def mfcc(audio: np.ndarray, samplerate: int = SAMPLE_RATE) -> np.ndarray:
    """MFCC as python_speech_features.mfcc(audio, samplerate) with defaults: array (n_frames, 13). ``audio`` must be on the int16
    scale (the reference passes the int16 array from wavfile.read, SI line 200); the log energy and the filterbank are not scale-free."""
    audio = np.asarray(audio, dtype=np.float64)
    nfft = 1
    while nfft < MFCC_WINLEN * samplerate:                       # base.py lines 19-22 (calculate_nfft): 512 at 16 kHz
        nfft *= 2
    sig = np.append(audio[0], audio[1:] - MFCC_PREEMPH * audio[:-1])                    # sigproc.preemphasis, line 140
    frames = _framesig(sig, MFCC_WINLEN * samplerate, MFCC_WINSTEP * samplerate)
    pspec = 1.0 / nfft * np.square(np.absolute(np.fft.rfft(frames, nfft)))             # sigproc.powspec, lines 102-113
    energy = np.sum(pspec, 1)
    energy = np.where(energy == 0, np.finfo(float).eps, energy)                         # base.py line 76
    feat = np.dot(pspec, _filterbanks(MFCC_NFILT, nfft, samplerate).T)
    feat = np.where(feat == 0, np.finfo(float).eps, feat)                               # base.py line 80
    feat = dct(np.log(feat), type=2, axis=1, norm="ortho")[:, :MFCC_NUMCEP]
    n = np.arange(MFCC_NUMCEP)
    feat = (1 + (MFCC_CEPLIFTER / 2.0) * np.sin(np.pi * n / MFCC_CEPLIFTER)) * feat     # base.py lines 186-190 (lifter)
    feat[:, 0] = np.log(energy)                                                         # appendEnergy=True, base.py line 50
    return feat


def to_int16_scale(wav_f32: np.ndarray) -> np.ndarray:
    """float32 in [-1, 1] to the int16 scale as an array of integers (equivalent to ffmpeg ``-acodec pcm_s16le``, SI line 174)."""
    return np.clip(np.round(np.asarray(wav_f32, np.float64) * 32768.0), -32768, 32767)


def audio_mfcc_tensor(wav_f32: np.ndarray) -> np.ndarray:
    """MFCC laid out as the reference does (SI lines 201-204): shape (13, n_mfcc_frames), float32."""
    return mfcc(to_int16_scale(wav_f32)).T.astype(np.float32)


# ---- video ---------------------------------------------------------------------------------------------------------------------

def video_size(path: str | Path) -> tuple[int, int]:
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if w <= 0 or h <= 0:
        raise RuntimeError(f"cannot read the frame size of {path}")
    return w, h


def iter_frames_25fps(path: str | Path, max_seconds: float | None = None):
    """Yield BGR uint8 frames resampled to 25 fps by ffmpeg (``-r 25``, RP line 267). One ffmpeg process, raw frames on a pipe."""
    w, h = video_size(path)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-i", str(path), "-an", "-sn", "-r", str(FRAME_RATE)]
    if max_seconds:
        cmd += ["-t", f"{max_seconds:.3f}"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = w * h * 3
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


def smooth_boxes(boxes_xyxy: np.ndarray) -> np.ndarray:
    """Per-frame (s, x, y) smoothed as RP lines 122-131: s = half of the larger box side, (x, y) the centre, median filter of 13."""
    b = np.asarray(boxes_xyxy, np.float64)
    s = np.maximum(b[:, 3] - b[:, 1], b[:, 2] - b[:, 0]) / 2
    y = (b[:, 1] + b[:, 3]) / 2
    x = (b[:, 0] + b[:, 2]) / 2
    return np.stack([signal.medfilt(s, CROP_MEDFILT), signal.medfilt(x, CROP_MEDFILT), signal.medfilt(y, CROP_MEDFILT)], 1)


def crop_face(frame: np.ndarray, s: float, x: float, y: float) -> np.ndarray:
    """The 224x224 face crop of RP lines 135-148: region [y - bs, y + bs(1+2cs)) x [x - bs(1+cs), x + bs(1+cs)) of the frame padded
    with 110, then resized. The padding is applied only where the region leaves the frame (same pixels as padding the whole frame)."""
    import cv2
    cs = CROP_SCALE
    y0, y1 = int(y - s), int(y + s * (1 + 2 * cs))
    x0, x1 = int(x - s * (1 + cs)), int(x + s * (1 + cs))
    h, w = frame.shape[:2]
    out = np.full((max(y1 - y0, 1), max(x1 - x0, 1), 3), CROP_FILL, np.uint8)
    sy0, sy1, sx0, sx1 = max(y0, 0), min(y1, h), max(x0, 0), min(x1, w)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    return cv2.resize(out, (FACE_SIZE, FACE_SIZE))
