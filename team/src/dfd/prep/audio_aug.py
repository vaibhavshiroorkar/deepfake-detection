"""Training-time nuisance augmentation of speech windows (hypothesis H-A2, E-008).

The transform family T of MATH_NOTES 14 for speech: additive noise, band-limiting, synthetic reverberation and an MP3 round trip. Each
transform is applied independently with the probability below; if none was drawn, the noise is applied, so every copy differs from the
clean window. The parameter ranges were fixed before the run from the nuisance types named in M-004 (recording condition, channel) and
were not tuned on dev_x. The peak is renormalised to 1 at the end, so level stays removed (H-A1). float32 in, float32 out, 16 kHz.
Training copies only: validation, dev_x and inference never see this module.
"""
from __future__ import annotations

import subprocess

import numpy as np
from scipy import signal

SR = 16000
P_NOISE, P_LOWPASS, P_REVERB, P_MP3 = 0.7, 0.5, 0.5, 0.5
SNR_DB = (5.0, 30.0)
CUTOFF_HZ = (3400.0, 7000.0)
RT60_S = (0.15, 0.8)
MP3_KBPS = (24, 32, 48, 64, 96, 128)


def _noise(x, rng):
    snr = rng.uniform(*SNR_DB)
    n = rng.standard_normal(x.size).astype(np.float32)
    if rng.random() < 0.5:   # coloured: one-pole low-pass, closer to room and channel noise than white noise
        n = signal.lfilter([1.0], [1.0, -0.9], n).astype(np.float32)
    p_x, p_n = float(np.mean(x ** 2)) + 1e-12, float(np.mean(n ** 2)) + 1e-12
    return x + n * np.sqrt(p_x / (p_n * 10 ** (snr / 10)))


def _lowpass(x, rng):
    fc = rng.uniform(*CUTOFF_HZ)
    sos = signal.butter(6, fc / (SR / 2), btype="low", output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)


def _reverb(x, rng):
    rt60 = rng.uniform(*RT60_S)
    t = np.arange(int(0.6 * SR)) / SR
    rir = rng.standard_normal(t.size) * np.exp(-6.9078 * t / rt60)   # amplitude decays by 60 dB in rt60 seconds
    rir[0] = 1.0                                                      # direct path
    rir /= np.sqrt(np.sum(rir ** 2))
    return signal.fftconvolve(x, rir)[: x.size].astype(np.float32)


def _mp3(x, rng):
    kbps = int(rng.choice(MP3_KBPS))
    pcm = np.clip(np.round(x * 32767.0), -32768, 32767).astype("<i2").tobytes()
    enc = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
                          "-c:a", "libmp3lame", "-b:a", f"{kbps}k", "-f", "mp3", "pipe:1"], input=pcm, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if enc.returncode != 0:
        raise RuntimeError("ffmpeg mp3 encode failed: " + enc.stderr.decode(errors="replace")[:200])
    dec = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "mp3", "-i", "pipe:0", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
                         input=enc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if dec.returncode != 0:
        raise RuntimeError("ffmpeg mp3 decode failed: " + dec.stderr.decode(errors="replace")[:200])
    y = np.frombuffer(dec.stdout, np.float32)
    out = np.zeros(x.size, np.float32)
    k = min(x.size, y.size)   # the encoder delay shifts the signal by about 1,100 samples; it is left as it is (a nuisance too)
    out[:k] = y[:k]
    return out


def augment(wav: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    """Return the augmented window and the names of the transforms applied (order: reverb, low-pass, MP3, noise)."""
    x = wav.astype(np.float32)
    chosen = [n for n, p in (("reverb", P_REVERB), ("lowpass", P_LOWPASS), ("mp3", P_MP3), ("noise", P_NOISE)) if rng.random() < p]
    if not chosen:
        chosen = ["noise"]
    fn = dict(reverb=_reverb, lowpass=_lowpass, mp3=_mp3, noise=_noise)
    for n in ("reverb", "lowpass", "mp3", "noise"):
        if n in chosen:
            x = fn[n](x, rng)
    peak = float(np.abs(x).max())
    return ((x / peak) if peak > 0 else x).astype(np.float32), chosen
