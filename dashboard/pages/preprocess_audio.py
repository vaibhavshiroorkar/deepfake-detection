import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from dashboard.lib import selectors, media, audio_ops as A

st.title("Audio — preprocessing")
st.caption("Toggle steps to experiment. Baseline (all off) = mono downmix + resample "
           "+ window. Original is always shown beside the processed result.")

row = selectors.render_selection()
if row is None:
    st.stop()

DATA_DIR = _REPO_ROOT / "data"
video_path = DATA_DIR / row["video_path"]
if not video_path.exists():
    st.error(f"Video not found: {video_path}")
    st.stop()

st.sidebar.header("Audio steps")
with st.sidebar.expander("Core", expanded=True):
    do_downmix = st.checkbox("Mono downmix", value=True)
    do_resample = st.checkbox("Resample 16 kHz", value=True)
    target_sr = st.select_slider("  target SR", [8000, 16000, 22050, 44100], 16000,
                                 disabled=not do_resample)
    window_sec = st.slider("Window (s)", 0.10, 1.00, 0.35, 0.05)
with st.sidebar.expander("Representation"):
    do_trim = st.checkbox("Trim leading silence")
    top_db = st.slider("  top_db", 10.0, 60.0, 30.0, disabled=not do_trim)
    do_mel = st.checkbox("Mel-spectrogram view")
    n_mels = st.slider("  n_mels", 32, 128, 64, disabled=not do_mel)
    hop = st.slider("  hop", 128, 512, 256, step=64, disabled=not do_mel)
with st.sidebar.expander("Quality & robustness"):
    do_denoise = st.checkbox("Noise reduction")
    denoise_str = st.slider("  strength", 0.5, 3.0, 1.0, disabled=not do_denoise)
    do_rms = st.checkbox("RMS normalize")
    rms_db = st.slider("  target dB", -30.0, -6.0, -20.0, disabled=not do_rms)
    do_band = st.checkbox("Bandpass")
    band = st.slider("  Hz", 50, 8000, (300, 3000), disabled=not do_band)
    do_addnoise = st.checkbox("Add background noise")
    snr = st.slider("  SNR dB", 0.0, 40.0, 20.0, disabled=not do_addnoise)

raw2d, native_sr = media.decode_audio(str(video_path))
# Baseline original for comparison: downmix + resample to AUDIO_SR (what the pipeline stores).
orig = A.resample(A.downmix(raw2d), native_sr, media.AUDIO_SR)

wav = A.downmix(raw2d) if do_downmix else raw2d.mean(axis=0)
sr = native_sr
if do_resample:
    wav = A.resample(wav, sr, target_sr)
    sr = target_sr
dropped = 0.0
if do_trim:
    wav, dropped = A.trim_silence(wav, sr, top_db)
    st.info(f"Leading silence trimmed: {dropped:.3f}s dropped.")
if do_denoise:
    wav = A.spectral_denoise(wav, sr, denoise_str)
if do_rms:
    wav = A.rms_normalize(wav, rms_db)
if do_band:
    wav = A.bandpass(wav, sr, float(band[0]), float(band[1]))
if do_addnoise:
    wav = A.add_noise(wav, snr, rng=np.random.default_rng(0))

duration, _ = media.frame_meta(str(video_path))
ts = media.sample_timestamps(duration, 16, window_sec)


def waveform_fig(y, rate, title, shift=0.0):
    fig, ax = plt.subplots(figsize=(11, 2.2))
    if y.size:
        ax.plot(np.arange(y.size) / rate, y, linewidth=0.5, color="#3b82f6")
        win = int(window_sec * rate)
        for t in ts:
            c = int((t - shift) * rate)
            s = max(0, c - win // 2)
            e = min(y.size, s + win)
            ax.axvspan(s / rate, e / rate, color="#f59e0b", alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("s")
    ax.margins(x=0)
    return fig


if orig.size == 0:
    st.warning("No audio stream in this clip.")
    st.stop()

st.subheader("Original")
st.pyplot(waveform_fig(orig, media.AUDIO_SR, "Original waveform (downmix+16kHz)"))
st.audio(orig, sample_rate=media.AUDIO_SR)

st.subheader("Processed")
st.pyplot(waveform_fig(wav, sr, f"Processed waveform (sr={sr})", shift=dropped))
if wav.size:
    st.audio(wav, sample_rate=sr)
if do_mel and wav.size:
    mel = A.mel_spectrogram(wav, sr, n_mels, hop)
    fig, ax = plt.subplots(figsize=(11, 2.6))
    im = ax.imshow(mel, aspect="auto", origin="lower", cmap="magma")
    ax.set_title(f"Mel-spectrogram ({n_mels} mels)")
    fig.colorbar(im, ax=ax, format="%+.0f dB")
    st.pyplot(fig)

st.divider()
st.code({
    "downmix": do_downmix, "resample": do_resample and target_sr, "window_s": window_sec,
    "trim": do_trim and top_db, "mel": do_mel and (n_mels, hop),
    "denoise": do_denoise and denoise_str, "rms": do_rms and rms_db,
    "bandpass": do_band and band, "add_noise": do_addnoise and snr,
}, language="python")
