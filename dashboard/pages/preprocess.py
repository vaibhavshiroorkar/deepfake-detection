"""Single-page preprocessing pipeline (PROJECT_OVERVIEW.md §7).

Reads top-to-bottom as a pipeline: every step is applied cumulatively and shows
the result AFTER it runs, so toggling a step visibly changes everything below it.
Ends with the exact tensor the model receives. Read-only — never writes
data/processed/, never trains. Reuses dashboard/lib ops verbatim.
"""
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from dashboard.lib import selectors, media, visual_ops as V, audio_ops as A

DATA_DIR = _REPO_ROOT / "data"

st.title("Preprocessing")
st.caption("One sequential pipeline. Each step is applied cumulatively and shows the "
           "result after it runs; the last step is exactly what the model receives. "
           "Read-only — never writes data/processed/, never trains.")

# ---- 1. Selection --------------------------------------------------------- #
row = selectors.render_selection()
if row is None:
    st.stop()

video_path = DATA_DIR / row["video_path"]
if not video_path.exists():
    st.error(f"Video not found: {video_path}")
    st.stop()

s1, s2 = st.columns(2)
with s1:
    n_frames = st.slider("Frames (N)", 4, 32, 16)
with s2:
    window_sec = st.slider("Audio window (s)", 0.10, 1.00, 0.35, 0.05)

duration, fps = media.frame_meta(str(video_path))
timestamps = media.sample_timestamps(duration, n_frames, window_sec)


def show_frames(container, frames, caption):
    """Render all N frames as a compact grid (no single large preview)."""
    container.caption(caption)
    for start in range(0, len(frames), 8):
        cols = container.columns(8)
        for j, col in enumerate(cols):
            if start + j < len(frames):
                col.image(frames[start + j], width="stretch")


def skipped(container):
    container.caption("skipped — passthrough")


# ========================= 2. VISUAL PIPELINE ============================== #
st.divider()
st.header("Visual pipeline")

# Stage: decode (baseline)
full_frames = media.decode_frames(str(video_path), timestamps)
cur = [cv2.resize(f, (224, 224), interpolation=cv2.INTER_CUBIC) for f in full_frames]
with st.container(border=True):
    l, r = st.columns([1, 2])
    l.markdown("**0 · Decode**")
    l.caption(f"{n_frames} frames sampled across {duration:.2f}s @ {fps:.1f} fps.")
    show_frames(r, cur, "Original")

# Stage: face detect + crop
with st.container(border=True):
    l, r = st.columns([1, 2])
    l.markdown("**1 · Face detection + crop**")
    do_detect = l.checkbox("Enable MTCNN crop", value=True)
    conf = l.slider("Confidence", 0.50, 0.99, 0.90, 0.01, disabled=not do_detect)
    margin = l.slider("Crop margin", 0.0, 0.6, 0.20, 0.05, disabled=not do_detect)
    if do_detect:
        detector, device = media.get_detector()
        l.caption(f"Detector on {device}.")
        out, flags = [], []
        for f in full_frames:
            crop, det = media.detect_and_crop(f, detector, conf, margin)
            out.append(crop)
            flags.append(det)
        cur = out
        l.metric("Faces detected", f"{sum(flags)}/{n_frames}")
        show_frames(r, cur, "Cropped")
    else:
        skipped(r)
        show_frames(r, cur, "Full frame")

# Stage: quality & robustness
with st.container(border=True):
    l, r = st.columns([1, 2])
    l.markdown("**2 · Quality & robustness**")
    do_sharpen = l.checkbox("Sharpen")
    sharpen_amt = l.slider("amount", 0.0, 3.0, 1.0, disabled=not do_sharpen, key="v_sharp")
    do_denoise = l.checkbox("Denoise")
    denoise_str = l.slider("strength", 1, 20, 5, disabled=not do_denoise, key="v_den")
    do_clahe = l.checkbox("CLAHE contrast")
    clahe_clip = l.slider("clip", 1.0, 8.0, 2.0, disabled=not do_clahe, key="v_clahe")
    do_blur = l.checkbox("Gaussian blur")
    blur_k = l.slider("kernel", 3, 31, 9, step=2, disabled=not do_blur, key="v_blur")
    do_jpeg = l.checkbox("JPEG re-compress")
    jpeg_q = l.slider("quality", 5, 95, 30, disabled=not do_jpeg, key="v_jpeg")
    do_ds = l.checkbox("Downscale→upscale")
    ds_factor = l.slider("scale", 0.1, 0.9, 0.25, disabled=not do_ds, key="v_ds")

    any_q = any([do_sharpen, do_denoise, do_clahe, do_blur, do_jpeg, do_ds])
    if any_q:
        out = []
        for img in cur:
            if do_sharpen:
                img = V.sharpen(img, sharpen_amt)
            if do_denoise:
                img = V.denoise(img, denoise_str)
            if do_clahe:
                img = V.clahe(img, clahe_clip)
            if do_blur:
                img = V.gaussian_blur(img, blur_k)
            if do_jpeg:
                img = V.jpeg_recompress(img, jpeg_q)
            if do_ds:
                img = V.downscale_upscale(img, ds_factor)
            out.append(img)
        cur = out
        show_frames(r, cur, "After quality steps")
    else:
        skipped(r)
        show_frames(r, cur, "Unchanged")

# Stage: mouth crop
with st.container(border=True):
    l, r = st.columns([1, 2])
    l.markdown("**3 · Mouth-region crop (96²)**")
    do_mouth = l.checkbox("Enable mouth crop")
    l.caption("For the lip-sync stream later. Approximated from the face crop.")
    if do_mouth:
        cur = [V.mouth_region(img, 96) for img in cur]
        show_frames(r, cur, "Mouth 96²")
    else:
        skipped(r)
        show_frames(r, cur, "Face crop")

# Stage: ImageNet normalize
with st.container(border=True):
    l, r = st.columns([1, 2])
    l.markdown("**4 · ImageNet normalize**")
    do_norm = l.checkbox("Enable normalize", value=True)
    l.caption("Zero-centers pixels the way the ImageNet-pretrained backbones expect.")
    norm_frames = cur  # what the visual backbone actually consumes
    if do_norm:
        arrs = [V.imagenet_normalize(img) for img in cur]
        lo, hi = V.normalized_range(arrs[len(arrs) // 2])
        l.metric("Pixel range", f"[{lo:.2f}, {hi:.2f}]")
        # De-normalize purely for display (grids stay uint8).
        disp = [np.clip((a * V.IMAGENET_STD + V.IMAGENET_MEAN) * 255, 0, 255).astype(np.uint8)
                for a in arrs]
        show_frames(r, disp, "Normalized (shown de-normalized)")
        model_faces = np.stack([np.transpose(a, (2, 0, 1)) for a in arrs]).astype(np.float32)
    else:
        skipped(r)
        show_frames(r, cur, "Raw [0,255]")
        model_faces = np.stack([np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))
                                for img in cur])

# ========================= 3. AUDIO PIPELINE =============================== #
st.divider()
st.header("Audio pipeline")

raw2d, native_sr = media.decode_audio(str(video_path))
has_audio = raw2d.size > 0


def waveform_fig(y, rate, title):
    fig, ax = plt.subplots(figsize=(10, 1.9))
    if y.size:
        ax.plot(np.arange(y.size) / rate, y, linewidth=0.5, color="#3b82f6")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("s", fontsize=8)
    ax.margins(x=0)
    return fig


if not has_audio:
    st.warning("No audio stream in this clip.")
    model_audio = np.zeros((n_frames, int(window_sec * media.AUDIO_SR)), np.float32)
else:
    wav = raw2d
    sr = native_sr

    # Stage: decode baseline (downmixed for display)
    with st.container(border=True):
        l, r = st.columns([1, 2])
        l.markdown("**0 · Decode**")
        l.caption(f"Native sample rate {native_sr} Hz, {raw2d.shape[0]} channel(s).")
        r.pyplot(waveform_fig(A.downmix(raw2d), native_sr, "Original (downmixed)"))

    # Stage: mono downmix
    with st.container(border=True):
        l, r = st.columns([1, 2])
        l.markdown("**1 · Mono downmix**")
        do_downmix = l.checkbox("Enable downmix", value=True)
        if do_downmix:
            wav = A.downmix(wav)
            r.pyplot(waveform_fig(wav, sr, "Mono"))
        else:
            wav = raw2d.mean(axis=0)
            skipped(r)
            r.pyplot(waveform_fig(wav, sr, "Channel-averaged for display"))

    # Stage: resample
    with st.container(border=True):
        l, r = st.columns([1, 2])
        l.markdown("**2 · Resample**")
        do_resample = l.checkbox("Enable resample", value=True)
        target_sr = l.select_slider("target SR", [8000, 16000, 22050, 44100], 16000,
                                    disabled=not do_resample)
        if do_resample:
            wav = A.resample(wav, sr, target_sr)
            sr = target_sr
            r.pyplot(waveform_fig(wav, sr, f"Resampled to {sr} Hz"))
        else:
            skipped(r)
            r.pyplot(waveform_fig(wav, sr, f"Native {sr} Hz"))

    # Stage: trim leading silence
    with st.container(border=True):
        l, r = st.columns([1, 2])
        l.markdown("**3 · Trim leading silence**")
        do_trim = l.checkbox("Enable trim")
        top_db = l.slider("top_db", 10.0, 60.0, 30.0, disabled=not do_trim)
        if do_trim:
            wav, dropped = A.trim_silence(wav, sr, top_db)
            l.metric("Dropped", f"{dropped:.3f}s")
            r.pyplot(waveform_fig(wav, sr, "Trimmed"))
        else:
            skipped(r)
            r.pyplot(waveform_fig(wav, sr, "Untrimmed"))

    # Stage: quality & robustness
    with st.container(border=True):
        l, r = st.columns([1, 2])
        l.markdown("**4 · Quality & robustness**")
        do_aden = l.checkbox("Noise reduction")
        aden_str = l.slider("strength", 0.5, 3.0, 1.0, disabled=not do_aden, key="a_den")
        do_rms = l.checkbox("RMS normalize")
        rms_db = l.slider("target dB", -30.0, -6.0, -20.0, disabled=not do_rms, key="a_rms")
        do_band = l.checkbox("Bandpass")
        band = l.slider("Hz", 50, 8000, (300, 3000), disabled=not do_band, key="a_band")
        do_addnoise = l.checkbox("Add background noise")
        snr = l.slider("SNR dB", 0.0, 40.0, 20.0, disabled=not do_addnoise, key="a_snr")
        if any([do_aden, do_rms, do_band, do_addnoise]):
            if do_aden:
                wav = A.spectral_denoise(wav, sr, aden_str)
            if do_rms:
                wav = A.rms_normalize(wav, rms_db)
            if do_band:
                wav = A.bandpass(wav, sr, float(band[0]), float(band[1]))
            if do_addnoise:
                wav = A.add_noise(wav, snr, rng=np.random.default_rng(0))
            r.pyplot(waveform_fig(wav, sr, "After quality steps"))
        else:
            skipped(r)
            r.pyplot(waveform_fig(wav, sr, "Unchanged"))

    # Stage: window extraction
    with st.container(border=True):
        l, r = st.columns([1, 2])
        l.markdown("**5 · Window extraction**")
        l.caption(f"One {window_sec:.2f}s window centered on each of the {n_frames} frames.")
        windows = A.extract_windows(wav, sr, timestamps, window_sec)
        # resample windows to the model's 16 kHz contract if needed
        if sr != media.AUDIO_SR:
            model_audio = np.stack([A.resample(w, sr, media.AUDIO_SR) for w in windows])
        else:
            model_audio = windows
        l.metric("Windows", f"{windows.shape[0]} × {windows.shape[1]}")
        r.pyplot(waveform_fig(windows.reshape(-1), sr, "Windows concatenated"))

    # Stage: mel-spectrogram view
    with st.container(border=True):
        l, r = st.columns([1, 2])
        l.markdown("**6 · Mel-spectrogram view**")
        do_mel = l.checkbox("Show mel-spectrogram")
        n_mels = l.slider("n_mels", 32, 128, 64, disabled=not do_mel)
        hop = l.slider("hop", 128, 512, 256, step=64, disabled=not do_mel)
        if do_mel:
            mel = A.mel_spectrogram(wav, sr, n_mels, hop)
            fig, ax = plt.subplots(figsize=(10, 2.2))
            im = ax.imshow(mel, aspect="auto", origin="lower", cmap="magma")
            fig.colorbar(im, ax=ax, format="%+.0f dB")
            ax.set_title(f"Mel ({n_mels} mels)", fontsize=9)
            r.pyplot(fig)
        else:
            skipped(r)

# ========================= 4. MODEL INPUT ================================== #
st.divider()
st.header("Model input")
st.caption("This is exactly what the DataLoader hands the model for this clip.")

mi1, mi2, mi3 = st.columns(3)
mi1.metric("Faces", f"{tuple(model_faces.shape)}")
mi2.metric("Audio", f"{tuple(model_audio.shape)}")
mi3.metric("Label", f"{int(row['label'])}  ({'real' if row['label'] == 0 else 'fake'})")

st.markdown("**Face tensor** `[N, 3, 224, 224]` float32 — shown de-normalized:")
disp_model = []
for f in model_faces:
    hwc = np.transpose(f, (1, 2, 0))
    if do_norm:
        hwc = hwc * V.IMAGENET_STD + V.IMAGENET_MEAN
    disp_model.append(np.clip(hwc * 255, 0, 255).astype(np.uint8))
for start in range(0, len(disp_model), 8):
    cols = st.columns(8)
    for j, col in enumerate(cols):
        if start + j < len(disp_model):
            col.image(disp_model[start + j], width="stretch")

st.code(
    f"faces : shape={tuple(model_faces.shape)} dtype={model_faces.dtype} "
    f"range=[{model_faces.min():.3f}, {model_faces.max():.3f}]\n"
    f"audio : shape={tuple(model_audio.shape)} dtype={model_audio.dtype} "
    f"range=[{model_audio.min():.3f}, {model_audio.max():.3f}]\n"
    f"label : {int(row['label'])}",
    language="text",
)
