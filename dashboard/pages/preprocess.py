"""Single preprocessing page with Config / Visual / Audio tabs (§7).

Tabs switch which section shows:
  - Config: pick the clip (Selection) and global settings, and see the model-input
    contract + a raw preview. The chosen clip and settings are cached in
    session_state for the other tabs.
  - Visual: the visual pipeline, each step applied cumulatively, ending in the
    face tensor the model receives.
  - Audio: the audio pipeline, same treatment, ending in the audio windows.

Note: st.tabs renders all three tab bodies every run, so Visual/Audio recompute
whenever anything changes — the "select a clip first" guards keep them cheap
until a clip is chosen in Config. Read-only — never writes data/processed/,
never trains. Reuses dashboard/lib ops.
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

from dashboard.lib import selectors, media
from preprocessing.ops import (
    faces as VF, audio as AF, extras_visual as VX, extras_audio as AX, constants as C,
)

DATA_DIR = _REPO_ROOT / "data"

st.title("Preprocessing")
st.caption("Pick a clip and settings under Config; inspect the Visual and Audio "
           "pipelines step by step. Each step is applied cumulatively and shows the "
           "result after it runs. Read-only — never writes data/processed/, never trains.")


def show_frames(container, frames, caption):
    """Render all N frames as a compact grid."""
    container.caption(caption)
    for start in range(0, len(frames), 8):
        cols = container.columns(8)
        for j, col in enumerate(cols):
            if start + j < len(frames):
                col.image(frames[start + j], width="stretch")


def skipped(container):
    container.caption("skipped — passthrough")


def waveform_fig(y, rate, title, figsize=(10, 1.9)):
    fig, ax = plt.subplots(figsize=figsize)
    if y.size:
        ax.plot(np.arange(y.size) / rate, y, linewidth=0.5, color="#3b82f6")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("s", fontsize=8)
    ax.margins(x=0)
    return fig


def cached_clip():
    """(row_dict, video_path, n_frames, window_sec) from the Config selection, or None."""
    if "pp_row" not in st.session_state:
        return None
    return (
        st.session_state["pp_row"],
        Path(st.session_state["pp_video_path"]),
        int(st.session_state.get("pp_n_frames", 16)),
        float(st.session_state.get("pp_window", 0.35)),
    )


def _prepared_clip():
    """Shared guard for the Visual/Audio tabs: cached clip + decode metadata, or None."""
    clip = cached_clip()
    if clip is None:
        st.info("Select a clip in the **Config** tab first.")
        return None
    row, video_path, n_frames, window_sec = clip
    if not video_path.exists():
        st.error(f"Video not found: {video_path}")
        return None
    duration, fps = media.frame_meta(str(video_path))
    timestamps = media.sample_timestamps(duration, n_frames, window_sec)
    return row, video_path, n_frames, window_sec, duration, fps, timestamps


# ============================== CONFIG ===================================== #
def render_config():
    row = selectors.render_selection()
    if row is None:
        return

    video_path = DATA_DIR / row["video_path"]
    if not video_path.exists():
        st.error(f"Video not found: {video_path}")
        return

    g1, g2 = st.columns(2)
    with g1:
        n_frames = st.slider("Frames (N)", 4, 32, 16, key="pp_n_frames")
    with g2:
        window_sec = st.slider("Audio window (s)", 0.10, 1.00, 0.35, 0.05, key="pp_window")

    # Cache the selection so the Visual and Audio tabs can read it.
    st.session_state["pp_row"] = row.to_dict()
    st.session_state["pp_video_path"] = str(video_path)

    st.divider()
    st.header("Preview")

    # The values the old Configuration section showed: the model-input contract.
    win_samples = int(window_sec * media.AUDIO_SR)
    m1, m2, m3 = st.columns(3)
    m1.metric("Faces", f"[{n_frames}, 3, 224, 224]")
    m2.metric("Audio", f"[{n_frames}, {win_samples}]")
    m3.metric("Label", f"{int(row['label'])}  ({'real' if row['label'] == 0 else 'fake'})")
    st.caption(f"Clip `{row['clip_id']}` — raw frames and audio before any preprocessing.")

    duration, fps = media.frame_meta(str(video_path))
    timestamps = media.sample_timestamps(duration, n_frames, window_sec)
    prev = [cv2.resize(f, (224, 224), interpolation=cv2.INTER_CUBIC)
            for f in media.decode_frames(str(video_path), timestamps)]

    # Compact preview: all N frames, 8 per row, in a narrow column (keeps the
    # thumbnail size fixed by always laying out 8 columns per row).
    left, _ = st.columns([3, 2])
    for start in range(0, len(prev), 8):
        cols = left.columns(8)
        for j, col in enumerate(cols):
            if start + j < len(prev):
                col.image(prev[start + j], width="stretch")

    raw2d, native_sr = media.decode_audio(str(video_path))
    if raw2d.size:
        mono = AF.downmix(raw2d)
        aleft, _ = st.columns([3, 2])
        aleft.pyplot(waveform_fig(mono, native_sr, f"Audio ({native_sr} Hz)", figsize=(6, 1.3)))
        aleft.audio(mono, sample_rate=native_sr)
    else:
        st.caption("No audio stream in this clip.")


# ============================== VISUAL ===================================== #
def render_visual():
    prep = _prepared_clip()
    if prep is None:
        return
    row, video_path, n_frames, window_sec, duration, fps, timestamps = prep

    st.header("Visual pipeline")

    full_frames = media.decode_frames(str(video_path), timestamps)
    cur = [cv2.resize(f, (224, 224), interpolation=cv2.INTER_CUBIC) for f in full_frames]

    with st.container(border=True):
        l, r = st.columns([1, 2])
        l.markdown("**0 · Decode**")
        l.caption(f"{n_frames} frames across {duration:.2f}s @ {fps:.1f} fps.")
        show_frames(r, cur, "Original")

    with st.container(border=True):
        l, r = st.columns([1, 2])
        l.markdown("**1 · Face detection + align + crop**")
        do_detect = l.checkbox("Enable MTCNN crop", value=True, key="v_detect")
        do_align = l.checkbox("5-point align", value=True, disabled=not do_detect, key="v_align")
        l.caption("Alignment warps the face onto a canonical 5-point template "
                  "(pose-normalized). Off = plain bbox crop.")
        conf = l.slider("Confidence", 0.50, 0.99, 0.90, 0.01, disabled=not do_detect, key="v_conf")
        margin = l.slider("Crop margin", 0.0, 0.6, 0.20, 0.05, disabled=not do_detect, key="v_margin")
        mouth_crops = None
        if do_detect:
            detector, device = media.get_detector()
            l.caption(f"Detector on {device}.")
            faces, mouths, flags = [], [], []
            for f in full_frames:
                face, mouth, det = media.detect_face_and_mouth(
                    f, detector, conf, margin, align=do_align)
                faces.append(face)
                mouths.append(mouth)
                flags.append(det)
            cur = faces
            mouth_crops = mouths          # parallel branch, kept for the mouth stage
            l.metric("Faces detected", f"{sum(flags)}/{n_frames}")
            show_frames(r, cur, "Cropped")
        else:
            skipped(r)
            show_frames(r, cur, "Full frame")

    with st.container(border=True):
        l, r = st.columns([1, 2])
        l.markdown("**2 · Quality & robustness**")
        do_sharpen = l.checkbox("Sharpen", key="v_sharpen_on")
        sharpen_amt = l.slider("amount", 0.0, 3.0, 1.0, disabled=not do_sharpen, key="v_sharp")
        do_denoise = l.checkbox("Denoise", key="v_denoise_on")
        denoise_str = l.slider("strength", 1, 20, 5, disabled=not do_denoise, key="v_den")
        do_clahe = l.checkbox("CLAHE contrast", key="v_clahe_on")
        clahe_clip = l.slider("clip", 1.0, 8.0, 2.0, disabled=not do_clahe, key="v_clahe")
        do_blur = l.checkbox("Gaussian blur", key="v_blur_on")
        blur_k = l.slider("kernel", 3, 31, 9, step=2, disabled=not do_blur, key="v_blur")
        do_jpeg = l.checkbox("JPEG re-compress", key="v_jpeg_on")
        jpeg_q = l.slider("quality", 5, 95, 30, disabled=not do_jpeg, key="v_jpeg")
        do_ds = l.checkbox("Downscale→upscale", key="v_ds_on")
        ds_factor = l.slider("scale", 0.1, 0.9, 0.25, disabled=not do_ds, key="v_ds")

        if any([do_sharpen, do_denoise, do_clahe, do_blur, do_jpeg, do_ds]):
            out = []
            for img in cur:
                if do_sharpen:
                    img = VX.sharpen(img, sharpen_amt)
                if do_denoise:
                    img = VX.denoise(img, denoise_str)
                if do_clahe:
                    img = VX.clahe(img, clahe_clip)
                if do_blur:
                    img = VX.gaussian_blur(img, blur_k)
                if do_jpeg:
                    img = VX.jpeg_recompress(img, jpeg_q)
                if do_ds:
                    img = VX.downscale_upscale(img, ds_factor)
                out.append(img)
            cur = out
            show_frames(r, cur, "After quality steps")
        else:
            skipped(r)
            show_frames(r, cur, "Unchanged")

    with st.container(border=True):
        l, r = st.columns([1, 2])
        l.markdown("**3 · ImageNet normalize** (face path)")
        do_norm = l.checkbox("Enable normalize", value=True, key="v_norm")
        l.caption("Zero-centers pixels the way the pretrained backbones expect.")
        if do_norm:
            arrs = [VF.imagenet_normalize(img) for img in cur]
            lo, hi = VF.normalized_range(arrs[len(arrs) // 2])
            l.metric("Pixel range", f"[{lo:.2f}, {hi:.2f}]")
            disp = [np.clip((a * C.IMAGENET_STD + C.IMAGENET_MEAN) * 255, 0, 255).astype(np.uint8)
                    for a in arrs]
            show_frames(r, disp, "Normalized (shown de-normalized)")
            model_faces = np.stack([np.transpose(a, (2, 0, 1)) for a in arrs]).astype(np.float32)
        else:
            skipped(r)
            show_frames(r, cur, "Raw [0,255]")
            model_faces = np.stack([np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))
                                    for img in cur])

    # Parallel branch: the mouth crop is a SEPARATE output for the lip-sync
    # stream (AV-HuBERT). It does not replace the face — the visual and
    # emotion streams still consume the face crop above.
    model_mouth = None
    with st.container(border=True):
        l, r = st.columns([1, 2])
        l.markdown("**⑃ Mouth branch** (lip-sync input, parallel)")
        do_mouth = l.checkbox("Produce 96² mouth crop", key="v_mouth")
        l.caption("Landmark-derived mouth region for the lip-sync stream. Runs "
                  "alongside the face crop — the visual/emotion streams keep the face.")
        if do_mouth:
            if mouth_crops is None:
                r.warning("Enable face detection above to derive landmark mouth crops.")
            else:
                show_frames(r, mouth_crops, "Mouth 96² (from mouth-corner landmarks)")
                model_mouth = np.stack([np.transpose(m.astype(np.float32) / 255.0, (2, 0, 1))
                                        for m in mouth_crops]).astype(np.float32)
        else:
            skipped(r)

    st.divider()
    st.header("Visual model input")
    st.caption(f"Face tensor `{tuple(model_faces.shape)}` — the exact input to the "
               "visual stream (shown de-normalized).")
    disp_model = []
    for f in model_faces:
        hwc = np.transpose(f, (1, 2, 0))
        if do_norm:
            hwc = hwc * C.IMAGENET_STD + C.IMAGENET_MEAN
        disp_model.append(np.clip(hwc * 255, 0, 255).astype(np.uint8))
    for start in range(0, len(disp_model), 8):
        cols = st.columns(8)
        for j, col in enumerate(cols):
            if start + j < len(disp_model):
                col.image(disp_model[start + j], width="stretch")
    st.code(f"faces : shape={tuple(model_faces.shape)} dtype={model_faces.dtype} "
            f"range=[{model_faces.min():.3f}, {model_faces.max():.3f}]", language="text")

    if model_mouth is not None:
        st.divider()
        st.header("Mouth model input")
        st.caption(f"Mouth tensor `{tuple(model_mouth.shape)}` → lip-sync stream "
                   "(AV-HuBERT). Separate from the face tensor above.")
        for start in range(0, len(mouth_crops), 8):
            cols = st.columns(8)
            for j, col in enumerate(cols):
                if start + j < len(mouth_crops):
                    col.image(mouth_crops[start + j], width="stretch")
        st.code(f"mouth : shape={tuple(model_mouth.shape)} dtype={model_mouth.dtype} "
                f"range=[{model_mouth.min():.3f}, {model_mouth.max():.3f}]", language="text")


# =============================== AUDIO ===================================== #
def render_audio():
    prep = _prepared_clip()
    if prep is None:
        return
    row, video_path, n_frames, window_sec, duration, fps, timestamps = prep

    st.header("Audio pipeline")

    raw2d, native_sr = media.decode_audio(str(video_path))
    if raw2d.size == 0:
        st.warning("No audio stream in this clip.")
        return

    wav = raw2d
    sr = native_sr

    with st.container(border=True):
        l, r = st.columns([1, 2])
        l.markdown("**0 · Decode**")
        l.caption(f"Native {native_sr} Hz, {raw2d.shape[0]} channel(s).")
        r.pyplot(waveform_fig(AF.downmix(raw2d), native_sr, "Original (downmixed)"))

    with st.container(border=True):
        l, r = st.columns([1, 2])
        l.markdown("**1 · Mono downmix**")
        do_downmix = l.checkbox("Enable downmix", value=True, key="a_downmix")
        if do_downmix:
            wav = AF.downmix(wav)
            r.pyplot(waveform_fig(wav, sr, "Mono"))
        else:
            wav = raw2d.mean(axis=0)
            skipped(r)
            r.pyplot(waveform_fig(wav, sr, "Channel-averaged for display"))

    with st.container(border=True):
        l, r = st.columns([1, 2])
        l.markdown("**2 · Resample**")
        do_resample = l.checkbox("Enable resample", value=True, key="a_resample")
        target_sr = l.select_slider("target SR", [8000, 16000, 22050, 44100], 16000,
                                    disabled=not do_resample, key="a_sr")
        if do_resample:
            wav = AF.resample(wav, sr, target_sr)
            sr = target_sr
            r.pyplot(waveform_fig(wav, sr, f"Resampled to {sr} Hz"))
        else:
            skipped(r)
            r.pyplot(waveform_fig(wav, sr, f"Native {sr} Hz"))

    with st.container(border=True):
        l, r = st.columns([1, 2])
        l.markdown("**3 · Trim leading silence**")
        do_trim = l.checkbox("Enable trim", key="a_trim")
        top_db = l.slider("top_db", 10.0, 60.0, 30.0, disabled=not do_trim, key="a_topdb")
        if do_trim:
            wav, dropped = AF.trim_leading_silence(wav, sr, top_db)
            l.metric("Dropped", f"{dropped:.3f}s")
            r.pyplot(waveform_fig(wav, sr, "Trimmed"))
        else:
            skipped(r)
            r.pyplot(waveform_fig(wav, sr, "Untrimmed"))

    with st.container(border=True):
        l, r = st.columns([1, 2])
        l.markdown("**4 · Quality & robustness**")
        do_aden = l.checkbox("Noise reduction", key="a_denoise_on")
        aden_str = l.slider("strength", 0.5, 3.0, 1.0, disabled=not do_aden, key="a_den")
        do_rms = l.checkbox("RMS normalize", key="a_rms_on")
        rms_db = l.slider("target dB", -30.0, -6.0, -20.0, disabled=not do_rms, key="a_rms")
        do_band = l.checkbox("Bandpass", key="a_band_on")
        band = l.slider("Hz", 50, 8000, (300, 3000), disabled=not do_band, key="a_band")
        do_addnoise = l.checkbox("Add background noise", key="a_noise_on")
        snr = l.slider("SNR dB", 0.0, 40.0, 20.0, disabled=not do_addnoise, key="a_snr")
        if any([do_aden, do_rms, do_band, do_addnoise]):
            if do_aden:
                wav = AX.spectral_denoise(wav, sr, aden_str)
            if do_rms:
                wav = AX.rms_normalize(wav, rms_db)
            if do_band:
                wav = AX.bandpass(wav, sr, float(band[0]), float(band[1]))
            if do_addnoise:
                wav = AX.add_noise(wav, snr, rng=np.random.default_rng(0))
            r.pyplot(waveform_fig(wav, sr, "After quality steps"))
        else:
            skipped(r)
            r.pyplot(waveform_fig(wav, sr, "Unchanged"))

    with st.container(border=True):
        l, r = st.columns([1, 2])
        l.markdown("**5 · Window extraction**")
        l.caption(f"One {window_sec:.2f}s window centered on each of the {n_frames} frames.")
        windows = AF.extract_windows(wav, sr, timestamps, window_sec)
        if sr != media.AUDIO_SR:
            model_audio = np.stack([AF.resample(w, sr, media.AUDIO_SR) for w in windows])
        else:
            model_audio = windows
        l.metric("Windows", f"{windows.shape[0]} × {windows.shape[1]}")
        r.pyplot(waveform_fig(windows.reshape(-1), sr, "Windows concatenated"))

    with st.container(border=True):
        l, r = st.columns([1, 2])
        l.markdown("**6 · Mel-spectrogram view**")
        do_mel = l.checkbox("Show mel-spectrogram", key="a_mel")
        n_mels = l.slider("n_mels", 32, 128, 64, disabled=not do_mel, key="a_nmels")
        hop = l.slider("hop", 128, 512, 256, step=64, disabled=not do_mel, key="a_hop")
        if do_mel:
            mel = AX.mel_spectrogram(wav, sr, n_mels, hop)
            fig, ax = plt.subplots(figsize=(10, 2.2))
            im = ax.imshow(mel, aspect="auto", origin="lower", cmap="magma")
            fig.colorbar(im, ax=ax, format="%+.0f dB")
            ax.set_title(f"Mel ({n_mels} mels)", fontsize=9)
            r.pyplot(fig)
        else:
            skipped(r)

    st.divider()
    st.header("Audio model input")
    st.caption(f"Audio windows `{tuple(model_audio.shape)}` @ {media.AUDIO_SR} Hz — the "
               "exact input to the cross-modal streams.")
    st.code(f"audio : shape={tuple(model_audio.shape)} dtype={model_audio.dtype} "
            f"range=[{model_audio.min():.3f}, {model_audio.max():.3f}]", language="text")


config_tab, visual_tab, audio_tab = st.tabs(["Config", "Visual", "Audio"])
with config_tab:
    render_config()
with visual_tab:
    render_visual()
with audio_tab:
    render_audio()
