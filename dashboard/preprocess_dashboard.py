"""
Streamlit preprocessing dashboard (PROJECT_OVERVIEW.md section 7).

A fast, small-sample iteration loop for *preprocessing decisions only*. You
pick one clip from a tiny sample, drag the same knobs the real pipeline uses
(frame count, face-crop margin, MTCNN confidence threshold, audio-window
length, leading-silence trim), and immediately see the resulting face crops and
the aligned audio waveform. When a setting looks right here, you change the
matching constant in preprocessing/ and re-run the batch pipeline for real.

What this dashboard is NOT (also section 7):
  * It never trains anything and never launches a training loop.
  * It never writes to data/processed/ -- it decodes into memory and throws the
    result away, so experimenting here can't corrupt the real cache that
    extract_clip.py builds. It is a viewer, not a producer.

Run it (from the repo root, so `preprocessing` imports resolve):

    uv run streamlit run dashboard/preprocess_dashboard.py
    # or: py -3.13 -m streamlit run dashboard/preprocess_dashboard.py

The knobs mirror these pipeline constants, deliberately, so a value chosen here
maps to exactly one place in the code:
    frames            -> extract_clip.NUM_FRAMES
    audio window (s)  -> extract_clip.AUDIO_WINDOW_SEC
    crop margin       -> crop_faces.crop_and_resize_face(margin_percentage=...)
    confidence thresh -> crop_faces.process_video(confidence_threshold=...)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Repo root on sys.path so `preprocessing` imports work when Streamlit launches
# this file by absolute path (its own dir, not the repo root, is on sys.path).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2  # noqa: E402
import av  # noqa: E402
import librosa  # noqa: E402

from preprocessing.crop_faces import crop_and_resize_face  # noqa: E402

DATA_DIR = _REPO_ROOT / "data"
FULL_MANIFEST = DATA_DIR / "full_manifest.csv"
AUDIO_SR = 16000  # matches extract_clip.AUDIO_SR

st.set_page_config(page_title="Preprocessing dashboard", layout="wide")


# --------------------------------------------------------------------------- #
# Cached, expensive-to-build resources                                         #
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading MTCNN face detector...")
def get_detector():
    """One MTCNN instance, reused across reruns (CPU -- this is a viewer)."""
    import torch  # local import: keeps torch out of the module top
    from facenet_pytorch import MTCNN

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return MTCNN(keep_all=False, device=device), device


@st.cache_data(show_spinner=False)
def load_sample(n: int, seed: int) -> pd.DataFrame:
    """A small, reproducible sample of clips from the full manifest."""
    if not FULL_MANIFEST.exists():
        return pd.DataFrame()
    df = pd.read_csv(FULL_MANIFEST)
    n = min(n, len(df))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def decode_audio(video_path: str) -> np.ndarray:
    """Full mono 16 kHz waveform for a clip. Mirrors extract_clip._decode_audio."""
    container = av.open(video_path)
    if not container.streams.audio:
        container.close()
        return np.zeros(0, dtype=np.float32)
    stream = container.streams.audio[0]
    native_sr = stream.rate
    chunks = []
    for frame in container.decode(stream):
        arr = frame.to_ndarray()
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        chunks.append(arr.astype(np.float32))
    container.close()
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    wav = np.concatenate(chunks)
    if np.issubdtype(wav.dtype, np.integer):
        wav = wav / np.iinfo(wav.dtype).max
    if native_sr != AUDIO_SR:
        wav = librosa.resample(wav, orig_sr=native_sr, target_sr=AUDIO_SR)
    return wav.astype(np.float32)


def sample_timestamps(duration_sec: float, num_frames: int, window_sec: float) -> np.ndarray:
    """Evenly spaced timestamps, inset by half the audio window (as in extract_clip)."""
    margin = window_sec / 2
    return np.linspace(margin, max(duration_sec - margin, margin), num_frames)


def trim_leading_silence(wav: np.ndarray, top_db: float = 30.0) -> tuple[np.ndarray, float]:
    """
    Trim leading silence and report how many seconds were dropped.

    FakeAVCeleb has a known shortcut bug: some fake-audio clips carry extra
    silence at t=0 that a model can cheat on (PROJECT_OVERVIEW.md section 6).
    This lets you SEE how much a clip would shift if that silence were trimmed.
    """
    if wav.size == 0:
        return wav, 0.0
    trimmed, index = librosa.effects.trim(wav, top_db=top_db)
    dropped_sec = index[0] / AUDIO_SR
    return trimmed, dropped_sec


# --------------------------------------------------------------------------- #
# Sidebar controls                                                             #
# --------------------------------------------------------------------------- #
st.sidebar.title("Preprocessing knobs")

if not FULL_MANIFEST.exists():
    st.error(
        f"{FULL_MANIFEST.relative_to(_REPO_ROOT)} not found. Run the audit first:\n\n"
        "`py -3.13 -m preprocessing.audit_dataset`"
    )
    st.stop()

sample_n = st.sidebar.slider("Sample size (clips to choose from)", 5, 30, 15)
sample_seed = st.sidebar.number_input("Sample seed", value=42, step=1)
sample = load_sample(sample_n, int(sample_seed))

# Show a readable label per clip: manipulation type + short id.
def clip_label_text(row) -> str:
    tag = "REAL" if row["label"] == 0 else "fake"
    return f"[{tag}] {row['manipulation_type']} — {row['clip_id']}"

choice = st.sidebar.selectbox(
    "Clip", options=list(range(len(sample))),
    format_func=lambda i: clip_label_text(sample.iloc[i]),
)
row = sample.iloc[choice]

st.sidebar.divider()
num_frames = st.sidebar.slider("Frames sampled (N)", 4, 32, 16,
                               help="extract_clip.NUM_FRAMES — batch time dimension.")
margin = st.sidebar.slider("Face-crop margin", 0.0, 0.6, 0.2, 0.05,
                           help="Padding around the MTCNN box, as a fraction of box size.")
conf_thresh = st.sidebar.slider("Detection confidence threshold", 0.50, 0.99, 0.90, 0.01,
                                help="Below this, the face is treated as not found (full-frame fallback).")
window_sec = st.sidebar.slider("Audio window (seconds)", 0.10, 1.00, 0.35, 0.05,
                               help="extract_clip.AUDIO_WINDOW_SEC — centered on each frame.")
do_trim = st.sidebar.checkbox("Trim leading silence", value=False,
                              help="Preview the FakeAVCeleb leading-silence shortcut fix.")


# --------------------------------------------------------------------------- #
# Main view                                                                    #
# --------------------------------------------------------------------------- #
st.title("Preprocessing dashboard")
st.caption(
    "Small-sample iteration for preprocessing params only. Reads the same crop "
    "logic the batch pipeline uses. It never writes to data/processed/ and never trains."
)

video_path = DATA_DIR / row["video_path"]
st.markdown(f"**Clip:** `{row['clip_id']}` &nbsp; **label:** "
            f"{'real (0)' if row['label'] == 0 else 'fake (1)'} &nbsp; "
            f"**type:** {row['manipulation_type']} &nbsp; **method:** {row['method']}")

if not video_path.exists():
    st.error(f"Video not found on disk: {video_path}")
    st.stop()

# --- Decode frames at the sampled timestamps and run detection live -------- #
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total / fps if fps > 0 else 0.0
timestamps = sample_timestamps(duration, num_frames, window_sec)

detector, device = get_detector()
crops, detected_flags = [], []
for t in timestamps:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame_bgr = cap.read()
    if not ok:
        crops.append(np.zeros((224, 224, 3), np.uint8))
        detected_flags.append(False)
        continue
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    box, prob = detector.detect(frame_rgb)
    if box is not None and prob is not None and prob[0] is not None and prob[0] >= conf_thresh:
        crop = crop_and_resize_face(frame_rgb, box[0], (224, 224), margin_percentage=margin)
        if crop is not None:
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))  # crop_* returns BGR
            detected_flags.append(True)
            continue
    # Fallback: full-frame resize (what the pipeline does when no face passes).
    crops.append(cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_CUBIC))
    detected_flags.append(False)
cap.release()

n_detected = sum(detected_flags)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Duration", f"{duration:.2f}s")
c2.metric("Source FPS", f"{fps:.1f}")
c3.metric("Faces detected", f"{n_detected}/{num_frames}")
c4.metric("Detector device", device)

st.subheader("Face crops")
st.caption("Red-captioned tiles are full-frame fallbacks (no face cleared the threshold).")
cols_per_row = 8
for start in range(0, len(crops), cols_per_row):
    cols = st.columns(cols_per_row)
    for j, col in enumerate(cols):
        k = start + j
        if k >= len(crops):
            break
        cap_txt = f"t={timestamps[k]:.2f}s" + ("" if detected_flags[k] else " ⚠fallback")
        col.image(crops[k], caption=cap_txt, width="stretch")

# --- Audio: full waveform + the windows aligned to each sampled frame ------ #
st.subheader("Audio")
wav = decode_audio(str(video_path))
dropped = 0.0
if do_trim:
    wav, dropped = trim_leading_silence(wav)
    st.info(f"Leading silence trimmed: {dropped:.3f}s dropped from t=0.")

if wav.size == 0:
    st.warning("No audio stream decoded for this clip.")
else:
    import matplotlib.pyplot as plt

    window_samples = int(window_sec * AUDIO_SR)
    t_axis = np.arange(wav.size) / AUDIO_SR
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.plot(t_axis, wav, linewidth=0.5, color="#3b82f6")
    # Shade the audio window paired with each sampled frame.
    for t in timestamps:
        center = int((t - dropped) * AUDIO_SR)
        s = max(0, center - window_samples // 2)
        e = min(wav.size, s + window_samples)
        ax.axvspan(s / AUDIO_SR, e / AUDIO_SR, color="#f59e0b", alpha=0.18)
    ax.set_xlabel("seconds")
    ax.set_ylabel("amplitude")
    ax.set_title(f"Waveform with {num_frames} aligned {window_sec:.2f}s windows (shaded)")
    ax.margins(x=0)
    st.pyplot(fig)
    st.audio(wav, sample_rate=AUDIO_SR)

st.divider()
st.caption(
    "When a setting here looks right, change the matching constant in "
    "`preprocessing/` (NUM_FRAMES, AUDIO_WINDOW_SEC, crop margin, confidence "
    "threshold) and re-run `preprocessing.precache` to rebuild the real cache."
)
