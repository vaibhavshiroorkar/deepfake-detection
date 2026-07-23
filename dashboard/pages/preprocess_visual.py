import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import streamlit as st
from dashboard.lib import selectors, media, visual_ops as V

st.title("Visual — preprocessing")
st.caption("Toggle steps to experiment. Baseline (all off) = the real pipeline. "
           "Original is always shown beside the processed result. Never writes data/processed/.")

row = selectors.render_selection()
if row is None:
    st.stop()

DATA_DIR = _REPO_ROOT / "data"
video_path = DATA_DIR / row["video_path"]
if not video_path.exists():
    st.error(f"Video not found: {video_path}")
    st.stop()

st.sidebar.header("Visual steps")
with st.sidebar.expander("Core", expanded=True):
    n_frames = st.slider("Frames (N)", 4, 32, 16)
    do_detect = st.checkbox("Face detection (MTCNN)", value=True)
    conf = st.slider("Confidence threshold", 0.50, 0.99, 0.90, 0.01, disabled=not do_detect)
    margin = st.slider("Crop margin", 0.0, 0.6, 0.20, 0.05, disabled=not do_detect)
with st.sidebar.expander("Representation"):
    do_mouth = st.checkbox("Mouth-region crop (96²)", value=False)
    do_norm = st.checkbox("ImageNet normalize", value=False)
with st.sidebar.expander("Quality & robustness"):
    do_sharpen = st.checkbox("Sharpen")
    sharpen_amt = st.slider("  amount", 0.0, 3.0, 1.0, disabled=not do_sharpen)
    do_denoise = st.checkbox("Denoise")
    denoise_str = st.slider("  strength", 1, 20, 5, disabled=not do_denoise)
    do_clahe = st.checkbox("CLAHE contrast")
    clahe_clip = st.slider("  clip", 1.0, 8.0, 2.0, disabled=not do_clahe)
    do_blur = st.checkbox("Gaussian blur")
    blur_k = st.slider("  kernel", 3, 31, 9, step=2, disabled=not do_blur)
    do_jpeg = st.checkbox("JPEG re-compress")
    jpeg_q = st.slider("  quality", 5, 95, 30, disabled=not do_jpeg)
    do_ds = st.checkbox("Downscale→upscale")
    ds_factor = st.slider("  scale", 0.1, 0.9, 0.25, disabled=not do_ds)

duration, fps = media.frame_meta(str(video_path))
window_sec = 0.35
ts = media.sample_timestamps(duration, n_frames, window_sec)
frames = media.decode_frames(str(video_path), ts)
detector, device = media.get_detector()


def process(frame_rgb):
    if do_detect:
        img, detected = media.detect_and_crop(frame_rgb, detector, conf, margin)
    else:
        img, detected = cv2.resize(frame_rgb, (224, 224)), False
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
    if do_mouth:
        img = V.mouth_region(img, 96)
    return img, detected


originals, processed, flags = [], [], []
for f in frames:
    originals.append(cv2.resize(f, (224, 224)))
    p, det = process(f)
    processed.append(p)
    flags.append(det)

n_det = sum(flags)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Duration", f"{duration:.2f}s")
c2.metric("Source FPS", f"{fps:.1f}")
c3.metric("Faces detected", f"{n_det}/{n_frames}")
c4.metric("Detector device", device)

if do_norm:
    arr = V.imagenet_normalize(processed[0])
    lo, hi = V.normalized_range(arr)
    st.caption(f"ImageNet normalize ON — processed pixel range now [{lo:.2f}, {hi:.2f}] "
               f"(display grids stay uint8).")

cols_per_row = 8
st.subheader("Original frames")
for s in range(0, len(originals), cols_per_row):
    cols = st.columns(cols_per_row)
    for j, col in enumerate(cols):
        if s + j < len(originals):
            col.image(originals[s + j], caption=f"t={ts[s+j]:.2f}s", width="stretch")

st.subheader("Processed frames")
for s in range(0, len(processed), cols_per_row):
    cols = st.columns(cols_per_row)
    for j, col in enumerate(cols):
        k = s + j
        if k < len(processed):
            cap = f"t={ts[k]:.2f}s" + ("" if flags[k] or not do_detect else " ⚠fallback")
            col.image(processed[k], caption=cap, width="stretch")

st.divider()
st.code({
    "frames": n_frames, "detect": do_detect, "conf": conf, "margin": margin,
    "mouth": do_mouth, "imagenet_norm": do_norm,
    "sharpen": do_sharpen and sharpen_amt, "denoise": do_denoise and denoise_str,
    "clahe": do_clahe and clahe_clip, "blur": do_blur and blur_k,
    "jpeg": do_jpeg and jpeg_q, "downscale": do_ds and ds_factor,
}, language="python")
