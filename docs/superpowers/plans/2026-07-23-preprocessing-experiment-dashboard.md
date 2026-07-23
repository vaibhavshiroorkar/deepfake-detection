# Preprocessing Experiment Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-file preprocessing viewer with a multi-page Streamlit
dashboard: two Data Preprocessing pages (Visual, Audio) where each preprocessing step is
an independent on/off toggle shown original-vs-processed, plus two read-only Streams
scaffolds.

**Architecture:** An `st.navigation` entry point (`dashboard/app.py`) declares two
sections. Pure, unit-tested step functions live in `dashboard/lib/{visual_ops,audio_ops}.py`;
data selection and media decoding live in `dashboard/lib/{selectors,media}.py`; the page
files in `dashboard/pages/` only compose and render. The old monolith
`dashboard/preprocess_dashboard.py` is split into these units and deleted.

**Tech Stack:** Streamlit 1.60, OpenCV 5, PyAV, librosa, scipy.signal, facenet-pytorch
MTCNN, matplotlib, pandas, numpy. All already installed — no new dependencies.

## Global Constraints

- Streamlit **never trains and never runs a training loop** (PROJECT_OVERVIEW.md §7).
- Dashboard is **read-only**: decodes into memory, **never writes `data/processed/`**.
- Run via the project venv: `uv run streamlit run dashboard/app.py` (streamlit is in
  `.venv`, not system `py -3.13`). Tests run with `uv run --extra cpu python -m pytest`.
- **No new dependency** without updating `pyproject.toml` and re-running `uv sync --extra cpu`.
- Every preprocessing step is **off by default** (baseline = real pipeline behaviour).
- Both preprocessing pages **always show the original beside the processed result**.
- Repo root must be on `sys.path` for `preprocessing.*` imports; page/lib files that may
  be launched by absolute path insert it (see Task 1 pattern).
- Tests live under `tests/dashboard/`. Reuse `preprocessing.crop_faces.crop_and_resize_face`
  — do not reimplement cropping.

---

### Task 1: Data selection — `dashboard/lib/selectors.py`

**Files:**
- Create: `dashboard/lib/__init__.py` (empty)
- Create: `dashboard/lib/selectors.py`
- Create: `tests/dashboard/__init__.py` (empty)
- Create: `tests/dashboard/test_selectors.py`

**Interfaces:**
- Consumes: `data/{train,val,test}.csv`, `data/full_manifest.csv` (columns include
  `clip_id, video_path, label, manipulation_type, method`).
- Produces:
  - `DATASETS: dict[str, str]` — label → filename (`"train" -> "train.csv"`, …,
    `"full_manifest" -> "full_manifest.csv"`).
  - `load_manifest(dataset: str) -> pd.DataFrame`
  - `filter_manifest(df: pd.DataFrame, manip_types: list[str], methods: list[str], label_filter: str) -> pd.DataFrame`
    where `label_filter in {"all","real","fake"}`; empty `manip_types`/`methods` mean no
    filter on that column. Returns a new filtered DataFrame.
  - `render_selection() -> pd.Series | None` — renders the shared Dataset/Target/Clip
    widgets, stores choices in `st.session_state`, returns the selected clip row (or
    `None` if the filter is empty).

- [ ] **Step 1: Write the failing test**

```python
# tests/dashboard/test_selectors.py
import pandas as pd
from dashboard.lib.selectors import filter_manifest

def _df():
    return pd.DataFrame({
        "clip_id": ["a", "b", "c", "d"],
        "label": [0, 1, 1, 1],
        "manipulation_type": ["RealVideo-RealAudio", "FakeVideo-FakeAudio",
                              "FakeVideo-RealAudio", "RealVideo-FakeAudio"],
        "method": ["real", "wav2lip", "faceswap", "real"],
    })

def test_empty_filters_return_everything():
    out = filter_manifest(_df(), [], [], "all")
    assert list(out["clip_id"]) == ["a", "b", "c", "d"]

def test_label_filter_real_only():
    out = filter_manifest(_df(), [], [], "real")
    assert list(out["clip_id"]) == ["a"]

def test_type_and_method_and_together():
    out = filter_manifest(_df(), ["FakeVideo-FakeAudio", "FakeVideo-RealAudio"],
                          ["wav2lip"], "fake")
    assert list(out["clip_id"]) == ["b"]

def test_empty_result_is_empty_frame_not_error():
    out = filter_manifest(_df(), ["FakeVideo-FakeAudio"], ["faceswap"], "all")
    assert len(out) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra cpu python -m pytest tests/dashboard/test_selectors.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.lib.selectors'`

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/lib/selectors.py
"""Shared Dataset/Target/Clip selection for the preprocessing pages.

Pure filtering (filter_manifest) is separated from the Streamlit widgets
(render_selection) so the filter logic is unit-testable without a running app.
All shared st.session_state keys are owned here.
"""
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DATA_DIR = _REPO_ROOT / "data"

# Label -> manifest filename. Deepfake-Eval-2024 is listed but has no manifest
# yet, so render_selection() disables it until data/deepfake_eval.csv exists.
DATASETS = {
    "train": "train.csv",
    "val": "val.csv",
    "test": "test.csv",
    "full_manifest": "full_manifest.csv",
}
_FUTURE_DATASETS = {"deepfake_eval": "deepfake_eval.csv"}

MANIP_TYPES = ["RealVideo-RealAudio", "RealVideo-FakeAudio",
               "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]


def load_manifest(dataset: str) -> pd.DataFrame:
    fname = DATASETS.get(dataset) or _FUTURE_DATASETS.get(dataset)
    if fname is None:
        raise KeyError(f"Unknown dataset {dataset!r}")
    path = DATA_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run the preprocessing pipeline first.")
    return pd.read_csv(path)


def filter_manifest(df: pd.DataFrame, manip_types: list[str],
                    methods: list[str], label_filter: str) -> pd.DataFrame:
    out = df
    if manip_types:
        out = out[out["manipulation_type"].isin(manip_types)]
    if methods:
        out = out[out["method"].isin(methods)]
    if label_filter == "real":
        out = out[out["label"] == 0]
    elif label_filter == "fake":
        out = out[out["label"] == 1]
    return out.reset_index(drop=True)


def render_selection():
    """Render shared Dataset/Target/Clip controls; return the selected clip row."""
    import streamlit as st

    st.subheader("Selection")
    c1, c2 = st.columns(2)
    with c1:
        options = list(DATASETS.keys()) + list(_FUTURE_DATASETS.keys())
        def _fmt(name):
            missing = name in _FUTURE_DATASETS
            return f"{name} (not available)" if missing else name
        dataset = st.selectbox("Dataset", options, format_func=_fmt, key="sel_dataset")
        if dataset in _FUTURE_DATASETS:
            st.info(f"'{dataset}' has no manifest yet. Pick a built split.")
            return None
    with c2:
        sample_n = st.slider("Sample size", 5, 40, 15, key="sel_sample_n")
        seed = st.number_input("Seed", value=42, step=1, key="sel_seed")

    df = load_manifest(dataset)
    methods_present = sorted(df["method"].dropna().unique().tolist())
    t1, t2, t3 = st.columns(3)
    with t1:
        manip = st.multiselect("Target: manipulation type", MANIP_TYPES, key="sel_types")
    with t2:
        methods = st.multiselect("Target: method", methods_present, key="sel_methods")
    with t3:
        label_filter = st.radio("Target: label", ["all", "real", "fake"], key="sel_label")

    filtered = filter_manifest(df, manip, methods, label_filter)
    if len(filtered) == 0:
        st.warning("No clips match the current target filters.")
        return None

    sample = filtered.sample(n=min(int(sample_n), len(filtered)),
                             random_state=int(seed)).reset_index(drop=True)

    def _label(i):
        r = sample.iloc[i]
        tag = "REAL" if r["label"] == 0 else "fake"
        return f"[{tag}] {r['manipulation_type']} — {r['clip_id']}"

    idx = st.selectbox("Clip", range(len(sample)), format_func=_label, key="sel_clip_idx")
    return sample.iloc[idx]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra cpu python -m pytest tests/dashboard/test_selectors.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/lib/__init__.py dashboard/lib/selectors.py tests/dashboard/__init__.py tests/dashboard/test_selectors.py
git commit -m "feat(dashboard): shared dataset/target/clip selector with tested filter"
```

---

### Task 2: Media decoding — `dashboard/lib/media.py`

**Files:**
- Create: `dashboard/lib/media.py`
- Test: `tests/dashboard/test_media.py`

**Interfaces:**
- Consumes: repo-root `sys.path` pattern; `preprocessing.crop_faces.crop_and_resize_face`.
- Produces:
  - `AUDIO_SR = 16000`
  - `sample_timestamps(duration_sec: float, num_frames: int, window_sec: float) -> np.ndarray`
  - `get_detector() -> tuple[MTCNN, str]` (Streamlit-cached; returns detector + device)
  - `decode_frames(video_path: str, timestamps: np.ndarray) -> list[np.ndarray]` — full RGB
    frames (uint8 HxWx3) at each timestamp.
  - `decode_audio(video_path: str) -> tuple[np.ndarray, int]` — `(waveform, native_sr)`,
    waveform float32 shape `[channels, n]` (2-D even if mono) so downmix stays a toggle.
  - `detect_and_crop(frame_rgb, detector, conf_thresh: float, margin: float) -> tuple[np.ndarray, bool]`
    — returns `(rgb_crop_224, detected)`; on miss returns `(full-frame resize, False)`.

- [ ] **Step 1: Write the failing test** (only the pure helper is unit-tested; decoding is
  covered by page AppTests against real clips)

```python
# tests/dashboard/test_media.py
import numpy as np
from dashboard.lib.media import sample_timestamps

def test_sample_timestamps_count_and_inset():
    ts = sample_timestamps(duration_sec=8.0, num_frames=16, window_sec=0.35)
    assert ts.shape == (16,)
    assert ts[0] >= 0.35 / 2 - 1e-9          # inset by half a window
    assert ts[-1] <= 8.0 - 0.35 / 2 + 1e-9
    assert np.all(np.diff(ts) > 0)           # strictly increasing

def test_sample_timestamps_degenerate_short_clip():
    ts = sample_timestamps(duration_sec=0.1, num_frames=4, window_sec=0.35)
    assert ts.shape == (4,)
    assert np.all(ts >= 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra cpu python -m pytest tests/dashboard/test_media.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.lib.media'`

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/lib/media.py
"""Media decoding for the dashboard: frames at timestamps, audio, face detection.

Mirrors preprocessing/extract_clip.py's decode logic (av for audio, OpenCV for
frames) but returns in-memory arrays and NEVER writes data/processed/.
"""
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import av

from preprocessing.crop_faces import crop_and_resize_face

AUDIO_SR = 16000
FRAME_SIZE = 224


def sample_timestamps(duration_sec: float, num_frames: int, window_sec: float) -> np.ndarray:
    margin = window_sec / 2
    return np.linspace(margin, max(duration_sec - margin, margin), num_frames)


def get_detector():
    import streamlit as st

    @st.cache_resource(show_spinner="Loading MTCNN face detector...")
    def _load():
        import torch
        from facenet_pytorch import MTCNN
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return MTCNN(keep_all=False, device=device), device

    return _load()


def decode_frames(video_path: str, timestamps: np.ndarray) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for t in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
        ok, bgr = cap.read()
        if not ok:
            frames.append(np.zeros((FRAME_SIZE, FRAME_SIZE, 3), np.uint8))
            continue
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def frame_meta(video_path: str) -> tuple[float, float]:
    """Return (duration_sec, fps)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = total / fps if fps > 0 else 0.0
    return duration, fps


def decode_audio(video_path: str) -> tuple[np.ndarray, int]:
    container = av.open(str(video_path))
    if not container.streams.audio:
        container.close()
        return np.zeros((1, 0), np.float32), AUDIO_SR
    stream = container.streams.audio[0]
    native_sr = stream.rate
    chunks = []
    for frame in container.decode(stream):
        arr = frame.to_ndarray()            # [channels, samples] or [samples]
        if arr.ndim == 1:
            arr = arr[None, :]
        chunks.append(arr.astype(np.float32))
    container.close()
    if not chunks:
        return np.zeros((1, 0), np.float32), native_sr
    wav = np.concatenate(chunks, axis=1)
    if np.issubdtype(wav.dtype, np.integer):
        wav = wav / np.iinfo(wav.dtype).max
    return wav.astype(np.float32), int(native_sr)


def detect_and_crop(frame_rgb, detector, conf_thresh: float, margin: float):
    box, prob = detector.detect(frame_rgb)
    if box is not None and prob is not None and prob[0] is not None and prob[0] >= conf_thresh:
        crop = crop_and_resize_face(frame_rgb, box[0], (FRAME_SIZE, FRAME_SIZE),
                                    margin_percentage=margin)
        if crop is not None:
            return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), True  # crop_* returns BGR
    return cv2.resize(frame_rgb, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_CUBIC), False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra cpu python -m pytest tests/dashboard/test_media.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/lib/media.py tests/dashboard/test_media.py
git commit -m "feat(dashboard): media decoding (frames, audio, face detect) helpers"
```

---

### Task 3: Visual step ops — `dashboard/lib/visual_ops.py`

**Files:**
- Create: `dashboard/lib/visual_ops.py`
- Test: `tests/dashboard/test_visual_ops.py`

**Interfaces:**
- Produces (all pure, all take/return `uint8 HxWx3 RGB` unless noted):
  - `sharpen(img, amount: float) -> img`
  - `denoise(img, strength: int) -> img`
  - `clahe(img, clip_limit: float) -> img`
  - `gaussian_blur(img, kernel: int) -> img`
  - `jpeg_recompress(img, quality: int) -> img`
  - `downscale_upscale(img, factor: float) -> img`
  - `mouth_region(face_224, size: int = 96) -> img` — lower-center crop of a 224 face,
    resized to `size×size` (dashboard approximation of the landmark mouth crop).
  - `imagenet_normalize(img_uint8) -> np.ndarray` (float32 HxWx3) and
    `normalized_range(arr) -> tuple[float,float]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/dashboard/test_visual_ops.py
import numpy as np
import pytest
from dashboard.lib import visual_ops as V

@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)

@pytest.mark.parametrize("fn,kw", [
    (V.sharpen, {"amount": 1.0}),
    (V.denoise, {"strength": 5}),
    (V.clahe, {"clip_limit": 2.0}),
    (V.gaussian_blur, {"kernel": 5}),
    (V.jpeg_recompress, {"quality": 30}),
])
def test_op_preserves_shape_and_dtype(img, fn, kw):
    out = fn(img, **kw)
    assert out.shape == img.shape
    assert out.dtype == np.uint8

def test_downscale_upscale_returns_same_shape(img):
    out = V.downscale_upscale(img, factor=0.25)
    assert out.shape == img.shape and out.dtype == np.uint8

def test_gaussian_blur_reduces_variance(img):
    assert V.gaussian_blur(img, kernel=9).var() < img.var()

def test_mouth_region_is_96(img):
    out = V.mouth_region(img, size=96)
    assert out.shape == (96, 96, 3)

def test_imagenet_normalize_range(img):
    arr = V.imagenet_normalize(img)
    assert arr.dtype == np.float32
    lo, hi = V.normalized_range(arr)
    assert lo < 0 and hi > 0            # zero-centered, not [0,1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra cpu python -m pytest tests/dashboard/test_visual_ops.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.lib.visual_ops'`

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/lib/visual_ops.py
"""Pure per-step visual preprocessing ops (RGB uint8 in, RGB uint8 out).

No Streamlit, no I/O — unit-testable and reusable. Enhancement steps (sharpen,
denoise, clahe) and degradation/robustness steps (blur, jpeg, downscale) share
this signature so pages can toggle them independently.
"""
import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def sharpen(img, amount: float):
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def denoise(img, strength: int):
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)


def clahe(img, clip_limit: float):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def gaussian_blur(img, kernel: int):
    k = kernel if kernel % 2 == 1 else kernel + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def jpeg_recompress(img, quality: int):
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def downscale_upscale(img, factor: float):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * factor)), max(1, int(h * factor))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def mouth_region(face_224, size: int = 96):
    h, w = face_224.shape[:2]
    crop = face_224[int(h * 0.60):int(h * 0.95), int(w * 0.25):int(w * 0.75)]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)


def imagenet_normalize(img_uint8) -> np.ndarray:
    arr = img_uint8.astype(np.float32) / 255.0
    return (arr - IMAGENET_MEAN) / IMAGENET_STD


def normalized_range(arr) -> tuple[float, float]:
    return float(arr.min()), float(arr.max())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra cpu python -m pytest tests/dashboard/test_visual_ops.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/lib/visual_ops.py tests/dashboard/test_visual_ops.py
git commit -m "feat(dashboard): pure visual preprocessing ops + tests"
```

---

### Task 4: Audio step ops — `dashboard/lib/audio_ops.py`

**Files:**
- Create: `dashboard/lib/audio_ops.py`
- Test: `tests/dashboard/test_audio_ops.py`

**Interfaces:**
- Produces (pure; mono waveform is float32 `[n]` unless noted):
  - `downmix(wav_2d) -> np.ndarray` — mean over channels → `[n]`.
  - `resample(wav, orig_sr: int, target_sr: int) -> np.ndarray`
  - `trim_silence(wav, sr: int, top_db: float) -> tuple[np.ndarray, float]` — `(trimmed, dropped_sec)`.
  - `rms_normalize(wav, target_db: float) -> np.ndarray`
  - `bandpass(wav, sr: int, low_hz: float, high_hz: float) -> np.ndarray`
  - `spectral_denoise(wav, sr: int, strength: float) -> np.ndarray`
  - `add_noise(wav, snr_db: float, rng=None) -> np.ndarray`
  - `extract_windows(wav, sr: int, timestamps, window_sec: float) -> np.ndarray` — `[N, win]`.
  - `mel_spectrogram(wav, sr: int, n_mels: int, hop: int) -> np.ndarray` — dB mel `[n_mels, frames]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/dashboard/test_audio_ops.py
import numpy as np
from dashboard.lib import audio_ops as A

def _tone(sr=16000, secs=1.0, f=440.0):
    t = np.arange(int(sr * secs)) / sr
    return (0.5 * np.sin(2 * np.pi * f * t)).astype(np.float32)

def test_downmix_averages_channels():
    stereo = np.stack([np.ones(10, np.float32), np.full(10, 3.0, np.float32)])
    assert np.allclose(A.downmix(stereo), 2.0)

def test_resample_changes_length_proportionally():
    wav = _tone(sr=16000, secs=1.0)
    out = A.resample(wav, 16000, 8000)
    assert abs(len(out) - 8000) <= 2

def test_trim_silence_drops_leading_zeros():
    wav = np.concatenate([np.zeros(16000, np.float32), _tone(secs=0.5)])
    trimmed, dropped = A.trim_silence(wav, 16000, top_db=30.0)
    assert dropped > 0.8                      # ~1s of leading silence dropped
    assert len(trimmed) < len(wav)

def test_add_noise_hits_target_snr_within_tolerance():
    wav = _tone(secs=1.0)
    noisy = A.add_noise(wav, snr_db=10.0, rng=np.random.default_rng(0))
    noise = noisy - wav
    snr = 10 * np.log10(np.mean(wav ** 2) / np.mean(noise ** 2))
    assert abs(snr - 10.0) < 1.5

def test_extract_windows_shape():
    wav = _tone(secs=2.0)
    ts = np.linspace(0.2, 1.8, 16)
    out = A.extract_windows(wav, 16000, ts, 0.35)
    assert out.shape == (16, int(0.35 * 16000))

def test_mel_spectrogram_rows_equal_n_mels():
    mel = A.mel_spectrogram(_tone(secs=1.0), 16000, n_mels=64, hop=256)
    assert mel.shape[0] == 64

def test_bandpass_runs_and_preserves_length():
    wav = _tone(secs=1.0)
    assert A.bandpass(wav, 16000, 300.0, 3000.0).shape == wav.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra cpu python -m pytest tests/dashboard/test_audio_ops.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.lib.audio_ops'`

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/lib/audio_ops.py
"""Pure per-step audio preprocessing ops. Mono waveform float32 [n] in/out
unless noted. No Streamlit, no I/O."""
import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt


def downmix(wav_2d) -> np.ndarray:
    arr = np.asarray(wav_2d, dtype=np.float32)
    return arr.mean(axis=0) if arr.ndim == 2 else arr


def resample(wav, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or wav.size == 0:
        return wav.astype(np.float32)
    return librosa.resample(wav.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)


def trim_silence(wav, sr: int, top_db: float):
    if wav.size == 0:
        return wav, 0.0
    trimmed, index = librosa.effects.trim(wav, top_db=top_db)
    return trimmed, float(index[0] / sr)


def rms_normalize(wav, target_db: float) -> np.ndarray:
    if wav.size == 0:
        return wav
    rms = np.sqrt(np.mean(wav ** 2)) + 1e-9
    target_rms = 10 ** (target_db / 20.0)
    return np.clip(wav * (target_rms / rms), -1.0, 1.0).astype(np.float32)


def bandpass(wav, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    if wav.size == 0:
        return wav
    nyq = sr / 2.0
    low, high = max(low_hz / nyq, 1e-4), min(high_hz / nyq, 0.999)
    sos = butter(4, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, wav).astype(np.float32)


def spectral_denoise(wav, sr: int, strength: float) -> np.ndarray:
    """Simple spectral gating: attenuate bins below strength*noise-floor."""
    if wav.size == 0:
        return wav
    stft = librosa.stft(wav)
    mag, phase = np.abs(stft), np.angle(stft)
    floor = np.median(mag, axis=1, keepdims=True) * strength
    mag = np.maximum(mag - floor, 0.0)
    out = librosa.istft(mag * np.exp(1j * phase), length=len(wav))
    return out.astype(np.float32)


def add_noise(wav, snr_db: float, rng=None) -> np.ndarray:
    if wav.size == 0:
        return wav
    rng = rng or np.random.default_rng()
    sig_power = np.mean(wav ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=wav.shape).astype(np.float32)
    return (wav + noise).astype(np.float32)


def extract_windows(wav, sr: int, timestamps, window_sec: float) -> np.ndarray:
    win = int(window_sec * sr)
    out = []
    for t in timestamps:
        center = int(t * sr)
        start = max(0, center - win // 2)
        end = start + win
        if end > len(wav):
            end = len(wav)
            start = max(0, end - win)
        w = wav[start:end]
        if len(w) < win:
            w = np.pad(w, (0, win - len(w)))
        out.append(w)
    return np.stack(out).astype(np.float32) if out else np.zeros((0, win), np.float32)


def mel_spectrogram(wav, sr: int, n_mels: int, hop: int) -> np.ndarray:
    if wav.size == 0:
        return np.zeros((n_mels, 0), np.float32)
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=n_mels, hop_length=hop)
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra cpu python -m pytest tests/dashboard/test_audio_ops.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/lib/audio_ops.py tests/dashboard/test_audio_ops.py
git commit -m "feat(dashboard): pure audio preprocessing ops + tests"
```

---

### Task 5: Nav entry point + Streams scaffolds — `dashboard/app.py`, `stream_spec.py`, two stream pages

**Files:**
- Create: `dashboard/lib/stream_spec.py`
- Create: `dashboard/pages/__init__.py` (empty)
- Create: `dashboard/pages/stream_visual.py`
- Create: `dashboard/pages/stream_audiovisual.py`
- Create: `dashboard/app.py`
- Test: `tests/dashboard/test_app_smoke.py`

**Interfaces:**
- Consumes: nothing from prior tasks (scaffolds are static text).
- Produces: `stream_spec.VISUAL_STREAM: dict`, `stream_spec.AUDIOVISUAL_STREAM: dict`
  each with keys `title, status, architecture` (list[str]); `dashboard/app.py` is the
  `streamlit run` entry point declaring the two sections.

- [ ] **Step 1: Write the failing test**

```python
# tests/dashboard/test_app_smoke.py
from streamlit.testing.v1 import AppTest

def test_stream_visual_scaffold_runs():
    at = AppTest.from_file("dashboard/pages/stream_visual.py", default_timeout=60).run()
    assert not at.exception
    assert any("not trained" in m.value.lower() for m in at.markdown + at.info)

def test_stream_audiovisual_scaffold_runs():
    at = AppTest.from_file("dashboard/pages/stream_audiovisual.py", default_timeout=60).run()
    assert not at.exception
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra cpu python -m pytest tests/dashboard/test_app_smoke.py -v`
Expected: FAIL — file `dashboard/pages/stream_visual.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/lib/stream_spec.py
"""Static architecture text for the read-only Streams scaffolds (no compute)."""

VISUAL_STREAM = {
    "title": "Visual stream",
    "status": "Not trained yet — this page fills in after Stage 2/3 training.",
    "architecture": [
        "Backbones: Xception, EfficientNet, DINOv2 (config-driven, one at a time)",
        "Temporal model: BiLSTM over per-frame embeddings",
        "Output: one clip-level embedding (projected to common_dim=256 for fusion)",
        "Never sees audio — labels are the video track's authenticity.",
    ],
}

AUDIOVISUAL_STREAM = {
    "title": "Audiovisual stream",
    "status": "Not trained yet — this page fills in after Stage 4/5 training.",
    "architecture": [
        "Lip-sync: AV-HuBERT (video) + Whisper (audio), scaled dot-product cross-attention",
        "Emotion: HSEmotions (video) + Wav2Vec2 (audio), cross-attention",
        "Each outputs a fixed-size mismatch feature vector (not a standalone score)",
        "Cross-modal by construction — catches audio/video disagreement.",
    ],
}
```

```python
# dashboard/pages/stream_visual.py
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st
from dashboard.lib.stream_spec import VISUAL_STREAM as S

st.title(S["title"])
st.info(S["status"])
st.subheader("Planned architecture")
for line in S["architecture"]:
    st.markdown(f"- {line}")
st.divider()
st.caption("W&B run metrics (loss, val accuracy, per-category accuracy) appear here "
           "after training. Read-only — this page never runs a model (PROJECT_OVERVIEW §7).")
```

```python
# dashboard/pages/stream_audiovisual.py
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st
from dashboard.lib.stream_spec import AUDIOVISUAL_STREAM as S

st.title(S["title"])
st.info(S["status"])
st.subheader("Planned architecture")
for line in S["architecture"]:
    st.markdown(f"- {line}")
st.divider()
st.caption("W&B run metrics appear here after training. Read-only — this page never "
           "runs a model (PROJECT_OVERVIEW §7).")
```

```python
# dashboard/app.py
"""Multi-page preprocessing experiment dashboard (PROJECT_OVERVIEW.md §7).

Run: uv run streamlit run dashboard/app.py
Never trains; never writes data/processed/. See
docs/superpowers/specs/2026-07-23-preprocessing-experiment-dashboard-design.md.
"""
import sys
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

st.set_page_config(page_title="Preprocessing dashboard", layout="wide")

nav = st.navigation({
    "Data Preprocessing": [
        st.Page("pages/preprocess_visual.py", title="Visual"),
        st.Page("pages/preprocess_audio.py", title="Audio"),
    ],
    "Streams": [
        st.Page("pages/stream_visual.py", title="Visual stream"),
        st.Page("pages/stream_audiovisual.py", title="Audiovisual stream"),
    ],
})
nav.run()
```

Note: `app.py` references the two preprocess pages built in Tasks 6–7. Create empty
stub files so `app.py` imports cleanly now; they are filled next:

```python
# dashboard/pages/preprocess_visual.py  (temporary stub, replaced in Task 6)
import streamlit as st
st.title("Visual — preprocessing")
st.info("Under construction.")
```

```python
# dashboard/pages/preprocess_audio.py  (temporary stub, replaced in Task 7)
import streamlit as st
st.title("Audio — preprocessing")
st.info("Under construction.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra cpu python -m pytest tests/dashboard/test_app_smoke.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/app.py dashboard/lib/stream_spec.py dashboard/pages/ tests/dashboard/test_app_smoke.py
git commit -m "feat(dashboard): st.navigation entry point + read-only stream scaffolds"
```

---

### Task 6: Visual preprocessing page — `dashboard/pages/preprocess_visual.py`

**Files:**
- Modify (replace stub): `dashboard/pages/preprocess_visual.py`
- Test: `tests/dashboard/test_preprocess_visual_smoke.py`

**Interfaces:**
- Consumes: `selectors.render_selection`, `media.{frame_meta,sample_timestamps,decode_frames,get_detector,detect_and_crop}`, `visual_ops.*`.
- Produces: a page (no importable API).

- [ ] **Step 1: Write the failing test**

```python
# tests/dashboard/test_preprocess_visual_smoke.py
from streamlit.testing.v1 import AppTest

def test_visual_page_runs_without_exception():
    at = AppTest.from_file("dashboard/pages/preprocess_visual.py", default_timeout=180).run()
    assert not at.exception
    # original + processed grids both rendered
    assert any("Original" in m.value for m in at.subheader)
    assert any("Processed" in m.value for m in at.subheader)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra cpu python -m pytest tests/dashboard/test_preprocess_visual_smoke.py -v`
Expected: FAIL — stub has no "Original"/"Processed" subheaders.

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/pages/preprocess_visual.py
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
    do_sharpen = st.checkbox("Sharpen"); sharpen_amt = st.slider("  amount", 0.0, 3.0, 1.0, disabled=not do_sharpen)
    do_denoise = st.checkbox("Denoise"); denoise_str = st.slider("  strength", 1, 20, 5, disabled=not do_denoise)
    do_clahe = st.checkbox("CLAHE contrast"); clahe_clip = st.slider("  clip", 1.0, 8.0, 2.0, disabled=not do_clahe)
    do_blur = st.checkbox("Gaussian blur"); blur_k = st.slider("  kernel", 3, 31, 9, step=2, disabled=not do_blur)
    do_jpeg = st.checkbox("JPEG re-compress"); jpeg_q = st.slider("  quality", 5, 95, 30, disabled=not do_jpeg)
    do_ds = st.checkbox("Downscale→upscale"); ds_factor = st.slider("  scale", 0.1, 0.9, 0.25, disabled=not do_ds)

duration, fps = media.frame_meta(str(video_path))
window_sec = 0.35
ts = media.sample_timestamps(duration, n_frames, window_sec)
frames = media.decode_frames(str(video_path), ts)
detector, device = media.get_detector()

def process(frame_rgb):
    if do_detect:
        img, detected = media.detect_and_crop(frame_rgb, detector, conf, margin)
    else:
        import cv2
        img, detected = cv2.resize(frame_rgb, (224, 224)), False
    if do_sharpen: img = V.sharpen(img, sharpen_amt)
    if do_denoise: img = V.denoise(img, denoise_str)
    if do_clahe:   img = V.clahe(img, clahe_clip)
    if do_blur:    img = V.gaussian_blur(img, blur_k)
    if do_jpeg:    img = V.jpeg_recompress(img, jpeg_q)
    if do_ds:      img = V.downscale_upscale(img, ds_factor)
    if do_mouth:   img = V.mouth_region(img, 96)
    return img, detected

originals, processed, flags = [], [], []
for f in frames:
    import cv2
    originals.append(cv2.resize(f, (224, 224)))
    p, det = process(f)
    processed.append(p); flags.append(det)

n_det = sum(flags)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Duration", f"{duration:.2f}s"); c2.metric("Source FPS", f"{fps:.1f}")
c3.metric("Faces detected", f"{n_det}/{n_frames}"); c4.metric("Detector device", device)

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra cpu python -m pytest tests/dashboard/test_preprocess_visual_smoke.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/pages/preprocess_visual.py tests/dashboard/test_preprocess_visual_smoke.py
git commit -m "feat(dashboard): visual preprocessing page with toggleable steps + original/processed"
```

---

### Task 7: Audio preprocessing page — `dashboard/pages/preprocess_audio.py`

**Files:**
- Modify (replace stub): `dashboard/pages/preprocess_audio.py`
- Test: `tests/dashboard/test_preprocess_audio_smoke.py`

**Interfaces:**
- Consumes: `selectors.render_selection`, `media.{frame_meta,sample_timestamps,decode_audio,AUDIO_SR}`, `audio_ops.*`, matplotlib.
- Produces: a page (no importable API).

- [ ] **Step 1: Write the failing test**

```python
# tests/dashboard/test_preprocess_audio_smoke.py
from streamlit.testing.v1 import AppTest

def test_audio_page_runs_without_exception():
    at = AppTest.from_file("dashboard/pages/preprocess_audio.py", default_timeout=180).run()
    assert not at.exception
    assert any("Original" in m.value for m in at.subheader)
    assert any("Processed" in m.value for m in at.subheader)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra cpu python -m pytest tests/dashboard/test_preprocess_audio_smoke.py -v`
Expected: FAIL — stub has no "Original"/"Processed" subheaders.

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/pages/preprocess_audio.py
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
    target_sr = st.select_slider("  target SR", [8000, 16000, 22050, 44100], 16000, disabled=not do_resample)
    window_sec = st.slider("Window (s)", 0.10, 1.00, 0.35, 0.05)
with st.sidebar.expander("Representation"):
    do_trim = st.checkbox("Trim leading silence")
    top_db = st.slider("  top_db", 10.0, 60.0, 30.0, disabled=not do_trim)
    do_mel = st.checkbox("Mel-spectrogram view")
    n_mels = st.slider("  n_mels", 32, 128, 64, disabled=not do_mel)
    hop = st.slider("  hop", 128, 512, 256, step=64, disabled=not do_mel)
with st.sidebar.expander("Quality & robustness"):
    do_denoise = st.checkbox("Noise reduction"); denoise_str = st.slider("  strength", 0.5, 3.0, 1.0, disabled=not do_denoise)
    do_rms = st.checkbox("RMS normalize"); rms_db = st.slider("  target dB", -30.0, -6.0, -20.0, disabled=not do_rms)
    do_band = st.checkbox("Bandpass"); band = st.slider("  Hz", 50, 8000, (300, 3000), disabled=not do_band)
    do_addnoise = st.checkbox("Add background noise"); snr = st.slider("  SNR dB", 0.0, 40.0, 20.0, disabled=not do_addnoise)

raw2d, native_sr = media.decode_audio(str(video_path))
# Baseline original for comparison: downmix + resample to AUDIO_SR (what the pipeline stores).
orig = A.resample(A.downmix(raw2d), native_sr, media.AUDIO_SR)

wav = A.downmix(raw2d) if do_downmix else raw2d.mean(axis=0)
sr = native_sr
if do_resample:
    wav = A.resample(wav, sr, target_sr); sr = target_sr
dropped = 0.0
if do_trim:
    wav, dropped = A.trim_silence(wav, sr, top_db)
    st.info(f"Leading silence trimmed: {dropped:.3f}s dropped.")
if do_denoise: wav = A.spectral_denoise(wav, sr, denoise_str)
if do_rms:     wav = A.rms_normalize(wav, rms_db)
if do_band:    wav = A.bandpass(wav, sr, float(band[0]), float(band[1]))
if do_addnoise: wav = A.add_noise(wav, snr, rng=np.random.default_rng(0))

duration, _ = media.frame_meta(str(video_path))
ts = media.sample_timestamps(duration, 16, window_sec)

def waveform_fig(y, rate, title, shift=0.0):
    fig, ax = plt.subplots(figsize=(11, 2.2))
    if y.size:
        ax.plot(np.arange(y.size) / rate, y, linewidth=0.5, color="#3b82f6")
        win = int(window_sec * rate)
        for t in ts:
            c = int((t - shift) * rate); s = max(0, c - win // 2); e = min(y.size, s + win)
            ax.axvspan(s / rate, e / rate, color="#f59e0b", alpha=0.15)
    ax.set_title(title); ax.set_xlabel("s"); ax.margins(x=0)
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
    ax.set_title(f"Mel-spectrogram ({n_mels} mels)"); fig.colorbar(im, ax=ax, format="%+.0f dB")
    st.pyplot(fig)

st.divider()
st.code({
    "downmix": do_downmix, "resample": do_resample and target_sr, "window_s": window_sec,
    "trim": do_trim and top_db, "mel": do_mel and (n_mels, hop),
    "denoise": do_denoise and denoise_str, "rms": do_rms and rms_db,
    "bandpass": do_band and band, "add_noise": do_addnoise and snr,
}, language="python")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra cpu python -m pytest tests/dashboard/test_preprocess_audio_smoke.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/pages/preprocess_audio.py tests/dashboard/test_preprocess_audio_smoke.py
git commit -m "feat(dashboard): audio preprocessing page with toggleable steps + original/processed"
```

---

### Task 8: Remove the monolith, full-suite verification, docs update

**Files:**
- Delete: `dashboard/preprocess_dashboard.py`
- Modify: `docs/PROJECT_OVERVIEW.md` (§14 reconciliation 4 — point run command at `app.py`)

- [ ] **Step 1: Delete the old single-file dashboard**

```bash
git rm dashboard/preprocess_dashboard.py
```

- [ ] **Step 2: Confirm nothing imports the deleted module**

Run: `uv run --extra cpu python -c "import ast,glob; [ast.parse(open(f).read()) for f in glob.glob('dashboard/**/*.py', recursive=True)]; print('parse OK')"`
Expected: `parse OK` and (separately) `grep -rn preprocess_dashboard dashboard docs` returns
only the design-doc "supersedes" mention — no code reference.

- [ ] **Step 3: Update the doc's run command**

In `docs/PROJECT_OVERVIEW.md` §14 reconciliation 4, change the run command line to:

```
   Run: `uv run streamlit run dashboard/app.py`. Pages: Data Preprocessing
   (Visual, Audio) with per-step on/off toggles shown original-vs-processed, and
   read-only Streams scaffolds.
```

- [ ] **Step 4: Run the full test suite**

Run: `uv run --extra cpu python -m pytest -q`
Expected: PASS — all dashboard tests + the existing `tests/test_manifest.py` (24+ passed,
0 failed).

- [ ] **Step 5: Full-app boot smoke (all four pages navigable)**

Run:
```bash
uv run --extra cpu python -c "
from streamlit.testing.v1 import AppTest
for p in ['dashboard/pages/preprocess_visual.py','dashboard/pages/preprocess_audio.py','dashboard/pages/stream_visual.py','dashboard/pages/stream_audiovisual.py']:
    at = AppTest.from_file(p, default_timeout=180).run()
    assert not at.exception, (p, at.exception)
    print('OK', p)
"
```
Expected: `OK` printed for all four pages.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(dashboard): remove monolith, point docs at multi-page app"
```

---

## Self-Review

**Spec coverage:**
- Nav two sections, preprocessing first → Task 5 (`app.py`). ✓
- Shared Dataset/Target/Clip selector persisted → Task 1 (`selectors.render_selection`). ✓
- Original beside processed, both pages → Tasks 6, 7 (Original/Processed subheaders). ✓
- Visual Core/Representation/Quality steps, off by default → Task 6. ✓
- Audio Core/Representation/Quality steps, mel below processed waveform → Task 7. ✓
- Streams read-only scaffolds, no compute → Task 5. ✓
- File layout (`app.py`, `lib/`, `pages/`) → Tasks 1–7. ✓
- Pure ops unit-tested; pages AppTest-smoked → Tasks 3,4 (unit) + 5,6,7 (smoke). ✓
- Never writes `data/processed/`; §7 no-train → enforced by design (no write calls, no
  model). ✓
- No new deps → confirmed installed (scipy/cv2/librosa/streamlit). ✓
- Monolith split then deleted → Task 8. ✓

**Placeholder scan:** no TBD/TODO; all code blocks complete; the Task 5 "stub" files are
explicitly temporary and replaced in Tasks 6–7 with full code shown. ✓

**Type consistency:** `render_selection() -> row` used in Tasks 6/7; `decode_audio ->
(2d, sr)` consumed via `downmix`; `sample_timestamps(duration,n,window)` signature
consistent across media + pages; `detect_and_crop(...) -> (crop, bool)` consumed in
Task 6. ✓
