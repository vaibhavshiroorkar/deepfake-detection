# Preprocessing Experiment Dashboard — Design

**Date:** 2026-07-23
**Status:** approved, ready for implementation plan
**Supersedes:** the single-file `dashboard/preprocess_dashboard.py` built earlier
2026-07-23 (its logic is reused, not discarded).

---

## 1. Purpose

Turn the one-page preprocessing viewer into a **multi-page experiment bench**. Two
jobs:

1. **Data Preprocessing** — build up any combination of preprocessing steps by
   toggling each one on/off (with its parameters), on a chosen dataset split and a
   chosen manipulation target, and see the effect immediately against the original.
   The config you land on is the same set of knobs a training run consumes, so
   "experiment the model in different situations" (including robustness to blur, JPEG
   re-compression, added noise) happens by carrying a config from here into a tracked
   run — **the dashboard itself never trains**.
2. **Streams** — read-only scaffolds for the five streams that do not exist yet
   (Stage 2+), showing planned architecture and a placeholder for the W&B metrics that
   will appear once each stream trains.

### Hard constraints (from PROJECT_OVERVIEW.md §7)

- Streamlit **never trains and never runs a training loop**. It reads/experiments; it
  does not produce model results.
- The dashboard is **read-only against the cache**: it decodes clips into memory and
  throws the result away. It **never writes `data/processed/`**, so experimenting here
  cannot corrupt the real cache that `extract_clip.py` builds.
- Runs through the project venv: `uv run streamlit run dashboard/app.py` (streamlit
  lives in `.venv`, not system `py -3.13`).

---

## 2. Navigation

Streamlit 1.60 `st.navigation` with two grouped sections, Data Preprocessing first:

```
Data Preprocessing
   ├─ Visual
   └─ Audio
Streams
   ├─ Visual stream          (read-only scaffold)
   └─ Audiovisual stream      (read-only scaffold)
```

`dashboard/app.py` is the single entry point that declares the pages and sections and
calls `st.navigation(...).run()`.

---

## 3. Shared selection (both preprocessing pages)

Rendered at the top of Visual and Audio, persisted in `st.session_state` so switching
pages keeps the current clip. Three controls:

- **Dataset** — `train / val / test / full_manifest`. Deepfake-Eval-2024 appears as a
  disabled option (greyed) until that manifest exists. Selecting a split reads
  `data/<split>.csv`; `full_manifest` reads `data/full_manifest.csv`.
- **Target** — filters the sample:
  - `manipulation_type` multiselect (RVRA / RVFA / FVRA / FVFA),
  - `method` multiselect (real / wav2lip / faceswap / fsgan — values present in the
    manifest's `method` column),
  - a real/fake radio (all / real only / fake only).
  All filters AND together; an empty multiselect means "no filter on that column".
- **Clip** — a `selectbox` over the filtered sample (labelled `[REAL|fake] <type> —
  <clip_id>`), plus a sample-size and seed control for reproducibility.

If the current filter yields zero clips, the page shows an inline notice and stops
before decoding.

---

## 4. Preprocessing pages — the toggleable step pipeline

Each page presents preprocessing as **independent steps you enable/disable**, grouped
into three collapsible sections. Every step is **off by default**, so the baseline is
the real pipeline's behaviour (face-crop + normalize for visual; downmix + resample +
window for audio). The pipeline order within a page is fixed and sensible — no
reordering UI (YAGNI).

**Every page always shows the ORIGINAL alongside the PROCESSED result**, side by side,
so the effect of the active toggles is always visible against the untouched input.

### 4.1 Visual page

Decodes N frames at evenly-spaced timestamps from the selected clip, then applies the
enabled steps to each frame. Renders two aligned grids: **Original frames** and
**Processed frames** (fallback tiles flagged where no face cleared the threshold).

Step groups:

**Core**
| Step | Off behaviour | Params |
|---|---|---|
| Face detection (MTCNN) | use full frame | confidence threshold |
| Crop margin | tight box | margin % |
| Frame sampling | always N | N frames |
| Resize 224² | — (always on) | interpolation (cubic/linear/area) |

**Representation**
| Step | Off behaviour | Params |
|---|---|---|
| Mouth-region crop (96²) | skip (show face crop) | — (for lip-sync stream later) |
| ImageNet normalize | raw [0,1] | — (reports the pixel-range shift) |

**Quality & robustness** (applied to the crop, before normalize)
| Step | Direction | Params |
|---|---|---|
| Sharpen (unsharp mask) | enhance | amount |
| Denoise (bilateral / NLM) | enhance | strength |
| Contrast/brightness (CLAHE) | enhance | clip limit |
| Gaussian blur | degrade | kernel |
| JPEG re-compression | degrade | quality % |
| Downscale→upscale | degrade | scale factor |

Metrics shown: duration, source FPS, faces-detected/N, detector device.

### 4.2 Audio page

Decodes the clip's full track, applies the enabled steps, renders **Original waveform**
and **Processed waveform** (both with the 16 aligned windows shaded), plus an audio
player for each. When the mel-spectrogram view is enabled, the processed panel renders
the mel-spectrogram **below** the processed waveform (in addition, not replacing it), so
the original-vs-processed waveform comparison is preserved.

Step groups:

**Core**
| Step | Off behaviour | Params |
|---|---|---|
| Mono downmix | keep channels | — |
| Resample 16 kHz | native SR | target SR |
| Window extraction | full track | window seconds |

**Representation**
| Step | Off behaviour | Params |
|---|---|---|
| Leading-silence trim | keep t=0 silence | top_db (+ dropped-seconds readout) |
| Mel-spectrogram view | waveform only | n_mels, hop length |

**Quality & robustness**
| Step | Direction | Params |
|---|---|---|
| Noise reduction (spectral gating) | enhance | strength |
| Loudness / RMS normalize | enhance | target level |
| Bandpass filter | enhance | low/high Hz |
| Add background noise | degrade | SNR dB |

The leading-silence trim is called out because it previews the fix for FakeAVCeleb's
known leading-silence shortcut bug (PROJECT_OVERVIEW.md §6).

### 4.3 Active-config summary

Each page renders a compact, copyable summary of the enabled steps + params (e.g. a
small `st.code` block of the config dict), so a promising configuration can be carried
by hand into a W&B-tracked training run.

---

## 5. Streams pages — read-only scaffolds

Each page shows, with **no compute**:

- **Status:** "Not trained yet — this page fills in after Stage N."
- **Planned architecture**, read from a static spec:
  - *Visual stream:* backbones Xception / EfficientNet / DINOv2, temporal model BiLSTM,
    one clip-level embedding out.
  - *Audiovisual stream:* lip-sync (AV-HuBERT video + Whisper audio, cross-attention),
    emotion (HSEmotions video + Wav2Vec2 audio, cross-attention); each outputs a
    fixed-size mismatch feature vector.
- **Placeholder panel** captioned "W&B run metrics (loss, val accuracy, per-category
  accuracy) appear here after training." A future iteration wires this to
  `wandb.Api()` (PROJECT_OVERVIEW.md §7, optional read-only viewer); this spec only
  builds the placeholder.

---

## 6. File layout

The current `dashboard/preprocess_dashboard.py` monolith is split into focused units,
each with one purpose and a well-defined interface. It is then deleted.

```
dashboard/
  app.py                     # st.navigation entry point; declares pages + sections
  lib/
    selectors.py             # manifest load, dataset/target filtering, clip picker (session_state)
    media.py                 # decode frames at timestamps, decode audio, MTCNN singleton
    visual_ops.py            # pure per-step visual functions (detect_crop, sharpen, denoise,
                             #   clahe, blur, jpeg, downscale, mouth_crop, imagenet_normalize)
    audio_ops.py             # pure per-step audio functions (downmix, resample, trim_silence,
                             #   rms_normalize, bandpass, denoise, add_noise, window, mel)
    stream_spec.py           # static architecture text for the two stream scaffolds
  pages/
    preprocess_visual.py
    preprocess_audio.py
    stream_visual.py
    stream_audiovisual.py
```

**Interface conventions:**

- `visual_ops` / `audio_ops` functions are **pure**: `(array, params) -> array`, no
  Streamlit calls, no I/O. This keeps them unit-testable and reusable by the real
  pipeline later, and keeps the page files thin (compose + render only).
- `media.py` reuses the existing decode logic (`av` for audio, OpenCV for frames) and
  `preprocessing.crop_faces.crop_and_resize_face` — the same crop the batch pipeline
  uses, so a value chosen in the dashboard maps to exactly one pipeline constant.
- `selectors.py` owns all `st.session_state` keys for the shared selection.

---

## 7. Dependencies

- Already present: `streamlit==1.60.0`, `opencv-python`, `av`, `librosa`, `matplotlib`,
  `facenet-pytorch`, `numpy`, `pandas`.
- **Possibly new** (decide during planning; prefer existing deps first):
  - Spectral-gating noise reduction — implement with `librosa`/`numpy` rather than
    adding `noisereduce`, to avoid a new pin.
  - JPEG re-compression, CLAHE, bilateral/NLM denoise, blur — all available in
    `opencv-python`, no new dep.
  - Bandpass filter — `scipy.signal` (scipy is a `librosa`/`scikit-learn` transitive
    dep; confirm it is importable before relying on it, else implement in numpy).

No new dependency should be added without updating `pyproject.toml` pins and re-running
`uv sync --extra cpu`.

---

## 8. Testing

- **Unit tests** (`tests/`) for the pure ops: each `visual_ops` / `audio_ops` function
  gets a shape/dtype/range test and, where meaningful, a behavioural assertion (e.g.
  `trim_silence` on a signal with known leading zeros drops the expected sample count;
  `add_noise` at a target SNR lands within tolerance; `imagenet_normalize` produces the
  expected mean shift).
- **Headless smoke test** for each page via `streamlit.testing.v1.AppTest`: run the page
  with a default clip, assert zero exceptions and that the original+processed panels
  render. This mirrors the smoke test already used for the monolith.
- `selectors.filter` gets a test that target filters AND correctly and that an empty
  result is handled.

---

## 9. Out of scope (YAGNI)

- No step reordering UI (fixed sensible order).
- No live W&B fetch yet (placeholder panel only).
- No model inference on the stream pages (read-only scaffolds).
- No writing to `data/processed/` or any persistence of experiment configs beyond the
  on-screen copyable summary.
- No new dataset beyond FakeAVCeleb (Deepfake-Eval-2024 selector is present but
  disabled until its manifest exists).
