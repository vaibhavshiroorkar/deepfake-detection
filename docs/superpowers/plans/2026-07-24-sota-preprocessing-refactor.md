# Plan / implementation record — SOTA preprocessing refactor (2026-07-24)

Design: [../specs/2026-07-24-sota-preprocessing-refactor-design.md](../specs/2026-07-24-sota-preprocessing-refactor-design.md).
Reference: [../../preprocessing.md](../../preprocessing.md).

## What was built

### New canonical ops layer — `preprocessing/ops/`
- `constants.py` — single home for `NUM_FRAMES/FRAME_SIZE/MOUTH_SIZE/AUDIO_SR/
  AUDIO_WINDOW_SEC`, `PIPELINE_VERSION`, `IMAGENET_MEAN/STD`, `ARCFACE_TEMPLATE_112`.
- `faces.py` — `detect`, `align_face` (5-point similarity warp to the ArcFace
  template — the SOTA upgrade), `crop_and_resize` (RGB fallback), `mouth_roi`,
  `imagenet_normalize`, `normalized_range`, composed `detect_align_crop`.
- `audio.py` — `sample_timestamps(start_offset=…)`, `decode`, `downmix`,
  `resample`, `trim_leading_silence`, `leading_silence_sec`, `extract_windows`.
- `extras_visual.py` — sharpen, denoise, clahe, gaussian_blur, jpeg_recompress,
  downscale_upscale.
- `extras_audio.py` — spectral_denoise, rms_normalize, bandpass, add_noise,
  mel_spectrogram.

### Real pipeline rewired
- `extract_clip.py` — decodes audio first, measures leading silence, offsets
  frame+audio sampling past it, aligns each face (`align=True` default), writes a
  versioned cache (`version.txt`). Returns `leading_silence_sec`.
- `audit_dataset.py` — adds `leading_silence_sec` column + a real-vs-fake-audio
  shortcut summary (`--no-silence` skips the slow pass).
- `dataset.py` — imports constants from `ops.constants` (dropped local ImageNet copy).

### Dashboard de-duplicated
- Deleted `dashboard/lib/visual_ops.py`, `dashboard/lib/audio_ops.py`.
- `media.py` — `detect_and_crop` / `detect_face_and_mouth` now wrap
  `ops.faces.detect_align_crop`; `decode_audio`/`sample_timestamps` re-export ops.
- `inference.py` — ImageNet from `ops.constants`.
- `pages/preprocess.py` — imports the new modules; adds a "5-point align" toggle
  (default on) to the Visual face step.

### Removed (redundant/legacy)
- `preprocessing/crop_faces.py` (standalone CLI) + `ffmpeg-python` (pyproject +
  uv.lock + README), the unused `mouth_region`, the 4 copied ImageNet constants.

### Tests
- New `tests/preprocessing/`: `test_ops_faces.py`, `test_ops_audio.py`,
  `test_extras_visual.py`, `test_extras_audio.py`, `test_extract_clip_contract.py`
  (data-gated, skips without the dataset).
- Removed `tests/dashboard/test_{visual,audio}_ops.py` (coverage moved).

## Verification (run this session)
- `pytest tests/` → **63 passed**.
- Live `extract_clip` on a sample clip → `frames (16,224,224,3) uint8`,
  `audio (16,5600) f32`, 16/16 faces aligned.
- `precache --splits val --limit 5` → versioned cache; `ClipDataset("val")` batch
  → `[2,16,3,224,224]` / `[2,16,5600]`, ImageNet-normalized.

## Follow-up (NOT done here — needs compute)
- Re-precache all splits and re-validate the Stage-2 visual stream against the
  AUC-0.994 bar, since alignment changed the cached pixels.
