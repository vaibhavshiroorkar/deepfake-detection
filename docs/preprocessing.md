# Preprocessing — main steps, extras, and the module map

This is the step-by-step reference for how a raw clip becomes model input. It
expands on the *contract* in [PROJECT_OVERVIEW.md §6](PROJECT_OVERVIEW.md#6-preprocessing--the-contract);
§6 stays authoritative for what the pipeline must **produce**, this file
documents how each step **works** and why.

Two layers, kept deliberately separate:

- **Main steps** — the real, state-of-the-art pipeline that produces the stored
  tensors every stream consumes. Each step is an individual, pure function.
- **Extras** — enhancement/degradation ops for robustness experiments, **off by
  default**. Baseline (all extras off) == the real pipeline.

Every step lives once, in `preprocessing/ops/`, and is imported by both the real
batch pipeline (`preprocessing/extract_clip.py`) and the inspection dashboard
(`dashboard/`). There is no second implementation.

---

## Module map (`preprocessing/ops/`)

| Module | Contents |
|---|---|
| `constants.py` | The single home for `NUM_FRAMES=16`, `FRAME_SIZE=224`, `MOUTH_SIZE=96`, `AUDIO_SR=16000`, `AUDIO_WINDOW_SEC=0.35`, `PIPELINE_VERSION`, `IMAGENET_MEAN/STD`, and the 5-point `ARCFACE_TEMPLATE_112`. |
| `faces.py` | **Main visual**: `detect`, `crop_and_resize`, `mouth_roi`, `imagenet_normalize`, and the composed `detect_align_crop`. |
| `audio.py` | **Main audio**: `sample_timestamps`, `decode`, `downmix`, `resample`, `leading_silence_sec` / `trim_leading_silence`, `extract_windows`. |
| `extras_visual.py` | sharpen, denoise, clahe, gaussian_blur, jpeg_recompress, downscale_upscale. |
| `extras_audio.py` | spectral_denoise, rms_normalize, bandpass, add_noise, mel_spectrogram. |

---

## Main visual pipeline (face path)

Applied per sampled frame; the result is stacked into `[16, 3, 224, 224]`.

1. **Frame sampling** — `audio.sample_timestamps`. 16 evenly-spaced timestamps,
   inset by half an audio window so every frame's ±0.175 s audio window stays in
   the clip. The start is offset past leading silence (see the audio section).

2. **Face detection** — `faces.detect`. MTCNN (`facenet-pytorch`, `keep_all=False`)
   returns the most-confident face's box, 5 landmarks, and probability; a face
   must clear the confidence threshold (default 0.90). MTCNN is the only detector
   the pinned environment ships, so it is used throughout.

3. **Crop + resize** — `faces.crop_and_resize`. A margin-padded bbox crop
   (`margin=0.20`) resized to 224×224, cubic. The margin clamps to the frame, so
   the crop can never introduce padding of its own: every pixel came from the
   source. Everything stays RGB end-to-end (no BGR round-trips). Framing follows
   the detector, so head roll and off-centre framing survive into the tensor.

   > **5-point alignment was removed from this step.** It warped the detected
   > landmarks onto a canonical ArcFace template, pose-normalizing the face. It is
   > parked in [ideas.md](ideas.md) together with the two traps it carried (border
   > padding had to be black, not reflected; and `align_inset` had to stay at 0),
   > so re-adding it does not start from scratch. Removal bumped
   > `PIPELINE_VERSION` to 4.

4. **ImageNet normalization** — `faces.imagenet_normalize`. `(x/255 − mean)/std`
   with the ImageNet stats the timm backbones were pretrained on. Applied once,
   here, for all three visual streams.

`faces.detect_crop` composes the detect/crop/mouth steps in **one**
detect call and is the single code path shared by the pipeline and dashboard.

### Mouth ROI (parallel output, for lip-sync — Stage 4)

`faces.mouth_roi` returns a 96×96 crop centered on the two mouth-corner
landmarks. It is a **parallel** output — it does not replace the face crop; the
visual and emotion streams keep the face. It is the single mouth implementation
(the old fixed-fraction `mouth_region` was removed).

> Note: AV-HuBERT's own preprocessing expects a grayscale, 68-landmark mean-face
> aligned mouth ROI. Matching that exactly needs a 68-landmark model (a new
> dependency) and is deferred to Stage 4; today's landmark-centered RGB crop is
> the best available with MTCNN's 5 points.

---

## Main audio pipeline

Produces `[16, window_samples]` (5600 samples at 16 kHz), one window aligned to
each frame.

1. **Decode** — `audio.decode`. PyAV (bundles ffmpeg — no system ffmpeg needed)
   → `[channels, samples]` float32 at the native sample rate.
2. **Mono downmix** — `audio.downmix`. Channel mean.
3. **Resample to 16 kHz** — `audio.resample` (librosa).
4. **Leading-silence handling** — `audio.leading_silence_sec`. Measures the
   near-silent prefix (librosa energy gate, `top_db=30`). This is FakeAVCeleb's
   known **shortcut bug**: fake-audio clips carry extra silence at t=0 that a
   model can cheat on. Rather than trim the audio (which would desync it from the
   frames), the measured offset is fed into `sample_timestamps` so **both** frame
   and audio sampling start past the silence — the shortcut is removed *and*
   frame↔audio alignment is preserved.
5. **Window extraction** — `audio.extract_windows`. One `window_sec` window
   centered on each frame timestamp, clamped and zero-padded to a fixed length.

Frame↔audio alignment is by **timestamp**, not sample index: frame *i* and its
audio window share the same `t_i`, so any stage can trace a frame to exactly the
samples it's paired with.

---

## Extras (robustness / augmentation — off by default)

These are **not** part of the stored contract. They are toggled independently in
the dashboard to probe how preprocessing choices affect a model. Baseline (all
off) reproduces the real pipeline.

- **Visual enhancement** (`extras_visual`): `sharpen`, `denoise`, `clahe`.
- **Visual degradation** (`extras_visual`): `gaussian_blur`, `jpeg_recompress`,
  `downscale_upscale` — simulate compression/quality loss seen in the wild.
- **Audio** (`extras_audio`): `spectral_denoise`, `rms_normalize`, `bandpass`,
  `add_noise`, and `mel_spectrogram` (a visualization/feature view, not a
  pipeline output).

---

## Output contract (unchanged — see PROJECT_OVERVIEW.md §6)

```
face_crop_sequence : [16, 3, 224, 224] float32, ImageNet-normalized
audio              : [16, 5600]        float32 waveform windows @ 16 kHz
mouth (parallel)   : [96, 96, 3]       per frame, for the lip-sync stream (Stage 4)
label              : scalar int, 1 = fake / 0 = real
```

`extract_clip.py` caches `frames.npy` / `audio.npy` / `timestamps.npy` under
`data/processed/<clip_id>/` plus a `version.txt` stamped with `PIPELINE_VERSION`.
Bumping `PIPELINE_VERSION` (e.g. because the crop changed the pixels) invalidates
old caches so they are transparently re-extracted.

> **Because removing alignment changes the cached pixels, `data/processed/` must
> be re-precached and the visual stream re-validated against the previous AUC-0.994
> bar.** See `preprocessing/precache.py` and stage-2. Current value: **4**
> (v4 = alignment removed, margin-padded bbox crop; v3 = black padding + no
> template inset; v2 = alignment + silence-aware sampling; v1 = plain MTCNN crop).
