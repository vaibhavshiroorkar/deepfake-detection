# Design — SOTA preprocessing refactor (2026-07-24)

## Problem

The preprocessing pipeline worked and was tested, but (a) it was not
state-of-the-art — faces were detected and center-cropped but never
pose-normalized, and PROJECT_OVERVIEW.md §6's mandated FakeAVCeleb
leading-silence QC was missing — and (b) it was implemented twice (once in
`preprocessing/`, once in `dashboard/lib/`), with the "main" pipeline and the
enhancement/degradation "extras" entangled. ImageNet stats were copy-pasted in
four files; window and mouth-crop logic each existed in two variants; a legacy
`crop_faces.py` CLI dragged in `ffmpeg-python`.

## Goal

Make the **main** deepfake preprocessing steps state-of-the-art and expose each
as an individual, reusable, testable function; isolate the **extras**; delete the
redundancy; document it.

## Constraints / decisions (confirmed with the user)

1. **No new dependencies.** MTCNN (the only installed detector) + OpenCV +
   librosa/scipy only. True 68-landmark AV-HuBERT mouth alignment is deferred to
   Stage 4 (needs a new dep).
2. **5-point face alignment default ON** — changes cached pixels ⇒ re-precache +
   re-validate against the AUC-0.994 bar.
3. **Leading silence: measure + aligned offset** — audit records
   `leading_silence_sec`; extraction offsets frame+audio sampling past it so the
   shortcut is removed and frame↔audio alignment is preserved (no audio trim).
4. **Remove the legacy `crop_faces.py` CLI + `ffmpeg-python`**, de-dup the 4
   ImageNet copies, delete the unused `mouth_region`.

## Approach

One canonical, pure ops layer `preprocessing/ops/` (constants, faces, audio,
extras_visual, extras_audio) imported by **both** the real pipeline and the
dashboard. `faces.detect_align_crop` is the single detect→align→crop→mouth path.
`audio.sample_timestamps(start_offset=…)` carries the silence offset so frames and
audio stay aligned. Cache gains a `PIPELINE_VERSION` stamp so stale unaligned
caches re-extract.

Contract preserved exactly: `[16,3,224,224]` ImageNet-normalized faces,
`[16,5600]` 16 kHz audio, 96² mouth, `label`.

Full step reference: [../../preprocessing.md](../../preprocessing.md).

## Alternatives considered

- **RetinaFace / InsightFace / dlib / mediapipe** for detection+alignment —
  rejected (decision 1): new deps with py3.13 wheel risk the repo has avoided.
  MTCNN's 5 landmarks are enough for ArcFace-style similarity alignment.
- **Trimming the audio** for the silence shortcut — rejected: it desyncs audio
  from the frame timeline. Offsetting the shared timestamps keeps alignment.
- **Neural VAD (Silero)** for silence — rejected (decision 1); librosa's energy
  gate is sufficient to measure and offset a leading-silence prefix.
