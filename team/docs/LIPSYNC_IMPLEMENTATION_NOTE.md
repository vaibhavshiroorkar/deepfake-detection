# Lip-sync module: implementation note (WS4)

Written 2026-10-26. Follows docs/LIPSYNC_DECISION_NOTE.md. Status: implemented as a prototype module, not wired into the pipeline, not evaluated. Decided autonomously within the user's authorisation to download the weights for an academic prototype - needs user review (choice of placeholder thresholds, YuNet instead of S3FD).

## Weights

| Item | Value |
|---|---|
| File | `data/models/syncnet/syncnet_v2.model` (git-ignored; record in `data/models/syncnet/SOURCE.json`) |
| Source used | Official: `http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model` (HTTP 200). The Hugging Face fallback was not needed |
| Date | 2026-10-26 |
| Size | 54,573,114 bytes |
| SHA-256 | `961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442` |
| Format | PyTorch state dict; loads with `torch.load(weights_only=True)` and strict key matching into the ported `S` class (13,631,616 parameters) |
| Code licence | MIT, `joonson/syncnet_python` LICENSE.md, "Copyright (c) 2016-present Joon Son Chung" (notice reproduced in `src/dfd/platform/syncnet/model.py`) |
| Weights licence | **Not stated by the source.** No licence appears on the download location, in the syncnet_python README or in LICENSE.md for this file. No permission is inferred. The user authorised the download for an academic prototype; the weights are treated as research-only and must not be redistributed or used commercially without clearing this with the owners. |

Nothing else was downloaded (no S3FD detector). Faces come from the project's YuNet detector.

## Ported code and constants

Reference: github.com/joonson/syncnet_python at master, fetched 2026-10-26 (SyncNetModel.py, SyncNetInstance.py, run_pipeline.py, run_syncnet.py, README.md). SI = SyncNetInstance.py, RP = run_pipeline.py.

| Constant | Value | Source |
|---|---|---|
| Video frame rate | 25 fps | SI line 2; RP line 31; RP line 267 (`ffmpeg -r 25`) |
| Audio | mono, 16 kHz, int16 scale | SI lines 2, 173-174 |
| Samples per video frame | 640 | SI line 214 |
| MFCC | python_speech_features defaults: 13 coefficients, 26 filters, 25 ms window, 10 ms step, nfft 512, pre-emphasis 0.97, lifter 22, log-energy in coefficient 0 | SI line 201 (library call with defaults), python_speech_features `base.py` lines 25-27, 50 |
| MFCC per video frame / window | 4 / 20 | SI line 232 |
| Video window | 5 frames | SI line 227 |
| Face crop | 224 x 224, crop scale 0.40, median filter 13 on box size and centre, padding value 110, BGR values 0-255 unnormalised | RP lines 29, 117, 129-131, 135-148, 142; SI line 190 |
| Offset search | +-15 frames (`vshift` 15), batch 20 | run_syncnet.py defaults |
| Offset | `vshift - argmin(mean distance)`; confidence `median(mean distance) - min`; min dist | SI lines 246-252 |
| Windows scored | `min(frames, floor(samples/640)) - 5` | SI lines 214, 220 |

`python_speech_features` (MIT, Copyright (c) 2013 James Lyons) is not installed in this environment, so `mfcc`, `framesig`, `preemphasis`, `powspec`, the mel filterbank and the lifter were ported with their defaults into `preprocess.py`. Checked against the library source (fetched, run from a scratch folder): maximum absolute difference 0.0 on a random int16-scale signal.

Files: `src/dfd/platform/syncnet/{__init__,model,preprocess,scoring}.py`, `src/dfd/platform/modules/lip_sync.py` (class `LipSyncModule`, `build_lip_sync_module()`), `tests/platform_lipsync/test_lip_sync.py`, `data/_results/platform/lip_sync_smoke.json`.

## Deviations from the reference (not measured for effect)

- YuNet boxes instead of S3FD boxes; the crop rule was designed around S3FD boxes.
- One face track by nearest-centre matching, interpolated across gaps, instead of the reference's IOU tracker with shot detection; the reference requires tracks of at least 100 frames (RP line 30), this module accepts clips from 2 s.
- Frames come from one ffmpeg raw pipe, not jpg files at `-qscale:v 2`; audio from `dfd.prep.audio.extract_audio`, rounded to int16.
- Only the first 30 s are analysed. Weight loading is strict (the reference copies only the keys present).

## Module behaviour

Section `cross_modal`, source `real`, findings `consistent`, `potential_mismatch` or `inconclusive` only. Model block: name "SyncNet v2 (Chung and Zisserman, Oxford VGG)", weights hash above, licence "code MIT (joonson/syncnet_python); weights licence not stated by the source", evaluation "not evaluated". The Chung and Zisserman paper is NR (not retrieved) in docs/RESEARCH.md; nothing is attributed to its contents.

- Not applicable: not a video, no audio stream, or no face in at least half of the frames.
- Inconclusive: audio under 2 s; less than 20 % of 20 ms blocks above RMS 0.01; two or more faces in more than 10 % of frames (warning that speaker attribution is not established; no face is picked); confidence below 1.0.
- Otherwise `consistent` if |offset| is at most 2 frames, else `potential_mismatch`.
- Score: confidence / 10 clipped to [0, 1], `calibrated` false, `threshold` null. Raw offset, min distance and confidence are observations.
- **Thresholds are uncalibrated placeholders of this project.** The reference code has no decision threshold (it prints offset, min dist and confidence). The values 1.0, 2 frames, 10, 2 s, 0.01 and 20 % are not taken from the reference and have not been evaluated on this project's data. They are constructor arguments for a later preregistered evaluation.
- No speech detector: music or noise passes the audio check.
- Inference errors propagate; the coordinator records the module as failed.

## What was tested

- `tests/platform_lipsync`: 15 tests, 15 passed (CPU): applicability branches, identity fields, mapping, vocabulary and wording, schema validity, track building, MFCC shape, window arithmetic, error propagation, strict weights load and hash against SOURCE.json.
- Integration smoke test on GPU, one clip, `data/lavdf/LAV-DF/train/000473.mp4` (LAV-DF train split, metadata: no fake, no modified audio or video; the test split was not touched). Results in `data/_results/platform/lip_sync_smoke.json`. This is a smoke test, not an evaluation:
  - original: offset 0 frames, confidence 5.58, finding consistent;
  - audio delayed 400 ms: offset -10 frames (400 ms = 10 frames), confidence 5.26, finding potential_mismatch;
  - audio delayed 1000 ms (outside the +-15 frame range): confidence 1.25, offset 13 (unrelated), finding potential_mismatch. The placeholder rule does not separate this case from an in-range offset; the confidence fell as expected but stayed above the placeholder minimum of 1.0.
  No accuracy, false-alarm rate or robustness claim follows from three runs on one clip.

## Limitations

The module measures audio-visual temporal agreement of one face track. A mismatch is not proof of manipulation (dubbing, bad muxing, variable frame rate, off-sync recordings). A synchronised fake passes. Tested scope is one clip. The decision rule is a placeholder. Weights licence unresolved. Next steps from the decision note remain open: read the SyncNet paper, preregister the evaluation (LAV-DF within-video comparison, false-positive rate on real videos), choose thresholds on development data only.

## Addendum 2026-10-26 (E-022): behaviour check on controlled cases
Module version syncnet-v2-port-2. Evidence and numbers: docs/EXPERIMENTS.md E-022 (development set only, LAV-DF train clips, no held-out check). Changes: MIN_CONFIDENCE 1.0 to 3.0; a heuristic speech-likelihood gate (spectral flatness and loudness variation, not a speech detector) gives inconclusive AUDIO_NOT_SPEECH_LIKE; an offset at the +-15 frame search boundary gives inconclusive OFFSET_AT_SEARCH_BOUNDARY. The statements above that say "No speech detector: music or noise passes" and "confidence below 1.0" describe port-1. Weights licence remains not stated. Tests: tests/platform_lipsync, 19 passed.

## Addendum 2026-10-26 (overnight): E-025 and syncnet-v2-port-3

The +-2 frame tolerance was replaced by an asymmetric band of -6 to +2 frames fitted on 230 real clips and validated on 230 disjoint real clips (E-025, D-073): false alarms on real clips 1.3 percent [0.4, 3.8] (3 of 230), against 11.7 percent for +-2 on the same clips. The same band flags none of 160 Wav2Lip-based FakeAVCeleb fakes (E-024 had 85 to 93 percent with +-2), so the offset cue does not separate them from real clips. The module is a measure of audio-visual offset on VoxCeleb2-derived talking heads, not a detector of lip-sync fakes. The SyncNet weights licence remains unresolved.
