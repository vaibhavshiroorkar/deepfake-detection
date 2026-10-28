# DEMO: running the live dashboard for the evaluation

Checked on 2026-10-26 on the development machine (Windows 10, RTX 5070, `.venv` with Python 3.13.15) and again on 2026-10-27 after the dashboard redesign (D-076); sections 3 to 5 quote the 2026-10-27 browser run (`docs/test_evidence/2026-10-27_redesign/`), the earlier API outputs stay in `docs/test_evidence/2026-10-26/`. They show that the software runs end to end. They are NOT evidence of detector accuracy: accuracy is reported only by the logged experiments in `docs/EXPERIMENTS.md`.

## 1. What must be present

| Need | Check | If missing |
|---|---|---|
| `.venv` built from the lock, with C2PA | `uv sync --locked --extra c2pa` | see docs/SETUP.md sections 2 to 4 |
| ffmpeg and ffprobe on PATH | `ffmpeg -version` | video and audio cannot be analysed (docs/SETUP.md 1a) |
| Frozen detector checkpoint `checkpoints/I-5-freeze-candidate/` | `ls checkpoints/I-5-freeze-candidate/metadata.json` | the face, image and speech modules report `failed` (no checkpoint); the other modules still run |
| SyncNet weights `data/models/syncnet/syncnet_v2.model` and YuNet in `data/_prep_bench/models/` | `ls data/models/syncnet` | lip_sync reports `unsupported` |
| Generated-video head `checkpoints/genvideo-E020/head.npz` and DINOv2-small (`data/_logs/encoder_weights.json`) | `ls checkpoints/genvideo-E020` | no head: generated_video reports `unsupported`; head present but DINOv2 weights missing: it reports `failed` |
| Demo inputs `data/_demo_inputs/` | step 2 below | rebuild them with step 2 (needs the FF++, LAV-DF and ASVspoof folders under `data/`) |
| A browser (any: Edge, Chrome, Firefox) | open the URL | the API still works with `curl` |

All of these are git-ignored local files; nothing is downloaded by the demo.

## 2. Prepare the inputs (once)

```
PYTHONPATH=src .venv/Scripts/python.exe scripts/platform/make_demo_inputs.py
```

It copies non-sealed files only (FF++ and ASVspoof validation or development splits, resolved with the split guard; one LAV-DF train clip; synthetic files) into `data/_demo_inputs/`:

| File | What it is |
|---|---|
| `1_face_swap_video_ffpp_val.mp4` | FF++ Deepfakes face swap, validation split, no audio |
| `2_original_video_ffpp_val.mp4` | its FF++ original (unmanipulated) video, validation split, no audio |
| `3_talking_head_with_audio_lavdf_train.mp4` | LAV-DF train clip with speech audio |
| `4_speech_audio_asvspoof_dev.flac` | ASVspoof 2019 LA development audio (a spoofed utterance) |
| `5_face_image_frame_ffpp_val.png` | one frame of file 1 as a still image |
| `6_synthetic_no_face_tone_audio.mp4` | synthetic test pattern, no face, 440 Hz tone |
| `7_unsupported_notes.txt` | unsupported type (error case) |
| `8_corrupt_video.mp4` | bytes that are not a video (error case) |

Do not use files from any `test_x` split or from the DF26 or FakeAVCeleb sealed halves.

## 3. Start the dashboard

```
PYTHONPATH=src .venv/Scripts/python.exe -m dfd.platform.dashboard.serve 8002 --live
```

Open `http://127.0.0.1:8002/` (127.0.0.1, not localhost). The first analysis loads the models (11.6 s in the browser run of 2026-10-27);
later ones took 0.4 to 2.9 s. Stop with Ctrl+C.

The page opens on the upload panel. No report of any kind is on screen until a file is analysed (redesign of 2026-10-27, D-076; until
2026-10-26 a MOCK scenario was shown at page load). The pill at the top right reads "Live analysis". Invented example reports are still
available from "Demo scenarios" at the top right; choosing one shows a hatched MOCK DATA banner and a MOCK badge on every check, and its
download is named `MOCK_...json`. Without `--live` the page says live analysis is not available, the pill reads "Demo data only", and the
first scenario is shown under the MOCK banner.

## 4. Run the demonstration

1. Drop a file on the upload area, or press "choose a file". The panel shows the file name, type, size, duration (video) and a preview
   played in the browser; nothing is analysed yet. The accepted types and the 512 MB limit are listed under the upload area. Below it
   the panel lists the checks that will try to run on that kind of file, three steps saying what happens on Analyse, and "Which checks
   run on which files" (added 2026-10-27, D-082).
2. Press Analyse. The status line goes Uploading, Queued, Running (with the human name of the check that is running and "N of M checks
   done"), then Done or Failed; a step list shows the checks as they run.
3. The report appears below, in this order: the file analysed (type, duration, dimensions, streams, size, completeness), "What the checks
   found" (a counting sentence, one chip per group, what is known about faces and speech, "Every check at a glance" with one row per check
   linking to its card, "No overall score or verdict is given"), checks
   that point in different directions (only when there are some), the checks grouped by meaning, "What this report cannot tell you", and
   the whole-report technical details.
4. Each check shows its human name, a status chip, a headline in plain words, one line on what it checks, at most one rounded number, and
   one line on what the result does not establish. "How this result was reached" opens the explanation under fixed headings: What we did,
   What we examined, What we observed, Why this was reported, What it means, What it does not establish, Limitations (D-082), plus the
   lip-sync offset curve or the 8 frames the experimental video model saw. "Show where each sentence comes from" (at the bottom) adds the field each sentence was built from. "Technical
   details" holds the raw fields: module id, version, raw score and threshold, model, weights hash, evaluation line with experiment ids,
   tested scope, warnings, observations.
5. "Analyse another file" (next to the status line and at the bottom) returns to the empty upload panel.

Observed on 2026-10-27 in Edge (scripts/platform/ui_demo_run.py; docs/test_evidence/2026-10-27_redesign/). Group as the page shows it, with the
rounded number where the card shows one:

| File | Face manipulation | Image | Synthetic speech | Audio-visual sync | AI-generated video (experimental) | File information, credentials |
|---|---|---|---|---|---|---|
| 1 face swap | merits attention, 1.00 (line 0.61) | did not apply | did not apply: no audio stream | did not apply: no audio stream | experimental, no finding | file information; no credentials found |
| 2 original | nothing found, 0.20 (line 0.61) | did not apply | did not apply: no audio stream | did not apply: no audio stream | experimental, no finding | file information; no credentials found |
| 3 talking head | nothing found, 0.32 (line 0.61) | did not apply | nothing found, 0.30 (line 0.70) | nothing found, best offset 0 frames | experimental, no finding (ranking score 1.00) | file information; no credentials found |
| 4 speech audio | did not apply | did not apply | nothing found, 0.15 (line 0.70) | did not apply | did not apply | file information; no credentials found |
| 5 image frame | did not apply | merits attention, 1.00 (line 0.61), face-crop path | did not apply | did not apply | did not apply | file information; no credentials found |
| 6 no face, tone | did not apply: no face | did not apply | nothing found, 0.00 (line 0.70), with "Audio may not be speech" | inconclusive: audio not speech-like | experimental, no finding (ranking score 1.00) | file information; no credentials found |
| 7 notes.txt | Failed: `UNSUPPORTED_MEDIA_TYPE`, "This kind of file is not supported.", no report | | | | | |
| 8 corrupt.mp4 | Failed: `UNREADABLE_FILE`, "The file could not be read; it may be damaged.", no report | | | | | |

Points to say while showing it:

- Each check answers for itself; the summary counts checks and never combines them into a score or an "authentic" answer.
- File 4 is a spoofed ASVspoof utterance and the speech check did not flag it: one clip is not an accuracy measurement, and the speech
  model's calibration is exploratory. Accuracy comes from the logged experiments only.
- The AI-generated video check sits in its own group, "Experimental, no finding made". Open its explanation: it shows the 8 frames the model
  saw, says the score is a ranking score and not a probability, that on the independent DF26 benchmark it ranked generated and real videos
  no better than chance (E-020), and that real videos also receive high scores (E-026: 338 of 440 real clips from FF++ and LAV-DF scored
  above its threshold). File 3 is a real clip with ranking score 1.00, which is exactly that problem.
- Audio-visual synchronisation measures timing only. Open its explanation to show the offset curve and the shaded band: "no mismatch" does
  not establish that the video is unmanipulated, and a mismatch would not prove manipulation.
- File 6 shows a known limitation honestly: the speech model scores any sound, so a pure tone still gets a speech result. The card carries
  "Audio may not be speech" because the synchronisation check's speech screen found the audio unlike speech; the speech result itself is
  unchanged (the detector is frozen).
- File 5 shows another one: the image analysis checks a still image with a face only with the face-swap model built for video frames.
  Since D-083 (2026-10-27) a separate card, "AI-generated image, whole image (experimental)", also gives the whole image to the
  generated-image model and shows its score under "Experimental, no finding made": when measured on 3,119 images with a face (E-028) it missed 81 percent of the generated ones, so
  the card makes no finding; the demo image's low score says nothing about reliability. Point out that the image analysis result is unchanged.
- No credentials found is not evidence of anything; C2PA can only report valid credentials with an operator-configured trust anchor.

## 5. Reading the outcomes

| What you see | Meaning |
|---|---|
| Group "Merits attention" (red triangle, or amber for a timing mismatch) | a frozen detector's decision at its fixed decision line, inside its tested scope only; or a timing mismatch, which is not proof of manipulation |
| Group "Nothing found within tested scope" (slate dash) | the check ran and found nothing it looks for; it does not establish that the file is unmanipulated |
| Group "Inconclusive" (question mark) | the check ran but reached no conclusion; its explanation quotes the reason |
| Group "File information" | metadata described the file; not evidence either way |
| Group "Experimental, no finding made" (flask, EXPERIMENTAL badge) | a research prototype; its score is shown as an experimental signal only and is never a finding |
| Group "Did not apply to this file" | the check does not apply to this input (no face, no audio, wrong media type); the reason is on the card |
| Group "Not available" | a required model or file is not installed |
| Group "Did not complete" (wrench) | that check failed or ran out of time; the other checks are still reported and the report is marked partial |
| Status "Failed: CODE: message" and no report | the whole file was rejected; nothing was analysed |

## 6. If something is unavailable

- Port in use: start with another port (`... serve 8003 --live`).
- No browser: drive the API directly: `curl -F "file=@data/_demo_inputs/3_talking_head_with_audio_lavdf_train.mp4" http://127.0.0.1:8002/api/analyse` returns a `job_id`; then `curl http://127.0.0.1:8002/api/jobs/<job_id>` until `state` is `done` or `failed`.
- A model folder missing: the corresponding card shows UNSUPPORTED or FAILED with the reason; the remaining modules still run, which can itself be shown.
- GPU unavailable: the modules fall back to CPU (slower; not timed here).

## 7. The older single-score page (frozen v1, optional)

The frozen submission page still runs: `PYTHONPATH=src .venv/Scripts/python.exe -m dfd.web --adapter real --checkpoint checkpoints/I-5-freeze-candidate --port 8001`, then open `http://127.0.0.1:8001/`. Without `--checkpoint` it refuses to start (by design: there is no default checkpoint and no fallback to the mock). On 2026-10-26 it answered `/api/analyse` for files 3, 5 and 4 with `inference_source: real` and no errors. The platform dashboard (section 3) is the one to demonstrate; the v1 page is kept unchanged as the frozen submission path.
