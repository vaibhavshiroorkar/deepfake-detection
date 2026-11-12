# FINAL DEMO RUNBOOK

Written 2026-10-30 for Task 31 (final end-to-end demonstration and its MP4 recording). Based on the code (deliverables_v2/website/serve_v2.py, src/dfd/platform/dashboard/serve.py), docs/DEMO.md, deliverables_v2/website/README.md and submission/README.md.

**Verification state.** The live website was last exercised end to end on 2026-10-28 and 2026-10-29 in Edge through Playwright (deliverables_v2/website/test_evidence/ui_v2_run.json: `problems: 0`; docs/TESTING_RECORD.md sections 16 and 17). It was **not** re-run while writing this runbook on 2026-10-30: no server was started. On 2026-10-30 the following were confirmed present on the development machine: `.venv/Scripts/python.exe`, ffmpeg on PATH, `checkpoints/I-5-freeze-candidate/` (metadata.json, weights_T-V.pt, head_T-I.npz, three speech heads), `data/models/syncnet/syncnet_v2.model`, and the eight demo inputs in `data/_demo_inputs/`. Do a full rehearsal (section 3) before recording.

## 1. Setup and prerequisites

| Need | Check command (run from the repository root) | If missing |
|---|---|---|
| Repository at the final tag | `git describe --tags` shows `submission-final` (or a later agreed commit) | `git checkout submission-final` (only if the team agrees; uncommitted files stay) |
| Python environment | `.venv/Scripts/python.exe --version` gives 3.13.15 | `uv sync --extra c2pa` (docs/SETUP.md section 2) |
| C2PA extra (optional) | `.venv/Scripts/python.exe -c "import c2pa"` | without it the credentials card does a marker scan only; say so if asked |
| ffmpeg and ffprobe | `ffmpeg -version` | video and audio cannot be analysed |
| Frozen checkpoint | `ls checkpoints/I-5-freeze-candidate/metadata.json` | face, image and speech cards show "Analysis failed" |
| SyncNet weights | `ls data/models/syncnet/syncnet_v2.model` | lip-sync card shows "Not available" |
| Generated-video head | `ls checkpoints/genvideo-E020/head.npz` | that card shows "Not available" |
| Demo inputs | `ls data/_demo_inputs` (8 files plus SOURCE.json) | rebuild: `PYTHONPATH=src .venv/Scripts/python.exe scripts/platform/make_demo_inputs.py` (copies non-sealed files only) |
| Browser | Edge, Chrome or Firefox | the API still works with curl (section 7) |
| Screen recorder | OBS Studio (records terminal and browser, MP4 output) or Xbox Game Bar (Win+Alt+R, records one app window to MP4) | install before the session |

**Start the server from the repository root.** Two model paths in the code are relative (prep/face.py, sweep/encoders.py), so starting elsewhere makes the detector cards fail.

## 2. Launch the correct application

```
PYTHONPATH=src .venv/Scripts/python.exe deliverables_v2/website/serve_v2.py 8012 --live
```

- Binds 127.0.0.1 only. Open **http://127.0.0.1:8012/** (use 127.0.0.1, not localhost).
- Evaluation Lab: **http://127.0.0.1:8012/#lab**.
- Stop with Ctrl+C.
- Ports used by the project: 8012 this site, 8002 original dashboard, 8000 the frozen v1 page by default. **Do not demonstrate the v1 page**: its wording ("Likely fake", "calibrated probability") predates D-087.
- Without `--live` the page shows invented scenarios under a MOCK banner. Make sure the header pill shows live analysis before recording.

Services and dependencies verified from the code: Flask app from `create_app` with routes `/api/mode`, `/api/vocabulary`, `/api/analyse` (live), `/api/jobs/<id>` (live), `/api/evallab/*`; one worker thread; uploads saved to the system temp folder and deleted after each job; 512 MB limit.

## 3. Authorised demonstration inputs

Only these files, all prepared by `scripts/platform/make_demo_inputs.py` from **non-sealed** splits or synthetic material (docs/DEMO.md section 2):

| File | Content | Use in the recording |
|---|---|---|
| `5_face_image_frame_ffpp_val.png` | one frame of an FF++ validation face-swap video | image path |
| `3_talking_head_with_audio_lavdf_train.mp4` | LAV-DF train clip with speech | video with sound: face, speech, lip sync, experimental generated video |
| `1_face_swap_video_ffpp_val.mp4` | FF++ face swap, validation, no audio | a face-video flag |
| `2_original_video_ffpp_val.mp4` | its FF++ original | nothing found within scope |
| `4_speech_audio_asvspoof_dev.flac` | a spoofed ASVspoof development utterance | audio path (the speech card did not flag it on 2026-10-27: say "one clip is not a measurement") |
| `6_synthetic_no_face_tone_audio.mp4` | synthetic pattern with a 440 Hz tone | a limitation: any sound gets a speech score; "Audio may not be speech" |
| `7_unsupported_notes.txt`, `8_corrupt_video.mp4` | error cases | input validation |

**Never open** any file of Celeb-DF, Celeb-DF++, FakeAVCeleb, DF26, or any test_in or test_x split on camera. Their terms forbid publishing derived data, and the sealed sets must not be touched.

## 4. Live demonstration script

Expected outputs below come from the recorded runs of 2026-10-27 (docs/DEMO.md section 4 table) and 2026-10-28 (v2 site). Labels on the v2 site are the D-087 verdict labels.

| Step | Action | Expected UI | What to say |
|---|---|---|---|
| 1 | Show the terminal at the repository root: `git describe --tags`, `ls checkpoints/I-5-freeze-candidate` | tag name; checkpoint files | "This is the final tagged version. The frozen checkpoint is local; weights and datasets are not distributed because of their licences." |
| 2 | Start the server (section 2) | "Running on http://127.0.0.1:8012" | "One local Flask server; it binds to this machine only." |
| 3 | Open the page | upload stage; accepted types and the 512 MB limit; live pill | "You can say what you are analysing, or let the page detect it from the file." |
| 4 | Upload file 5 (image), press Analyse | progress, "N of M checks done", then results; the image card on the face-crop path with a weak-evidence label; an experimental whole-image card under "Experimental, no finding made"; metadata and "No content credentials found" | "Each check reports for itself. The bar says model output, not a probability. The experimental card never makes a finding: on modern generators the image model misses most fakes (E-028)." |
| 5 | Open "How this was reached" and "Technical details" on the image card | explanation headings; raw score, threshold 0.6139, weights hash, evaluation line | "Every sentence is built from fields of the result; the threshold was fixed on validation data." |
| 6 | "Analyse another file", upload file 3 (talking head with audio) | face card "No strong evidence of manipulation detected"; speech card; lip-sync card with the offset curve and the shaded band -6 to +2; generated-video card experimental | "Lip sync measures timing only: a well-timed fake passes (0 of 160 Wav2Lip clips flagged, E-025). The generated-video card is at chance on held-out data and flags many real clips (E-020, E-026), so it never decides." |
| 7 | Upload file 1 (face swap, no audio) | face card "Possible manipulation" with high uncertainty; speech and lip sync "Did not apply: no audio stream" | "A flag is a possible sign. The false-alarm rate of this threshold was about 1 percent on FF++ but 11 to 14 percent on other real videos (E-032, post hoc)." |
| 8 | Upload file 4 (audio) | speech card; image and video cards did not apply | "Only the first four seconds are scored. One clip is not an accuracy measurement." |
| 9 | Upload file 7, then file 8 | error stage with "Unsupported input", codes UNSUPPORTED_MEDIA_TYPE and UNREADABLE_FILE | "Bad inputs are rejected before any model runs." |
| 10 | Scroll to "What this report cannot tell you" | the fixed statement | "Nothing on this page says a file is authentic or proven fake." |
| 11 | Open #lab (Evaluation Lab), choose the stored I-4 score file and a development block, show the settings | catalogue of stored score files; the guard text | "This page re-analyses stored scores with our metrics and group bootstrap. It runs no model and cannot approve a detector. A final split is accepted only as a re-analysis of a read already in the ledger." Do not start a final re-analysis on camera unless rehearsed. |
| 12 | Measured performance evidence (section 5) | terminal outputs | "These are the logged numbers behind the paper." |

## 5. Measured performance evidence without new research runs

All of these read stored files or run the software tests; none trains, evaluates a new model or reads a sealed set for the first time.

| Evidence | Command (repository root) | Expected | Notes |
|---|---|---|---|
| Software tests | `CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe -m pytest tests -p no:cacheprovider -rs -q` with the 13 data-dependent tests deselected (ids in docs/TESTING_RECORD.md section 2) | last recorded 713 passed, 2 skipped, 13 deselected | takes a few minutes; record it separately and cut, or show the saved log; say "software behaviour, not accuracy" |
| Post hoc false-alarm rates | `.venv/Scripts/python.exe scripts/eval/benchmark_audit_2026-10-27/frozen_rates.py I-4 ffpp:test_in celebdf_family:test_x dfd:test_x` | FPR about 0.0143, 0.1098, 0.1405 | recomputes from stored sealed scores; it prints a SHA-256 of a score file; do not show per-file rows |
| Sealed results table | open the slide "Sealed tests T1 to T3" or docs/EXPERIMENTS.md E-014 result block | T1 +0.1980, T2 +0.1633, T3 +0.0176 with the Bonferroni caveat | these are the logged numbers |
| Ledger | `wc -l ledger/final_ledger.jsonl` | 76 | proves one read per tag and dataset |
| Run time | the first analysis includes model loading (11.6 s on 2026-10-27); later ones took 0.4 to 2.9 s | | observation from one run, not a benchmark |

Do **not** run `python -m dfd.eval.run --final`, any training script, or any script under scripts/freeze/ during the demo.

## 6. Avoid exposing secrets, private data and local paths

- Close other browser tabs, email, chat and the GitHub token page. Never show `gh auth status --show-token` or environment variables.
- Do not open `data/` folders other than `data/_demo_inputs/`. Do not open `data/_final/` CSV files on screen.
- The terminal prompt shows `D:\Final Year Project` and the Windows user name `admin`; acceptable, but you can set a short prompt first (`PS1='$ '` in Git Bash).
- The page's technical details show the uploaded file's SHA-256 and weights hashes: harmless.
- Do not show the private repository URL with access tokens; showing `git remote -v` is fine if it contains no token.

## 7. Known failures and recovery

| Symptom | Cause | Recovery |
|---|---|---|
| "Address already in use" | port 8012 busy | start on 8013: `... serve_v2.py 8013 --live` and open that port |
| Page shows "Demo data only" and a MOCK banner | started without `--live` | stop and restart with `--live` |
| Face, image and speech cards "Analysis failed" | server started outside the repository root, or checkpoint missing | stop, `cd` to the root, restart; check section 1 |
| Lip-sync "Not available" | SyncNet weights missing | say so; the other checks still run (that itself is a feature to show) |
| First analysis slow | models loading on first use | wait; mention it |
| GPU busy or missing | another process; driver | the modules fall back to CPU (slower, not timed) |
| Browser cannot reach the page | used localhost or another host | use http://127.0.0.1:8012/ |
| No browser available | | `curl -F "file=@data/_demo_inputs/3_talking_head_with_audio_lavdf_train.mp4" http://127.0.0.1:8012/api/analyse` returns a job id; then `curl http://127.0.0.1:8012/api/jobs/<job_id>` |

## 8. Recording shot list (target about 8 to 10 minutes)

| Time | Shot | Content |
|---|---|---|
| 0:00 to 0:30 | title card or slide 1 | project title, three authors, guide, "final version, tag submission-final" |
| 0:30 to 1:30 | terminal | tag, checkpoint listing, environment check (Python, ffmpeg), start the server |
| 1:30 to 2:00 | browser | landing page, live pill, accepted types |
| 2:00 to 3:15 | browser | image analysis (step 4 and 5) |
| 3:15 to 4:45 | browser | video with sound (step 6), lip-sync curve, experimental card explanation |
| 4:45 to 5:30 | browser | face-swap video (step 7) |
| 5:30 to 6:00 | browser | audio file (step 8) |
| 6:00 to 6:30 | browser | error cases (step 9) |
| 6:30 to 7:15 | browser | "What this report cannot tell you"; Evaluation Lab (step 11) |
| 7:15 to 8:30 | terminal and slide | test result, frozen_rates output, ledger count, T1 to T3 slide |
| 8:30 to 9:00 | slide | limitations and closing sentence |

Speak the "What to say" column; or record silently and add captions. Each member can narrate the part they own (after the allocation is approved).

## 9. Final MP4 verification checklist

- [ ] File is MP4 (H.264 video, AAC audio if narrated); plays in VLC and in the browser.
- [ ] Resolution at least 1280 by 720; text in the terminal and on cards is readable.
- [ ] Shows: final tag, setup check, server start in live mode, at least one image, one video with sound and one audio input, the per-check cards and an explanation, an error case, the Evaluation Lab, and measured evidence (tests, logged sealed results, false-alarm replay).
- [ ] No MOCK banner during the live part (unless explicitly introduced as mock).
- [ ] No restricted dataset media (Celeb-DF, Celeb-DF++, FakeAVCeleb, DF26, sealed splits).
- [ ] No token, password, private email or unrelated window.
- [ ] Nothing said that the system proves a file real or fake, or calls a score a probability.
- [ ] Record the SHA-256: `sha256sum <file>.mp4`, and the duration: `ffprobe -v error -show_entries format=duration -of csv=p=0 <file>.mp4`.
- [ ] Add the file name, hash and duration to submission/README.md and regenerate MANIFEST.txt (scripts/eval/make_submission_manifest.py), or upload the MP4 separately as the task requires.

## 10. Files required for hand-over and reproducibility

| Item | Path | Included in submission/ | Why |
|---|---|---|---|
| Source code | src/, deliverables_v2/website/, scripts/, tests/ | no (repository, tag submission-final) | the software |
| Environment | pyproject.toml, uv.lock, docs/SETUP.md | no (repository) | rebuild the environment |
| Records | docs/DECISIONS.md, EXPERIMENTS.md, PROTOCOL.md, STATUS.md | yes (submission/records/) | evidence behind every number |
| Ledger | ledger/final_ledger.jsonl | no (repository) | proof of single sealed reads |
| Deliverables | submission/*.pdf, sources/ | yes | Tasks 28 to 30 |
| Demo recording | to be produced | to add | Task 31 |
| Checkpoints, datasets, sealed scores | checkpoints/, data/ | **no**, by design | licences (non-commercial data, research-only weights) |
| Demo guide | docs/DEMO.md, this file | repository | repeat the demo |
