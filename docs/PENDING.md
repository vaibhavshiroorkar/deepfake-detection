# Pending work

Everything still open, in the order it will happen. A finished item moves to the Unreleased section of the [changelog](CHANGELOG.md).
The reasons behind each item are in the [project log](PROJECT-LOG.md). The rules for each test are in [P-001](../results/preregistration/P-001-shortcut-neutral-av-study.md).

## 0. Exhibition on 20 November (priorities set by the college, 11-17)

Internal and external examiners sit together. First priority: a live demonstration of the working project. Second: the PPT in the college's structure. Third: the guide-approved report and its printout. Fourth: the certificate on bond paper for the external examiner's signature. The White Book can follow later.

The demo, the PPT, the overnight checks and `docs/SECURITY.md` are done and merged into `main` (changelog). The demo runs from the home PC and is opened on the laptop through Tailscale (decided by Vaibhav 11-17).

**Claude, the 18th:** the deliverables remake and the report in the college format are done (changelog). Vaibhav: upload `report.zip` and `paper.zip` to Overleaf (both compile locally with Tectonic: report 105 pages, paper 9), and confirm the new paper title with the guide: "Multimodal Audiovisual Deepfake Detection: Evaluating Face, Voice, and Real-Only Lip-Sync Cues Beyond Dataset Shortcuts" (chosen by Vaibhav 11-18).

**Claude, the 18th (decided by Vaibhav 11-18, in this order):**

1. Measure the voice model on today's voice recordings before trusting voice-only answers beyond a rough guide.
2. Public site hardening, before it is left public unattended (`docs/SECURITY.md`): per-visitor rate limits behind the Funnel proxy, memory limits on FFmpeg, uploads in a container without `data/`. Until then `-Public` only while someone watches.

**Claude, the night of the 19th: generated-video check follow-ups.** The check is live (changelog). Open: rescore the demo clips and update `docs/DEMO.md`; finish the external tests (`runs/gen-probe/`, own Veo 3 set and DF26 fakes with sound; DF26 real clips have no sound track, so the site answers "can't check" for them) and the real-speech voice test (`scripts/voice_realworld.py`); put the results in the changelog. Pictures: the picture calibration has no generated-video reference yet, so the check is ignored for pictures.

**Lines for videos in parts:** measured and live (changelog). Any rebuild of `web-calibration-v2.json` drops them; rerun `scripts/build_parts_lines.py` after one.

**Handover for a new chat (written 11-20).** State: the public site runs on this PC (`scripts/demo_host.ps1`, GPU, Tailscale funnel) and must never be stopped; to load new Python code, stop and start again with `-Public`, never filter the start command's output, then confirm "Funnel on" with `-Action status`. GitHub Pages serves the pages; `main` on GitHub has everything. Run `scripts/ci_local.sh` before each push and do not edit files while a push's CI runs (it tests the working tree). Open items:
- **The middle answer (Vaibhav to decide).** Declaring "Real" was tested and rejected (`results/verdict/report.json`: a familiarity check called every DF26 2026-generator fake familiar). Options offered: rename "Can't tell" to "Some signs, not enough to flag" (recommended), fold it into "No signs found", or keep it.
- **The outside face check (Vaibhav to decide).** Measure it on our own clips (about 600 of the month's 2,000 operations: 30 real and 30 fake stills from our test sets) to set its line and choose one model or two; and choose between calling it in parallel (faster, now) or only when our own checks have not already flagged (cheaper). Do not mention the service by name in docs or on the page (Vaibhav, 11-19).
- **Untracked results to sort (Vaibhav):** the run logs are now git-ignored. Left: `results/v2-eval/scores.jsonl` and `scores-{0,1,2}-of-3.jsonl` (different runs, not copies of each other: commit the one `report.json` was built from), `results/voice-realworld/report.json` (the real-speech voice test, still to be written into the changelog) and `results/realworld/user-reals.csv` (scores of Vaibhav's own recordings).

**Headset false alarm (found 11-19, measured 11-20).** Vaibhav's 21 laptop-microphone recordings read as real; only a headset webcam video sent through WhatsApp flags (voice 1.00). Vaibhav: record one sentence twice in Voice Recorder, with and without the headset. If only the headset one flags, add a page note that noise-cancelling headsets can make a real voice read as processed.

**Claude, after the 20th: website models, next stage.** The voice swap is done (changelog). Open: (1) score Vaibhav's own phone and laptop recordings, real and voice-cloned, as a held-out set (names like `real_phone_vaibhav_01.mp4`); never train on them. (2) A face model trained on face swaps and generated video together, judged on DF26 and Veo 3; no tested face model works on DF26. (3) Multi-dataset fine-tuning of the voice model only if (1) shows false alarms on real recordings. DF26 and MNW stay evaluation-only. (4) LAV-DF word edits dropped from 37% to 8% caught with the new voice model; check whether the lip-sync model can carry them.

**Claude, also:**

- On request: answer Overleaf compile errors; adjust slides or the demo after Vaibhav's review and rehearsal.

**Deferred until after the 20th:** before any public hosting, the hardening listed in `docs/SECURITY.md` (time and memory limits on every FFmpeg call, an allowlist unpickler for AV-HuBERT, an admin review, uploads in a container); the Docker image and Hugging Face Space, the admin carry-over, retiring the old in-browser demo, renaming the Streamlit dashboard, the deep wording pass and the comparison tables (unless time is left on the 19th).

**Vaibhav, the 18th:**

1. Set up the demo once (`docs/DEMO.md`): Tailscale on the PC and the laptop with one account; the PC never sleeps and Windows Update is paused for a week; the 7 demo clips copied to the laptop (`data/demo/upload` and `data/demo/watch`); one test from outside the house.
2. Check the rebuilt deliverables: the PPT (`deliverables/build/Deepfake_Detection_Presentation.pdf`), the overnight results in the paper and report, and the "Need" slide's four sourced figures.
3. Compile `report.zip` and `paper.zip` on Overleaf; get the report approved by the guide; print it; spiral-bind one copy if the guide verifies it.
4. Check the names on the report's certificate page and print it on bond paper.

**Vaibhav, the 19th:** rehearse the demo once with `docs/DEMO.md`, and the talk once with the speaker notes.

**Vaibhav, the 20th:** before leaving home run `scripts/demo_host.ps1 -Action start` and `-Action status`, and note the address; carry the report printout, the spiral-bound copy, the certificate and the laptop; get the external examiner's signature.

## 2b. P-002 preparation, alongside (Claude)

Chosen 11-16: prepare the rebuild now, with no GPU and no change to anything H2 uses; build it after the submission. Design: [P-002 design](research/P-002-design.md).

- Freeze P-002 (`results/preregistration/P-002-draft.md`, now drafted) after the baselines are reproduced: set the bars of Q1 to Q6, then rename it to `P-002.md` and mark it frozen, before any P-002 model is trained.
- AV-Deepfake1M++: size (1.46 TB) and licence (gated EULA) are checked; the metadata format is still unknown. Decide on a stratified subset or an external disk, since this PC has about 519 GB free. Follow up on access.
- Confirm the assumed licences in `docs/research/P-002-backbones.md` (CLIP, XLS-R) before P-002 uses them.

## 2c. The 18th, after the exhibition work in section 0 (decided 11-17 in a side chat; the 19th is for study)

- Read the overnight results (A15 to A19), now in the paper, report and a backup slide on `main` (merged 11-17 at Vaibhav's request). To undo them: the tag `pre-overnight`.
- If time is left: a deep wording pass on the paper, report and slide notes (words only, never claims or numbers), and the comparison tables drafted in the side chat (protocol table, raw and cleaned score tables). Every competitor cell must be checked against the paper's PDF first.
- A check of `README.md`, `docs/FINAL.md` and the handbook for stale numbers and links.

## 3. When H2 finishes (Claude)

- Cut version 1.0.0 at the final-year submission: changelog heading, `pyproject.toml`, tag `v1.0.0`.
- After the submission: reorganize `scripts/` (68 files) into `p001/` (current study), `data/` (fetching and manifests) and `legacy/` (the September to October pipeline). Deferred on 11-17: it touches 115 files that cite script paths, including the paper and report, and changes nothing a reader sees, so it is not worth the risk two days before submitting.

## 4. Vaibhav

- Confirm the report's front matter, all filled in from the October template:
  - the Head of Department, Dr. Jagannath Nirmal;
  - the logo, the Somaiya Vidyavihar mark, which stands in for `1777459645157_kjsce.png`;
  - the three roll numbers;
  - that the slide master from the October deck is still the college's current one.
- Ask the guide what "Batch" means on the report. It is set to 2023-27; the academic year is 2026-27.
- Upload `deliverables/build/overleaf/report.zip` and `paper.zip` to Overleaf, compile with pdfLaTeX, and send any error.
- Copy `fig_flow1.png` to `fig_flow7.png` from the Overleaf project into `deliverables/figures/` and commit them; until then `deliverables/overleaf.py` builds the report with framed placeholders for those seven flowcharts.
- Read the paper and the slides and challenge every claim. Practise the talk once with the speaker notes.
- Decide whether to delete `mlartifacts/`, `mlruns/` and `mlflow.db` (6.8 GB of September to October tracking records) and `dist/` (an old package build and the October `final-report.pdf`).
- Send the DeepSpeak reminder around 11-22 if there is no answer. Forward any reply from AV-Deepfake1M, AV-Deepfake1M++ or KoDF.

## 5. Frontends (decided 11-16, in progress)

- The web app becomes the main site: Home with a hero, Detector, Working, Research, About us, Contact (a form, sent through Resend to Vaibhav's mail), plus the hidden Admin page. Everything on the current site carries over and gets updated to P-001; Vaibhav decides later what to drop.
- Build approach: evolve the current FastAPI web app (approach A). Host it on a Hugging Face Docker Space (free CPU Basic: 2 vCPU, 16 GB, sleeps after 48 h idle). Check on Vaibhav's account that free Docker Spaces still work; fall back to this PC with a temporary link.
- Frontend built 11-17 (see the changelog), spike, calibration, detector backend and result card done 11-17 (changelog). Still open, after the 20th (section 0 takes the demo work first): the admin carry-over (including a switch to turn the detector off), the Docker image and the Hugging Face Space.
- The detector runs the P-001 pipeline on an upload: neutralize, the three paths, and the threshold.
- Retire the old in-browser demo: `hosting/browser/`, `docs/hosting.md`, the README link. Vaibhav deletes the Space on Hugging Face if he wants it gone.
- Keep the Streamlit dashboard as the teaching and experiment tool, rename it, and update its pages to the P-001 pipeline.

- **Voice model on real microphone speech (found 11-18).** The website's voice model is now AntiDeepfake XLS-R 1B (changelog), trained on many speech corpora; it beat the study's model on In-the-Wild, ASVspoof 2019 and Veo 3 (`results/v2-eval/report.json`). Still unmeasured: real laptop and phone speech. Vaibhav: record about 20 short real voice clips from different people and devices; Claude scores them as a held-out set (section 0, after the 20th, item 1).
- **Pictures on face-only data (asked by Vaibhav 11-19).** Step 1 is done: the 11-19 model test scored one frame per video of the FF++ and Celeb-DF test lists through the website's picture path (`results/v2-eval/report.json`). Step 2 (Claude, after the 20th): train a separate website picture model on FF++ training frames plus FakeAVCeleb, test on Celeb-DF, reusing the October team's FF++ split (E-005 in `results/legacy-team/EXPERIMENTS.md`). The P-001 face model stays untouched for the paper.
- Explainability (asked 11-18): the heatmaps now show on the result card (changelog). Not built: a free-text description of the picture by a vision-language model. A captioner can say what is in the frame but cannot tell what is fake, and it adds a downloaded model to review (`docs/SECURITY.md`). Vaibhav decides whether it is wanted.
- **Suspected generator model (asked by Vaibhav 11-19, planned for after the 20th).** Today the site names a tool only when the file's own label does. A model that guesses the generator (Veo, Sora, Kling, Seedance, Wan, Pixverse) needs: an "unknown" answer for tools it has not seen; a measured accuracy on held-out clips; and its own test set, since HiFi-AVDF is a test set for the site. Show it only above a measured confidence. DF26 stays evaluation-only.
- **Visible AI watermark check (found 11-19; asked again by Vaibhav 11-20: a Gemini sparkle should name Gemini in the Generator audit).** Vaibhav's Veo videos from Gemini carry the visible Gemini sparkle; Seedance shows "AI生成". A check that finds a known visible mark that stays fixed in a corner while the video moves would catch these openly labelled fakes. It must be tested on real videos for false alarms first. Not before the exhibition.
- **Voice model on re-sent videos (found 11-19).** One of Vaibhav's Veo videos, sent through WhatsApp, read as real to the voice model (raw scores -4.9 to -8.7), though it caught 96% of Veo 3 voices in the 11-19 test. Measure the voice model on the Veo 3 set before and after a WhatsApp-like re-encode (AAC at a low bitrate); if it drops, add such re-encodes to the held-out tests.
- **Save result as PDF (suggested 11-19).** A button that downloads the answer card, for the exhibition and for users.
- **Voice heat strip (suggested 11-19).** Which seconds the voice model found most unusual, beside the face heatmaps. Needs per-window voice scores from the scorer.
- **Lip-sync score moves with the cut (found 11-19).** The demo clip `FakeAVCeleb_cloned-voice_B__id00841__00078_fake.mp4` read lip sync 0.963 in `docs/demo-clips.csv` (parts from 0.0 s) and 0.399 through the live site (parts from 0.04 s), with the same answer (voice raised). A one-frame shift should not move a score that much. Claude, after the 20th: score the same clip at shifts of 0 to 80 ms and find which step sets the start.

## 5b. Dashboard and web app check (older notes)

- Decide whether the dashboard and web app (about 11,000 lines, 40% of `src/`) stay as the demo.
- If they go: delete them with the docs only they read: `docs/research/findings.md`, `docs/research/result-traceability.md`, `docs/handbook/01` to `10`, `docs/obstacles.md`, `docs/model-selection.md`, `docs/handoff.md` and its `ddf handoff update` command, `scripts/update_result_registry.py`, and the checks and tests that cover them.
- If they stay: update their content to the P-001 study. Today they cite the older findings.
- Write `docs/SECURITY.md` in the same review. The web app takes uploaded media, decodes it with ffmpeg and loads downloaded model weights; the public demo runs through an HTTPS tunnel; the admin page relies on `DDF_ADMIN_TOKEN`. For each one: the vulnerability, the countermeasure, and whether it is verified, assumed or unknown. Two are already verified: `.env` is git-ignored, and the sync scorer loads checkpoints with `weights_only=True`.

## 6. Full run, after the deadline

The full design, with what to keep, what to replace and why, is in [P-002 design](research/P-002-design.md). The items below are its first steps.

- Leave-one-corpus-out training as the main protocol: train on every corpus but one, test on the one left out, and rotate. It keeps every test unseen and adds training variety. It needs a new preregistration first, because LAV-DF and DFDC would stop being locked targets.
- H3: predict which modality was faked, with reverse models trained on LAV-DF (amendment A2).
- AVH-Align and HAVIC baselines on the same protocol.
- The StreamFusion MLP, the secondary fusion that A8 dropped for the 3-day window.
- AV-Deepfake1M, AV-Deepfake1M++, KoDF and DeepSpeak v2, if access arrives. They are scored by frozen models only.
- Measure speaker overlap between FakeAVCeleb, LAV-DF and AV-HuBERT's pretraining data (VoxCeleb2, LRS3).
- H1 residual: video bitrate still carries the label after neutralization. Test a fixed-bitrate encode under a new amendment.
- Make the neutralizer refuse a clip unless both its picture and its sound last 3.00 s, measured per stream, not from the container (handbook mistake 67; counts in `results/clip-lengths/summary.json`: 301 clips a frame short, 4 HiFi-AVDF clips with short sound, 23 with none). Apply it under a new amendment, since it changes which clips are kept.
- Finish neutralizing LAV-DF's dev and train sets (`runs/p001-resume2.sh`, last step), for H3. It stopped in the 11-16 crash and resumes from its cut log. 13 LAV-DF train files written before the crash cannot be read (ffprobe: invalid mvhd, found 11-19); redo them.
- Fix the `data/legacy-team/ffpp` links, which point at folders deleted with `team/`.
- Speed up the visual and audio cache build. It is CPU-bound at about 340 clips an hour when shared.
- Clear the remaining lint findings, mostly in the slide helpers.
- Write the full paper for a venue (ICASSP or Interspeech 2027, or a CVPR or ICCV forensics workshop).
