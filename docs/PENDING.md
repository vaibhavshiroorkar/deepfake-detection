# Pending work

Everything still open, in the order it will happen. A finished item moves to the Unreleased section of the [changelog](CHANGELOG.md).
The reasons behind each item are in the [project log](PROJECT-LOG.md). The rules for each test are in [P-001](../results/preregistration/P-001-shortcut-neutral-av-study.md).

## 0. Exhibition on 20 November (priorities set by the college, 11-17)

Internal and external examiners sit together. First priority: a live demonstration of the working project. Second: the PPT in the college's structure. Third: the guide-approved report and its printout. Fourth: the certificate on bond paper for the external examiner's signature. The White Book can follow later.

The demo, the PPT, the overnight checks and `docs/SECURITY.md` are done and merged into `main` (changelog). The demo runs from the home PC and is opened on the laptop through Tailscale (decided by Vaibhav 11-17).

**Claude, the 18th:** the deliverables remake and the report in the college format are done (changelog). Vaibhav: upload `report.zip` and `paper.zip` to Overleaf (both compile locally with Tectonic: report 105 pages, paper 9), and confirm the new paper title with the guide: "Multimodal Audiovisual Deepfake Detection: Evaluating Face, Voice, and Real-Only Lip-Sync Cues Beyond Dataset Shortcuts" (chosen by Vaibhav 11-18).

**Claude, the 18th (decided by Vaibhav 11-18, in this order):**

1. Picture calibration (`scripts/build_picture_calibration.py`, running on the GPU): a real face picture scored above 99 in 100 real video clips, so pictures get their own reference and are refused until it exists. When it is built: check its false-alarm and catch rates, recheck a real face picture, then commit it with `docs/DEMO.md` and `docs/demo-clips.csv` (already rescored with whole-video checking: Wav2Lip clip now "can't tell") and update the report's picture paragraph.
2. Measure the voice model on today's voice recordings before trusting voice-only answers beyond a rough guide.

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
- Read the paper and the slides and challenge every claim. Practise the talk once with the speaker notes.
- Decide whether to delete `mlartifacts/`, `mlruns/` and `mlflow.db` (6.8 GB of September to October tracking records) and `dist/` (an old package build and the October `final-report.pdf`).
- Send the DeepSpeak reminder around 11-22 if there is no answer. Forward any reply from AV-Deepfake1M, AV-Deepfake1M++ or KoDF.

## 5. Frontends (decided 11-16, in progress)

- The web app becomes the main site: Home with a hero, Detector, How it works, Research, About us, Contact (a form, sent through Resend to Vaibhav's mail), plus the hidden Admin page. Everything on the current site carries over and gets updated to P-001; Vaibhav decides later what to drop.
- Build approach: evolve the current FastAPI web app (approach A). Host it on a Hugging Face Docker Space (free CPU Basic: 2 vCPU, 16 GB, sleeps after 48 h idle). Check on Vaibhav's account that free Docker Spaces still work; fall back to this PC with a temporary link.
- Frontend built 11-17 (see the changelog), spike, calibration, detector backend and result card done 11-17 (changelog). Still open, after the 20th (section 0 takes the demo work first): the admin carry-over (including a switch to turn the detector off), the Docker image and the Hugging Face Space.
- Vaibhav: decide the sample clips. FakeAVCeleb clips probably cannot be shown publicly; record your own (with consent for a fake version), or launch without samples.
- The detector runs the P-001 pipeline on an upload: neutralize, the three paths, and the threshold.
- Retire the old in-browser demo: `hosting/browser/`, `docs/hosting.md`, the README link. Vaibhav deletes the Space on Hugging Face if he wants it gone.
- Keep the Streamlit dashboard as the teaching and experiment tool, rename it, and update its pages to the P-001 pipeline.

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
- Make the neutralizer refuse a clip with less than 3.00 s of video, not only less than 3.00 s of sound (handbook mistake 67). Apply it under a new amendment, since it changes which clips are kept.
- Finish neutralizing LAV-DF's dev and train sets (`runs/p001-resume2.sh`, last step), for H3. It stopped in the 11-16 crash and resumes from its cut log.
- Fix the `data/legacy-team/ffpp` links, which point at folders deleted with `team/`.
- Speed up the visual and audio cache build. It is CPU-bound at about 340 clips an hour when shared.
- Clear the remaining lint findings, mostly in the slide helpers.
- Write the full paper for a venue (ICASSP or Interspeech 2027, or a CVPR or ICCV forensics workshop).
