# Pending work

Everything still open, in the order it will happen. A finished item moves to the Unreleased section of the [changelog](CHANGELOG.md).
The reasons behind each item are in the [project log](PROJECT-LOG.md). The rules for each test are in [P-001](../results/preregistration/P-001-shortcut-neutral-av-study.md).

## 1. Running now

## 2. Overnight 11-17 to 11-18: amendments A15 to A19, exploratory (Claude)

Chosen by Vaibhav on 11-17 at 20:20, to run while he sleeps. Restore point: tag `pre-overnight`. Stop rule: 18 November, 08:00; whatever has not finished is recorded as unfinished, not rushed. Each check is fixed in P-001 before its result exists, and H1 to H4 keep their verdicts. A15, A16, A17 and A19 are done (changelog). Their paper and report notes wait on branch `overnight-notes` for Vaibhav's review; main is unchanged.

- A18: H1 again with a fixed video bitrate instead of fixed quality. Running detached (`runs/p001-a18.sh`, log `runs/p001-a18.log`); audit expected around 01:00. Then its README, its paragraph on branch `overnight-notes`, and a rebuild.
- Then: the slide and poster notes on `overnight-notes`, rebuilt deliverables, and a morning summary for Vaibhav in the session chat.

## 2b. P-002 preparation, alongside (Claude)

Chosen 11-16: prepare the rebuild now, with no GPU and no change to anything H2 uses; build it after the submission. Design: [P-002 design](research/P-002-design.md).

- Freeze P-002 (`results/preregistration/P-002-draft.md`, now drafted) after the baselines are reproduced: set the bars of Q1 to Q6, then rename it to `P-002.md` and mark it frozen, before any P-002 model is trained.
- AV-Deepfake1M++: size (1.46 TB) and licence (gated EULA) are checked; the metadata format is still unknown. Decide on a stratified subset or an external disk, since this PC has about 519 GB free. Follow up on access.
- Confirm the assumed licences in `docs/research/P-002-backbones.md` (CLIP, XLS-R) before P-002 uses them.

## 2c. The 18th: documentation day, with Vaibhav (decided 11-17 in a side chat; the 19th is for study)

- Read the overnight results (A15 to A19) and decide whether their notes go into the paper, report and slides. The wording waits on branch `overnight-notes`, not on `main`.
- A deep wording pass on the paper, report and slide notes: rhythm, groups of three, restated points, build-ups. Words only, never claims or numbers; rebuild every deliverable after. The word scan is already clean apart from two "not only X" lines in the report.
- Comparison tables drafted in the side chat on 11-17: a protocol table (who trains on fakes, splits by person, cleans training and test data, controls false alarms, preregisters) and two score tables (raw files as papers report them, cleaned files under our protocol), each read against the file-cue ceiling. Every competitor cell was read from arXiv HTML pages and must be checked against the PDF before it goes into the report's literature chapter or a backup slide. Draft: the side chat's transcript.
- `docs/SECURITY.md`, and a check of `README.md`, `docs/FINAL.md` and the handbook for stale numbers and links.

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
- Frontend built 11-17 (see the changelog), spike, calibration, detector backend and result card done 11-17 (changelog). Still open, in the order of the [website design](superpowers/specs/2026-11-17-website-design.md): the admin carry-over (including a switch to turn the detector off), `docs/SECURITY.md`, the Docker image and the Hugging Face Space. Small: the drop zone's focus ring is clipped at the right edge at phone width.
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
