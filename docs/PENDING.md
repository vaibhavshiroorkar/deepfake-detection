# Pending work

Everything still open, in the order it will happen. A finished item moves to the Unreleased section of the [changelog](CHANGELOG.md).
The reasons behind each item are in the [project log](PROJECT-LOG.md). The rules for each test are in [P-001](../results/preregistration/P-001-shortcut-neutral-av-study.md).

## 1. Running now

Two chains run as detached processes. Each resumes from its own log if stopped. Do not restart or shut down the PC while they run. The 2-day chain and the A10 rescoring are done.

| Chain | Log | What it does | Expected finish |
|---|---|---|---|
| Resume 2 | `runs/p001-resume2.log` | Visual and audio training, 3 seeds each (visual seed 17 done at 12:47) | Mon 11-16, evening |
| H2 | `runs/p001-h2.log` | Caches the three targets, scores them with each visual and audio model as it lands, then runs the fusion test (amendment A8) | Late Mon or early Tue |

## 2. While the chains run (Claude)

- Label the main claims in the handbook as verified, assumed, false or unknown (`FINAL.md` is done).
- Update handbook section 5.1b and the project log with the target, A9, A10 and H4 results.
- Run the full test suite once the CPU is free.

## 3. When H2 finishes (Claude)

- Copy `runs/p001/h2/` into `results/p001-h2/` with a README and receipts.
- Write the sentences that interpret H2 in the paper and the report (marked `% PENDING` in both), and the H2 speaker notes. The tables and the verdict fill in by themselves.
- Add the visual and audio results on the three targets to the report, if they are worth a table.
- Cut version 1.0.0 at the final-year submission: changelog heading, `pyproject.toml`, tag `v1.0.0`.
- Rename the package from `deepfake-generalization` (its old folder name) to `deepfake-detection` in `pyproject.toml` and `uv.lock`, and reinstall. This waits until no chain is running, because the reinstall rewrites the environment the chains use.
- Reorganize `scripts/` into `p001/` (current study), `data/` (fetching and manifests) and `legacy/` (the September to October pipeline). This waits until no chain is running, because the chains call scripts by path.

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

## 5. Dashboard and web app check (to schedule)

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
- Fix the `data/legacy-team/ffpp` links, which point at folders deleted with `team/`.
- Speed up the visual and audio cache build. It is CPU-bound at about 340 clips an hour when shared.
- Clear the remaining lint findings, mostly in the slide helpers.
- Write the full paper for a venue (ICASSP or Interspeech 2027, or a CVPR or ICCV forensics workshop).
