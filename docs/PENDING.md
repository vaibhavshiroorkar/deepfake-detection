# Pending work

Everything still open, in the order it will happen. A finished item moves to the Unreleased section of the [changelog](CHANGELOG.md).
The reasons behind each item are in the [project log](PROJECT-LOG.md). The rules for each test are in [P-001](../results/preregistration/P-001-shortcut-neutral-av-study.md).

## 1. Running now

Three chains run as detached processes. Each resumes from its own log if stopped. Do not restart or shut down the PC while they run.

| Chain | Log | What it does | Expected finish |
|---|---|---|---|
| 2-day | `runs/p001-2day.log` | AV-HuBERT features for HiFi-AVDF, LAV-DF test and DFDC; frozen sync scores; H4; audits of the neutral copies | Mon 11-16, about 09:00 |
| Resume 2 | `runs/p001-resume2.log` | FakeAVCeleb visual and audio cache, then visual and audio training, 3 seeds each | Cache Mon about 17:00; training overnight |
| H2 | `runs/p001-h2.log` | Waits for the two above; caches the three targets, scores them with the frozen visual and audio models, then runs the fusion test (amendment A8) | Tue 11-17, about midday |

## 2. When the 2-day chain finishes (Claude)

- Raise the resume 2 chain from idle to normal priority.
- Run `runs/p001-finalize.sh`: copy the target and H4 outputs into `results/p001-sync-targets/` and `results/p001-h4/`, then rebuild `results/numbers.json`, the figures, the slides, the poster and the Overleaf zips.
- Add a README with receipts to each new results folder.
- Commit the audits of the neutral copies under `results/shortcut-audit/*-neutral/`.
- Report the 23 HiFi-AVDF clips that have no audio track (14 real, 9 fake), and the clips with no face, in the results README and the paper.
- Write the paper and report text that interprets the cross-corpus and H4 results. The numbers fill in by themselves; the reading of them does not.
- Fill the pending entries in [FINAL.md](FINAL.md); update handbook section 5.1b and the project log.
- Run the full test suite once the CPU is free.

## 3. When H2 finishes (Claude)

- Copy `runs/p001/h2/` into `results/p001-h2/` with a README and receipts.
- Add the H2 keys to `scripts/build_numbers.py` and an H2 figure to `scripts/make_figures.py`.
- Add H2 and the in-domain visual and audio results to the paper, report, slides and poster. Fusion and modality prediction are listed as "not run" in all four today, so those lines change too.
- Cut version 1.0.0 at the final-year submission: changelog heading, `pyproject.toml`, tag `v1.0.0`.
- Rename the package from `deepfake-generalization` (its old folder name) to `deepfake-detection` in `pyproject.toml` and `uv.lock`, and reinstall. This waits until no chain is running, because the reinstall rewrites the environment the chains use.
- Reorganize `scripts/` into `p001/` (current study), `data/` (fetching and manifests) and `legacy/` (the September to October pipeline). This waits until no chain is running, because the chains call scripts by path.

## 4. Vaibhav

- Ask the guide what "Batch" means on the report. It is set to 2023-27; the academic year is 2026-27.
- Upload `deliverables/build/overleaf/report.zip` and `paper.zip` to Overleaf, compile with pdfLaTeX, and send any error.
- Read the paper and the slides and challenge every claim. Practise the talk once with the speaker notes.
- Decide whether to delete `mlartifacts/`, `mlruns/` and `mlflow.db` (6.8 GB of September to October tracking records) and `dist/` (an old package build and the October `final-report.pdf`).
- Send the DeepSpeak reminder around 11-22 if there is no answer. Forward any reply from AV-Deepfake1M, AV-Deepfake1M++ or KoDF.

## 5. Dashboard and web app check (to schedule)

- Decide whether the dashboard and web app (about 11,000 lines, 40% of `src/`) stay as the demo.
- If they go: delete them with the docs only they read: `docs/research/findings.md`, `docs/research/result-traceability.md`, `docs/handbook/01` to `10`, `docs/obstacles.md`, `docs/model-selection.md`, `docs/handoff.md` and its `ddf handoff update` command, `scripts/update_result_registry.py`, and the checks and tests that cover them.
- If they stay: update their content to the P-001 study. Today they cite the older findings.

## 6. Full run, after the deadline

- H3: predict which modality was faked, with reverse models trained on LAV-DF (amendment A2).
- AVH-Align and HAVIC baselines on the same protocol.
- The StreamFusion MLP, the secondary fusion that A8 dropped for the 3-day window.
- AV-Deepfake1M, AV-Deepfake1M++, KoDF and DeepSpeak v2, if access arrives. They are scored by frozen models only.
- Measure speaker overlap between FakeAVCeleb, LAV-DF and AV-HuBERT's pretraining data (VoxCeleb2, LRS3).
- H1 residual: video bitrate still carries the label after neutralization. Test a fixed-bitrate encode under a new amendment.
- Fix the `data/legacy-team/ffpp` links, which point at folders deleted with `team/`.
- Speed up the visual and audio cache build. It is CPU-bound at about 340 clips an hour when shared.
- Clear the remaining lint findings, mostly in the slide helpers.
- Write the full paper for a venue (ICASSP or Interspeech 2027, or a CVPR or ICCV forensics workshop).
