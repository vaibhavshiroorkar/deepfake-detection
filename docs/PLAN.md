# PLAN

Last updated: 2026-10-24 (Session 1)

This plan lists the stages of the project, the question each stage must answer, the day-by-day schedule to the submission date and where each output is recorded. A stage is marked done only when its output exists in the repository and the decisions it needed are logged in DECISIONS.md.

## Stages

| Stage | Question | Output | Status |
|---|---|---|---|
| 0 | What exactly is being detected, from which inputs, and what counts as success | Problem statement, requirements | done: docs/REQUIREMENTS.md written 2026-10-24; the definition of comparable (D-025) is Proposed |
| 1 | Which data, which labels, how are train and test kept apart | Data plan, leakage rules | done for the datasets on the machine (manifests, splits, leakage check with content de-duplication, D-021 and D-026); builders for In-the-Wild, DFD and LAV-DF still to write |
| 2 | Metrics, splits, seeds, threshold policy, fixed before any model exists | Evaluation protocol and acceptance test | done: docs/PROTOCOL.md; acceptance A1 to A8 pass (V-001 and V-002) |
| 3 | The simplest end-to-end MVP | Baseline v0 with results | not started; the MVP form is decided (D-032: frozen pretrained features with a linear head) |
| 4 | Other models in the same harness, so the final choice has evidence | Comparison table with accuracy, cost, ease of fine-tuning | not started |
| 5 | Iterations 1 to 7 or more, each from a written hypothesis | One experiment note per iteration | not started |
| 6 | Whether and where remote heart-rate (rPPG) evidence enters | Feasibility test, then a decision | not started |
| 7 | Paper (IEEE, LaTeX), technical report, presentation | Documents assembled from the logs | scaffolds built and committed (D-014); no results |

## Deadline and constraints

- Submission date: 2026-10-30 (D-009, revised 2026-10-24). Today is day 1 of 7.
- Scope is not cut (D-016). The schedule is organised so that every iteration is complete by itself: hypothesis, run, result, decision, and the matching paragraph in the paper and slide. If work stops after any day, the story up to the last finished iteration is still valid. The "cut line" of each day states what is guaranteed by the end of that day if everything before it was completed.
- Machine (measured, ENV-DESKTOP-GPU): one RTX 5070 with 12 GB (about 11 GiB usable), Ryzen 9 7900X with 24 threads, 31 GB RAM, 404 GB free on D:.
- Run sizes must come from measurements. Where nothing is measured yet this plan says "to be measured" and a measurement is scheduled in the table below. Nothing in this plan pre-writes the outcome of any iteration.
- The last day is reserved for assembly, proofreading and rehearsal. No new experiments on the last day.
- Data preparation (video decoding, face cropping, audio extraction) is likely to be the real bottleneck and is not yet measured. Its measurement is scheduled for the morning of day 2, ahead of any model work.

## Measurements to schedule

| Measurement | Why | When | Status |
|---|---|---|---|
| Download throughput on this connection | Decides how many gigabytes can arrive per day and therefore the download order | Day 1 | first reading 4.09 MB/s on one connection from one server (EXPERIMENTS.md); to be re-measured during the first approved download |
| Decode, face crop and audio extraction throughput on a fixed sample of clips | Likely bottleneck; sets the size of every later experiment | Day 2 morning | to be measured |
| Peak VRAM, seconds per epoch and parameters of each candidate backbone at the batch size used | Cost axes in DECISIONS.md; fits runs inside 11 GiB | Day 3 (MVP) and Day 4 (candidates) | to be measured |
| Time of one full run (one seed) of the MVP | Sets how many seeds and iterations fit per day | Day 3 | to be measured |
| rPPG extraction cost per clip on CPU | Decides how many clips the Stage 6 test can use | Day 2 (small sample), Day 5 (test) | to be measured |
| Tiny-network training step time and VRAM (synthetic data) | Already measured: float32 134.5 ms, bfloat16 77.6 ms per step at batch 256; peak 5771 MiB and 3787 MiB | Done in Session 0 | measured (EXPERIMENTS.md, ENV-DESKTOP-GPU) |

## Day-by-day schedule

| Day | Date | Planned work | GPU / background jobs | Writing | Cut line (guaranteed if this day is completed) |
|---|---|---|---|---|---|
| 1 | Sat 2026-10-24 (Session 1, today) | Corrections and decisions D-011 to D-019 logged. Five parallel streams: templates and scaffold, dataset due diligence, three literature streams. Merge and spot-check their output. Data plan rebuilt from requirements (D-020, D-021): keep, drop or defer for every dataset, bandwidth budget, manual-download sheets. The user submits the gated request forms today and approves the list. | After approval: automatic batch of open data (about 52 GB, about 3.5 hours at the measured speed: audio, generated images, LAV-DF; LAV-DF needs the user's Hugging Face token); the user's gated downloads join the same bandwidth budget as they are approved, FF++ first. Re-measure download throughput on the first real download. No GPU jobs. | Paper, report and slide scaffolds built from the templates, placeholders only. | Documentation baseline and data acquisition started. No experiments. |
| 2 | Sun 2026-10-25 | Morning: measure data-preparation throughput; run the completeness checks (counts, decoding, SHA-256) on what has arrived. The gated datasets arrive by drive on day 3 (D-022), so day 2 uses the open data (audio, generated images and LAV-DF if it is there) to build and test the data-preparation and leakage-check code. Stage 0: problem statement and requirements. Stage 1: manifests, grouping keys, split files, leakage-check script that proves zero overlap (D-012). Stage 2: metric definitions, threshold policy, bootstrap settings, seeds, the `--final` guard, an acceptance test of the harness with a dummy scorer. Ask the user the Stage 2 questions (O-005). Write the MVP hypothesis and success and failure criteria (O-006) before any run. | Data preparation (decode, crop, extract audio) on CPU in the background. | Related Work refreshed from the merged literature; data and protocol sections written. | Protocol frozen before any model exists: a defensible evaluation with no results yet. |
| 3 | Mon 2026-10-26 | Gated data arrives by drive (D-022): copy it into data/, compare the file lists and sizes with the lists made on the source machine, then run the completeness checks. Stage 3: the audio MVP and the generated-image MVP (iteration I-0) on the open data during the day, each with at least 3 seeds under the frozen protocol; the video MVP trains on FF++ once it has passed the checks, and falls back to LAV-DF if FF++ is not on the machine by the time set in D-022. Stage 4 begins: candidate backbones from the shortlist (O-007) in the same harness, with the cost axes measured. | MVP runs; candidate runs sized from the MVP run time. | Method and Experiments set-up written; MVP result paragraph and slide from E-xxx. | I-0 complete with results, first comparison started. |
| 4 | Tue 2026-10-27 | Stage 4 finishes with the model decision written from evidence (O-007, O-008). Iterations I-1 to I-3, each from a written hypothesis that follows from the previous result. | GPU queue; run sizes from day 3 measurements. | One paragraph and one slide per finished iteration. | Story complete through I-3 and the backbone decision. |
| 5 | Wed 2026-10-28 | Iterations I-4 and I-5 (including fusion, O-009). Stage 6: rPPG feasibility test with success and failure thresholds written first (O-010). | GPU queue for iterations; rPPG extraction on CPU in parallel. | Paragraphs and slides for I-4, I-5 and the rPPG decision. | Story complete through I-5 and the rPPG adopt or not-adopt decision. |
| 6 | Thu 2026-10-29 | Iterations I-6 and I-7 if the time measured on earlier days allows. Freeze every iteration; the final test split is touched once per frozen iteration with `--final`. Software wrapper (O-011). Last experiment starts no later than midday, to be confirmed from measured run times. Complete drafts of paper, report and slides with all tables and figures taken from logged experiments. | Final runs; no new hypotheses after midday. | Full drafts. | All finished iterations frozen and tagged; complete drafts. |
| 7 | Fri 2026-10-30 | Assembly, proofreading, citation check against RESEARCH.md, consistency check that every number traces to an E-xxx ID, rehearsal, final commit and push. No new experiments. | None. | Final documents. | Submission set. |

If a day's work overruns, the next day's iterations shift; an iteration that has not finished by the end of day 6 is reported as not run, not as a result. The reading of the cut lines is: after day N, the paper, report and slides can be completed with only the evidence from iterations that are frozen.

## Iteration slots

Each slot receives its hypothesis, success criterion and failure criterion before its run, in docs/EXPERIMENTS.md. Outcomes are written only after the run. Nothing is pre-written here.

| Slot | Role | Hypothesis | Experiment ID | Outcome |
|---|---|---|---|---|
| I-0 | MVP | to be written before the run | none yet | none yet |
| I-1 | Iteration 1 | to be written before the run | none yet | none yet |
| I-2 | Iteration 2 | to be written before the run | none yet | none yet |
| I-3 | Iteration 3 | to be written before the run | none yet | none yet |
| I-4 | Iteration 4 | to be written before the run | none yet | none yet |
| I-5 | Iteration 5 | to be written before the run | none yet | none yet |
| I-6 | Iteration 6 | to be written before the run | none yet | none yet |
| I-7 | Iteration 7 | to be written before the run | none yet | none yet |
| I-8 | Spare iteration (use only if a hypothesis justifies it) | to be written before the run | none yet | none yet |

## Risks to the schedule

1. Seven calendar days for the MVP, at least 7 iterations, an rPPG test and three documents is tight. The cut lines make the loss of later iterations visible instead of silent.
2. Data preparation may be slower than model training. It is measured first on day 2.
3. Gated datasets (D-020) arrive on Monday 2026-10-26 by drive (D-022), which puts the video MVP on the evening of day 3 at the earliest. The fallback for the visual path is LAV-DF; the cross-dataset claim then rests on whatever has arrived (D-021). The sizes of some gated sets are not stated on their pages, so the budget uses estimates, and a drive copy can be corrupted, which the file-list comparison and the SHA-256 check are meant to catch.
4. GitHub: no remote is configured and the CLI is not logged in on this machine, so pushing is blocked until the user logs in (see STATUS.md).
5. The decisions O-007 to O-009 each need the user's input or an autonomous decision tagged for review (CLAUDE.md rule 10).
6. Neither FF++ nor Celeb-DF v2 has an audio stream (0 of 20 and 0 of 250 videos probed; the FF++ README says only the source video without audio was downloaded; measurement M-001 in EXPERIMENTS.md). DFD and the other datasets were not probed. The only data in the plan that carries audio inside video, with labels for it, is LAV-DF (downloading since 22:07 on 2026-10-24, 25.6 GB) and FakeAVCeleb (waiting for the owners' approval).
7. What the audio-visual claim would be if neither LAV-DF nor FakeAVCeleb is on the machine and verified by Tuesday 2026-10-27: the project could not train or evaluate any model on audio and video together, because no labelled data for it would exist here. The honest claim would then be: the visual path is evaluated on video frames (FF++ trained, Celeb-DF dev and test), the audio path on standalone speech (ASVspoof 2019 trained, In-the-Wild dev and test), and the interface can score the audio track of a video with the audio model and show the two scores side by side, but the combination of the two for a video is an untested demonstration with no measured accuracy, and no sentence in any document may claim audio-visual detection performance or a validated fusion rule (CLAUDE.md rule 8). If only LAV-DF arrives, audio-visual results are in-dataset only (its identity-independent test split) and are reported that way, with no cross-dataset audio-visual claim. If both arrive, the cross-dataset audio-visual test is LAV-DF training against FakeAVCeleb, after the identity-overlap check of D-012 (both take real videos from VoxCeleb2).
