# PLAN

Last updated: 2026-10-25 (Session 3: I-1 to I-7 and the baseline have logged runs; the freeze is not run)

This plan lists the stages of the project, the question each stage must answer, the day-by-day schedule to the submission date and where each output is recorded. A stage is marked done only when its output exists in the repository and the decisions it needed are logged in DECISIONS.md.

## Stages

| Stage | Question | Output | Status |
|---|---|---|---|
| 0 | What exactly is being detected, from which inputs, and what counts as success | Problem statement, requirements | done: docs/REQUIREMENTS.md written 2026-10-24; the definition of comparable (D-025) is Proposed |
| 1 | Which data, which labels, how are train and test kept apart | Data plan, leakage rules | done for the datasets on the machine (manifests, splits, leakage check with content de-duplication, D-021 and D-026); builders for In-the-Wild, DFD and LAV-DF still to write |
| 2 | Metrics, splits, seeds, threshold policy, fixed before any model exists | Evaluation protocol and acceptance test | done: docs/PROTOCOL.md; acceptance A1 to A8 pass (V-001 and V-002) |
| 3 | The simplest end-to-end MVP | Baseline v0 with results | done on Sun 2026-10-25, ahead of the Mon 2026-10-26 plan: I-0 working version, checkpoint I-0-mvp, parity PASS, page checked on the real checkpoint, E-004, tag iter-0-mvp |
| 4 | Other models in the same harness, so the final choice has evidence | Comparison table with accuracy, cost, ease of fine-tuning | sweep E-001 to E-003 final (D-049); ease of fine-tuning measured in E-005 for the video task |
| 5 | Seven experimental iterations I-1 to I-7 after the MVP, each from a written hypothesis and the diagnostic battery; I-8 is integration and freeze | One experiment note per iteration; ladder in docs/ITERATIONS.md (Decided, D-041) | not started |
| 6 | Whether and where remote heart-rate (rPPG) evidence enters | Feasibility test, then a decision; it is iteration I-7 with its thresholds written in docs/ITERATIONS.md (frozen by D-041) | not started; the sanity check S-1 is planned in parallel with I-1 to I-3 |
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
| 2 | Sun 2026-10-25 | Sweep E-001 to E-003 (feature extraction, linear heads, ranking on dev_x under D-036); LAV-DF download and verification in the background; step times measured (D-042); the MVP code path (heads, temperature scaling, checkpoint metadata, library, web page on a mock adapter first); design of the rPPG sanity check S-1. | GPU queue for the sweep; LAV-DF job; CPU for S-1 extraction. | Notes for the sweep and the data log. | Sweep ranked, MVP code path complete. |
| 3 | Mon 2026-10-26 | I-0 MVP on all three tasks (FF++, DFD and Celeb-DF v2 are already on the machine; only FakeAVCeleb, WildDeepfake and Celeb-DF++ are still pending, and none is on the critical path): 3 seeds, intervals, the diagnostic battery, the web page on a real checkpoint (D-040). Provisional target 18:00, recomputed from measured speeds (STATUS.md). | GPU queue. | Paragraph and slide for I-0. | Working MVP (D-040) and diagnosis for I-1. |
| 4 | Tue 2026-10-27 | I-1 (video task first, budget D-042), I-2, I-3, each lever chosen from the previous diagnosis (ITERATIONS.md); S-1 runs on CPU. | GPU queue; CPU for S-1. | One paragraph and one slide per finished iteration. | Story complete through I-3. |
| 5 | Wed 2026-10-28 | I-4, I-5, I-6 (fusion needs LAV-DF extracted and verified, O-009); I-7 rPPG: S-1 result, then separation and added value on dev_x only if S-1 passes; adopt or do-not-adopt decision on dev_x. | GPU queue; rPPG on CPU. | Paragraphs and slides for I-4 to I-7. | Story complete through I-7. |
| 6 | Thu 2026-10-29 | I-8 integration and freeze, no new experiments: freeze code, preprocessing and checkpoints; tag each iteration; evaluate each frozen iteration once on the sealed test sets with `--final` and run the two confirmatory comparisons (PROTOCOL.md addendum); end-to-end demo with `inference_source` real; complete drafts with all numbers from logged experiments; delete the LAV-DF tar by hand (D-039). Experiments end on Wed night; an iteration not finished by then is reported as not run. | Final evaluations only. | Complete drafts. | Everything frozen and evaluated once. |
| 7 | Fri 2026-10-30 | Assembly, proofreading, citation check against RESEARCH.md, consistency check that every number traces to an E-xxx ID, rehearsal, final commit and push. No new experiments. | None. | Final documents. | Submission set. |

If a day's work overruns, the next day's iterations shift; an iteration that has not finished by the end of day 6 is reported as not run, not as a result. The reading of the cut lines is: after day N, the paper, report and slides can be completed with only the evidence from iterations that are frozen.

## Iteration slots

Each slot receives its hypothesis, success criterion and failure criterion before its run, in docs/EXPERIMENTS.md. Outcomes are written only after the run. Nothing is pre-written here.

| Slot | Role | Hypothesis | Experiment ID | Outcome |
|---|---|---|---|---|
| I-0 | MVP (D-040), done 2026-10-25 | E-001 to E-004 | E-001, E-002, E-003, E-004 | see EXPERIMENTS.md (exploratory, D-041) |
| I-1 | Fine-tuning depth on T-V (D-051, D-042), done 2026-10-25 | E-005 | E-005 | SUPPORTED (mechanical): IMG-17 fine-tuned, validation 0.9899, dev_x 0.9096 against the probe 0.7396 |
| I-2 | Speech preprocessing, trim and peak-normalise (D-054) | E-007 | E-007 | SUPPORTED: dev_x 0.6723, +0.0623 over I-0 |
| I-3 | Nuisance augmentation of speech training clips (D-055) | E-008 | E-008 | SUPPORTED: dev_x 0.7455, +0.0732 over the I-2 head |
| I-4 | Average of three speech encoders (D-056) | E-009 | E-009 | SUPPORTED: dev_x 0.8130, +0.0675 over the single head |
| I-5 | Frozen heads on LAV-DF, pre-check, then fusion (D-057) | E-010 | E-010 | NOT TESTED: the visual head is below chance on LAV-DF (0.3838) |
| I-6 | In-domain LAV-DF heads and fusion (D-058); visual head on produced videos only (D-059) | E-011, E-012 | E-011, E-012 | E-011 SUPPORTED mechanically (fused AUC 0.9935) but the post-hoc control shows a pipeline cue (visual 0.507 within produced videos); E-012 NOT SUPPORTED (0.517) |
| I-7 | rPPG (O-010), thresholds in ITERATIONS.md | E-006 | E-006 | S-1 FAILS (0.301 against 0.6): rPPG signal not recoverable from these data; nothing further run (D-052) |
| BASE | Xception re-run baseline of D-025 (D-060) | E-013 | E-013 | validation 0.9872, dev_x 0.8750; delta 0.0144 (D-061) |
| I-8 | Integration and freeze, no experiments | n/a | none | version-2 checkpoint built, 304 tests, GPU parity PASS (2026-10-25); sealed staircase and the two confirmatory comparisons NOT run (freeze Thu 2026-10-29) |

The slots follow the order in which levers were triggered (D-056 item 4): the lever labels of ITERATIONS.md were starting guesses and were replaced by the lever used when the diagnostic battery pointed elsewhere.

## Risks to the schedule

1. Seven calendar days for the MVP, at least 7 iterations, an rPPG test and three documents is tight. The cut lines make the loss of later iterations visible instead of silent.
2. Data preparation may be slower than model training. It is measured first on day 2.
3. Data arrival: FF++, DFD and Celeb-DF v2 are already on the machine and verified. Only FakeAVCeleb, WildDeepfake and Celeb-DF++ are pending; none is on the critical path. LAV-DF access works (2026-10-24) and its download is queued behind the encoder weights; the audio-visual iterations need it (fallback claim below).
4. GitHub: resolved (logged in, origin set, main pushed).
5. The decisions O-007 to O-009 each need the user's input or an autonomous decision tagged for review (CLAUDE.md rule 10).
6. Neither FF++ nor Celeb-DF v2 has an audio stream (0 of 20 and 0 of 250 videos probed; the FF++ README says only the source video without audio was downloaded; measurement M-001 in EXPERIMENTS.md). DFD and the other datasets were not probed. The only data in the plan that carries audio inside video, with labels for it, is LAV-DF (access granted 2026-10-24, download queued behind the encoder weights, 25.6 GB) and FakeAVCeleb (waiting for the owners' approval).
7. What the audio-visual claim would be if neither LAV-DF nor FakeAVCeleb is on the machine and verified by Tuesday 2026-10-27: the project could not train or evaluate any model on audio and video together, because no labelled data for it would exist here. The honest claim would then be: the visual path is evaluated on video frames (FF++ trained, Celeb-DF dev and test), the audio path on standalone speech (ASVspoof 2019 trained, In-the-Wild dev and test), and the interface can score the audio track of a video with the audio model and show the two scores side by side, but the combination of the two for a video is an untested demonstration with no measured accuracy, and no sentence in any document may claim audio-visual detection performance or a validated fusion rule (CLAUDE.md rule 8). Amendment 2026-10-25 (D-051): LAV-DF is on the machine, so in-dataset audio-visual evaluation is possible, but the audio head is near chance outside its training corpus (In-the-Wild dev_x AUC 0.61 [0.54, 0.73] with 12 speakers; trivial recording features reverse sign between ASVspoof and In-the-Wild, M-004). The branch of this fallback text that applies is therefore: the audio score of a video is shown beside the visual score with the warning AUDIO_CALIBRATION_EXPLORATORY, and no sentence claims audio-visual detection performance or a validated fusion rule until the I-6 pre-check (the frozen heads scored on LAV-DF validation, audio lower bound above 0.55) and the fusion are logged. If only LAV-DF arrives, audio-visual results are in-dataset only (its identity-independent test split) and are reported that way, with no cross-dataset audio-visual claim. If both arrive, the cross-dataset audio-visual test is LAV-DF training against FakeAVCeleb, after the identity-overlap check of D-012 (both take real videos from VoxCeleb2).
