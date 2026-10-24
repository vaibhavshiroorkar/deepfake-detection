# DECISIONS

Last updated: 2026-10-24 (Session 1)

This document records every decision taken in the project, why it was taken, what evidence supported it and what could prove it wrong. It is written to be read by a reviewer with a strong mathematical background, so claims are tied to stated criteria and to measurable evidence. A decision without evidence is labelled as such.

## How decisions are made

### Entry template

Every decision uses this template. Fields are never left out; "none yet" or "not recorded" is written where there is nothing to say.

```
### D-NNN: <short title>
- Date: YYYY-MM-DD
- Status: <one of the vocabulary below>
- Question: <what had to be decided>
- Options considered: <list, including "build our own" where relevant>
- Criteria: <what makes one option better, stated before looking at results>
- Evidence: <experiment IDs, paper IDs from RESEARCH.md, or "none yet">
- Decision and reason: <text>
- Rejected and why: <text>
- Consequences / what could prove this wrong: <text>
```

### Status vocabulary

| Status | Meaning |
|---|---|
| Decided (user) | The user made this decision. It changes only when the user says so. |
| Decided autonomously - needs user review | The implementer made this decision because it was small or blocking. The user reviews it at the next session. |
| Proposed | The implementer recommends an option with evidence; nobody has accepted it yet. |
| Open | The question is known and has not been answered. Options and criteria may be listed; no choice has been made. |
| Superseded | A later decision replaced this one. The text stays. |

### Superseded text is kept word for word

When a decision is replaced, the old text is not edited or deleted. It stays in the entry (or in the old entry, whose status changes to `Superseded by D-NNN`) and the new text cites it. This keeps the history auditable: a reader can see what was believed at each date and why it changed. The one case so far is D-009.

### Comparison rules

Whenever two or more candidates (models, losses, fine-tuning methods, fusion schemes, features) are compared, all of the following hold. A comparison that breaks any rule is not used as evidence for a decision.

1. Same data, same split, same compute budget and same metric for every candidate. The budget is stated (epochs, steps or wall-clock) before the runs.
2. At least 3 random seeds per candidate. Seeds are listed in the experiment record.
3. Report the mean, the standard deviation over seeds, and a bootstrap confidence interval of the metric on the held-out data. Resampling is done over independent units, which means identities or source videos and not individual frames, because frames of one video are not independent. The number of resamples and the interval level are fixed in Stage 2 before any model exists.
4. Candidates are compared in a paired way: on the same test units, using the difference of the metric per resample, not by comparing two separate intervals by eye.
5. If the intervals of the difference include zero, the evidence does not separate the candidates. The simpler or cheaper candidate is preferred and this is stated in the decision entry.
6. Reported numbers always name their protocol. In-dataset numbers and cross-dataset numbers are never mixed in one table row or one claim.

### Cost axes

For every candidate model the following are measured under the same conditions, on the machine recorded in ENV-DESKTOP-GPU, and reported next to accuracy:

| Axis | How it is measured |
|---|---|
| Parameters | Count of trainable and total parameters. |
| FLOPs | Forward-pass floating point operations for one input of the size actually used. |
| Peak VRAM | Maximum allocated GPU memory during a training step at the batch size used. |
| Seconds per epoch | Wall-clock time per epoch, including data loading. |
| Epochs to a fixed validation target | Number of epochs needed to reach a validation metric value fixed in advance. |
| Sensitivity to learning rate | Change of the validation metric when the learning rate is varied by a fixed factor in each direction. |
| Ease of fine-tuning | Number of hyperparameters that had to be tuned, stability across seeds (standard deviation), and whether the default recipe worked without adjustment. |

### A frozen choice is never reopened using test-set results

Once a choice is frozen (a model, a threshold policy, a split), it can be reopened only on the basis of validation data or newly stated evidence about the problem, never because of how it performed on the held-out test set. The test set is used to report the result of the frozen choice. Looking at test results and then changing the choice turns the test set into a validation set and makes the reported numbers optimistic. If a test result is surprising, the surprise is written in the experiment record and a new, separately labelled held-out set is needed to check any resulting change.

## Decisions taken by the user on 2026-10-24 (D-001 to D-010)

These ten were decided by the user on 2026-10-24 and were first given to the implementer in the Session 0 prompt. In Session 1 the user stated the reasons below. Only the reasons the user stated are recorded; where the user gave none, the field says "not recorded".

### D-001: Build the whole system from zero in this repository
- Date: 2026-10-24
- Status: Decided (user)
- Question: Whether to reuse earlier work or start again.
- Options considered: Reuse earlier code or text; build from zero.
- Criteria: Not recorded.
- Evidence: none yet
- Decision and reason: Build the whole system from zero in this repository. No code or text is carried over from earlier work. Reason (stated by the user): clarity about what exists and where, and documentation throughout, so the development path is traceable from the first iteration.
- Rejected and why: Reuse of earlier work. Reason beyond the above not recorded.
- Consequences / what could prove this wrong: All components must be built and verified here, which costs time. See D-014 for how the supplied document templates are used under this rule (identity details, logos, colours and layout only).

### D-002: Input scope is images, videos and audio
- Date: 2026-10-24
- Status: Decided (user) for the scope; how each input type is handled is Open (O-002). Refined by D-018 and D-019.
- Question: Which input types the system must accept.
- Options considered: Images only; images and video; images, video and audio.
- Criteria: Not recorded.
- Evidence: none yet
- Decision and reason: The scope is images, videos and audio. Reason (stated by the user): analysing video needs its audio. How image, video and audio inputs are each handled is not decided; it is logged as O-002.
- Rejected and why: Narrower scopes. Reason not recorded.
- Consequences / what could prove this wrong: A wide scope divides the effort between modalities. Datasets with audio labels are fewer and smaller than visual ones (see RESEARCH.md, R-013 and R-014), which could limit what claims the audio branch can support.

### D-003: Deliverables
- Date: 2026-10-24
- Status: Decided (user)
- Question: What the project must produce.
- Options considered: Not recorded.
- Criteria: Not recorded.
- Evidence: none yet
- Decision and reason: The deliverables are the software, an IEEE-format research paper (LaTeX, written in Overleaf), a technical report and a presentation, all produced in this repository. Reason (stated by the user): this is a college project that needs a research paper, a technical report and a presentation.
- Rejected and why: Not recorded.
- Consequences / what could prove this wrong: The paper and report are assembled from the living documents, so the logs must be complete and honest from the first session.

### D-004: Method is MVP first, then at least 7 hypothesis-driven iterations
- Date: 2026-10-24
- Status: Decided (user)
- Question: How the work is organised.
- Options considered: Not recorded.
- Criteria: Not recorded.
- Evidence: none yet
- Decision and reason: Build the simplest MVP first, then at least 7 further iterations. Each iteration starts from a written hypothesis and ends with a measured result, including failures. Candidate models are run in the same pipeline from the MVP stage so that the final choice has supporting evidence. Reason (stated by the user): a strong system is not built in one go; the aim is to show how it improves. The model choice must be defensible against "why this model and not others", so comparison evidence is gathered early.
- Rejected and why: Not recorded.
- Consequences / what could prove this wrong: The comparison rules above must hold from the MVP onward, so the shared pipeline and evaluation protocol (Stages 1 and 2) must exist before the MVP.

### D-005: Goal is comparability on our own protocol, not the best model
- Date: 2026-10-24
- Status: Decided (user)
- Question: What counts as success.
- Options considered: Best-in-world performance; documented path to results comparable with published detectors.
- Criteria: Not recorded.
- Evidence: none yet
- Decision and reason: Show a documented path to results comparable with published detectors on a protocol we define and run ourselves. The goal is not the best model in the world. Reason (stated by the user): the aim is to show improvement and comparable results, not the best model.
- Rejected and why: Best-in-world performance as a goal. Reason beyond the above not recorded.
- Consequences / what could prove this wrong: Published numbers come from different protocols and are not directly comparable (RESEARCH.md shows this repeatedly). "Comparable" therefore has to be defined precisely in Stage 0 and 2, for example by re-running a published detector on our protocol rather than quoting its paper's number.

### D-006: rPPG is a candidate improvement, not an adopted component
- Date: 2026-10-24
- Status: Decided (user)
- Question: Whether remote photoplethysmography (rPPG, the heartbeat-in-skin-colour idea behind Intel FakeCatcher) becomes part of the system.
- Options considered: Adopt; do not adopt; evaluate first.
- Criteria: Adoption depends on a feasibility test in Stage 6.
- Evidence: Literature is mixed (R-005 to R-010). Older work reports high accuracy on 2019-era data; 2025 and 2026 papers report that modern face-swap fakes can carry a plausible pulse (R-008) and that an older rPPG detector degrades from AUC 0.999 to 0.622 on newer talking-face fakes in one reproduction (R-009).
- Decision and reason: rPPG will be evaluated as a candidate improvement. It is not adopted. Adoption depends on the Stage 6 feasibility test. Reason (stated by the user): the user expects that a heartbeat-in-skin-colour signal may raise accuracy. This is an untested expectation, which is why it is an experiment and not an adoption.
- Rejected and why: Immediate adoption, because the expectation is untested (reason stated by the user).
- Consequences / what could prove this wrong: If the feasibility test fails, rPPG is reported as a negative result and not included. See O-010.

### D-007: Hardware is local and fine-tuning is local
- Date: 2026-10-24
- Status: Decided (user)
- Question: Where training runs.
- Options considered: Local GPU; rented cloud GPU.
- Criteria: Not recorded.
- Evidence: ENV-DESKTOP-GPU in EXPERIMENTS.md (measured 2026-10-24).
- Decision and reason: Hardware is local; fine-tuning is done locally; plans must fit inside the measured VRAM. Measured: one NVIDIA GeForce RTX 5070 with 12 GB (about 11 GiB usable while the desktop is running). Reason (stated by the user): the machine can fine-tune models locally.
- Rejected and why: Cloud GPUs. Reason not recorded.
- Consequences / what could prove this wrong: Large backbones or long video clips may not fit. Mixed precision reduced peak memory by about a third in the sizing probe, which helps, but each candidate must be measured (cost axes).

### D-008: Hosting is GitHub with small commits and a tag per frozen iteration
- Date: 2026-10-24
- Status: Decided (user)
- Question: Where the code and documents are hosted and how history is kept.
- Options considered: Not recorded.
- Criteria: Not recorded.
- Evidence: none yet
- Decision and reason: Hosting is GitHub, committed in small steps, tagged per frozen iteration. Reason (stated by the user): keeps the history of the work visible in small commits.
- Rejected and why: Not recorded.
- Consequences / what could prove this wrong: At the end of Session 0 no git remote existed and the GitHub CLI was not logged in. At the start of Session 1 the CLI still reported no login on this machine, so nothing has been pushed (see STATUS.md blockers).

### D-009: Timeline (revised in Session 1)
- Date: 2026-10-24
- Status: Decided (user). The wording below under "Superseded text" is superseded by the current text on the same date; the old text is kept word for word.
- Question: How long the project may take.
- Options considered: Not recorded.
- Criteria: Not recorded.
- Evidence: none yet
- Decision and reason (current): The final submission date is 30 October 2026. Reason (stated by the user): this is the final date. The work is therefore planned day by day from 2026-10-24 to 2026-10-30 (see PLAN.md).
- Rejected and why: The open-ended timeline in the superseded text.
- Consequences / what could prove this wrong: Seven calendar days, including today, for the MVP, at least 7 iterations, an rPPG feasibility test, and three written deliverables. See D-016: scope is not cut; the schedule makes each iteration complete by itself so that the story up to the last finished iteration remains valid.

> Superseded text (kept word for word from the 2026-10-24 Session 0 entry):
> - Status: Decided (user) for the principle; exact dates Open
> - Decision and reason: More than two weeks are available and the work should finish as soon as is sound. Exact dates are open.
> - Consequences / what could prove this wrong: Without dates, scope has to be controlled by the stage gates in PLAN.md. The user should supply a submission date so that the plan can be checked against it.

### D-010: Documentation method
- Date: 2026-10-24
- Status: Decided (user)
- Question: How progress, decisions and evidence are recorded.
- Options considered: Not recorded.
- Criteria: Not recorded.
- Evidence: none yet
- Decision and reason: Use the living-document set listed in CLAUDE.md. CLAUDE.md was not present in the repository during Session 0, so that session used the eight documents named in the Session 0 prompt. CLAUDE.md was added to the repository at the start of Session 1; its set (section 3) matches those eight plus `paper/`, `report/` and `slides/`. Reason (stated by the user): keeps work resumable between sessions and every choice traceable.
- Rejected and why: Not recorded.
- Consequences / what could prove this wrong: The commit message format in CLAUDE.md (`type(scope): summary`) differs from the provisional format used for the two Session 0 commits; history was not rewritten (see STATUS.md, Mistakes and corrections).

## Decisions taken in Session 1 (2026-10-24)

### D-011: Output of the system
- Date: 2026-10-24
- Status: Decided (user)
- Question: What the system outputs for an input (formerly O-001).
- Options considered: (a) one score only; (b) a clip-level real/fake probability, plus a separate score for each modality present, plus a fused score when more than one modality is present; (c) option (b) plus temporal localisation.
- Criteria: The claim must match what the training data supports; localisation needs temporal labels that not every dataset has.
- Evidence: R-014 (FakeAVCeleb separates real and fake audio and video), R-015 (localisation is possible with the audio-visual dissonance method but requires segment-level handling). No experiment yet.
- Decision and reason: Option (b), confirmed by the user in Session 1. A still image has only a visual score; a standalone audio file has only an audio score; a video with audio has visual, audio and fused scores. No temporal localisation in the first version.
- Rejected and why: (a) discards per-modality information that the scope (D-002, D-018, D-019) needs. (c) is left out of the first version because temporal labels are not available in every dataset.
- Consequences / what could prove this wrong: Per-modality scores need modality-specific labels, which only some datasets provide (to be recorded in DATA_LOG.md). The fused score's definition is still open (O-009).

### D-012: Split and evaluation policy
- Date: 2026-10-24
- Status: Decided (user) as an outline. The script and the dataset-specific details are still to be written (remaining part of O-004).
- Question: How data is split and how results are reported so that generalisation is measured honestly.
- Options considered: Random frame split (known to leak); video-level split; source-disjoint and identity-disjoint split; generator-disjoint; dataset-disjoint (see O-004 list).
- Criteria: No person or source video appears on both sides of any split; the headline number must measure generalisation to data never used in training.
- Evidence: R-017 (DeepfakeBench): for the same Xception detector trained on FF++ c23, the mean AUC over the six FF++ within-domain tests is 0.945 and over the eight cross-domain tests is 0.772. R-009 and R-007 use identity-disjoint splits.
- Decision and reason: Group every sample by source (original video and, where metadata allows, person identity) and keep each group on one side of any split. Test on datasets never used in training as the headline generalisation result. Report in-dataset numbers but never as the headline. The final test split is touched once per frozen iteration, guarded by the `--final` flag required in CLAUDE.md rule 3. A script proves zero overlap between the splits. Confirmed by the user in Session 1.
- Rejected and why: Random frame-level splits, because frames of one video are not independent and leak identity and source.
- Consequences / what could prove this wrong: Some datasets give no identity metadata; for those the grouping falls back to source video and this limitation is recorded in DATA_LOG.md (`not established`). A dataset-disjoint headline needs at least two datasets of the same kind of content, which depends on D-013.

### D-013: Data access policy
- Date: 2026-10-24
- Status: Decided (user)
- Question: Which datasets may be considered, given the time available.
- Options considered: Include datasets that need an application; use only datasets obtainable without approval.
- Criteria: Time.
- Evidence: Stream B due diligence (DATA_LOG.md) once merged.
- Decision and reason: Use only datasets that can be obtained without applying for approval. Datasets that need a request form, a signed agreement or manual approval are excluded. Reason (stated by the user): time. Large downloads may run in the background all day; that is acceptable. The limit is 404 GB free on drive D:.
- Rejected and why: Approval-gated datasets, for time.
- Consequences / what could prove this wrong: If too few open talking-head video datasets remain for a dataset-disjoint test (D-012), the user will be asked what to do. This was the case to watch in the Session 1 dataset review.

### D-014: Writing starts now, from the supplied templates
- Date: 2026-10-24
- Status: Decided (user)
- Question: When the written deliverables start and what they are built from.
- Options considered: Write at the end from the logs; write continuously from now.
- Criteria: Not recorded.
- Evidence: none yet
- Decision and reason: Writing starts now. The paper, the technical report and the presentation are built from the templates in the folder `Document Template` and grow with every milestone. Clarification stated by the user in Session 1: the supplied PowerPoint and PDF are previous versions of earlier project documents; they are used only for their look and identity, not their content. The logos, colours, bars, layout and all identity details (college, department, branch, guide, names, roll numbers) in them are correct and are to be reused. No body text, figures or results from them are carried over (this is consistent with D-001).
- Rejected and why: Writing only at the end.
- Consequences / what could prove this wrong: Early drafts contain placeholders only; every number that later enters a document must come from a logged experiment, with its E-xxx ID traceable.

### D-015: Working method: parallel subagents under rules
- Date: 2026-10-24
- Status: Decided (user)
- Question: Whether independent work is done in parallel.
- Options considered: Sequential work by one agent; parallel subagents.
- Criteria: Not recorded.
- Evidence: none yet
- Decision and reason: Independent tasks run in parallel with subagents. Each subagent gets a self-contained brief with the hard rules from CLAUDE.md, writes only to its own file under `docs/_agents/<stream>.md` (the template stream also writes its own `paper/`, `report/`, `slides/` and `assets/branding/` files), and does not edit shared documents. Only the main agent merges, spot-checks and commits. Every agent output is spot-checked before merging; anything not re-verified is marked `NR`. No agent starts a GPU job or a large download.
- Rejected and why: Sequential work, for time.
- Consequences / what could prove this wrong: Merge and verification effort falls on the main agent. Agent errors that survive the spot-check would enter the shared documents.

### D-016: The scope is not cut for time
- Date: 2026-10-24
- Status: Decided (user)
- Question: Whether scope shrinks to fit the deadline.
- Options considered: Cut scope; keep scope and organise the schedule.
- Criteria: Not recorded.
- Evidence: none yet
- Decision and reason: The scope is not cut for time. The schedule is organised so that each iteration is complete by itself: hypothesis, run, result, decision, and the matching paragraph in the paper and slide. If time runs out, the story up to the last finished iteration is still valid.
- Rejected and why: Cutting scope in advance.
- Consequences / what could prove this wrong: With 7 calendar days the plan may not reach all iterations; the cut line for each day is stated in PLAN.md so the guaranteed story is explicit.

### D-017: Environment: Python 3.13 with PyTorch 2.14.1+cu130
- Date: 2026-10-24
- Status: Decided (user)
- Question: Which Python and PyTorch build the project uses (provisional choice from Session 0).
- Options considered: System Python 3.14.0; uv-managed Python 3.13.15. PyTorch cu126, cu130 and cu132 builds were available at 2.14.1 (cu128 stopped at 2.11.0).
- Criteria: Support in machine-learning libraries; native kernels for the RTX 5070 (sm_120); the driver's reported CUDA version (13.2).
- Evidence: ENV-DESKTOP-GPU in EXPERIMENTS.md: the build ships `sm_120` kernels and matrix multiplication and mixed precision ran correctly.
- Decision and reason: Python 3.13.15 with PyTorch 2.14.1+cu130, pinned in `uv.lock`. Reason: library support (Python 3.14 is newer than many machine-learning libraries support). Accepted by the user in Session 1.
- Rejected and why: Python 3.14 (less library support). PyTorch cu126 and cu132 were not needed.
- Consequences / what could prove this wrong: A library needed later may not support Python 3.13 or this PyTorch; this would be recorded as a new decision.

### D-018: Image scope covers both face-manipulated and fully generated images
- Date: 2026-10-24
- Status: Decided (user)
- Question: What "image" means in scope (part of O-002).
- Options considered: (a) face-manipulated images only; (b) fully generated images only; (c) both.
- Criteria: Not recorded beyond the trade-off shown to the user: the widest claim needs one or two extra generated-image datasets.
- Evidence: R-011 (generated-image detection across GANs and diffusion models, trained on one generator and tested on others).
- Decision and reason: Both. The two kinds of image are reported as separate test conditions, because a detector trained on face-swap video makes no claim about fully generated images without evidence (O-002 criterion).
- Rejected and why: Single-kind scopes, because the user chose the wider scope.
- Consequences / what could prove this wrong: Needs open (D-013) generated-image datasets in addition to face-video datasets; their licences and sizes are recorded in DATA_LOG.md.

### D-019: Audio has a separate standalone mode
- Date: 2026-10-24
- Status: Decided (user)
- Question: Whether audio is analysed only inside videos or also on its own (part of O-002).
- Options considered: (a) a separate audio-only mode with its own speech-spoofing data; (b) audio only as the soundtrack of videos.
- Criteria: Not recorded.
- Evidence: R-012 and R-013 (ASVspoof 2019 and AASIST) as a reference for standalone speech anti-spoofing; R-014 for audio inside video.
- Decision and reason: Option (a): audio files can be scored on their own using speech-spoofing data, and audio is also used inside videos.
- Rejected and why: Option (b), because the user chose the separate mode.
- Consequences / what could prove this wrong: Adds a second audio data source and a standalone audio evaluation to the plan; the audio-in-video and standalone audio results are reported as separate protocols.

## Open decisions

Settled by Session 1 decisions: O-001 (D-011). O-004 is settled as an outline by D-012 (script and details pending). O-002 is partly answered by D-018 and D-019 (what the modalities are); how each input type is handled remains open. O-003 is constrained by D-013. None of the questions below is otherwise answered. For each one, the options are those the implementer would compare, the criteria are stated before any result exists, and the evidence column says what must be collected. Paper IDs refer to RESEARCH.md. Pointers to papers show where options come from; they are not choices.

| ID | Question | Options to compare | Criteria (stated before results) | Evidence needed |
|---|---|---|---|---|
| O-002 | How image-only, video and audio inputs are each handled, and which claims each supports (the scope is fixed by D-018 and D-019) | Images: face crop then classifier (as R-001) versus whole-image detector (as R-011). Video: per-frame scores aggregated over time versus a clip-level temporal model. Audio: waveform model (R-012) versus spectral-feature model. Cross-modal: consistency between audio and lips (R-015). | A claim is made only for the kind of input the training and test data cover. Each path must have a held-out test with its own protocol. | The generators present in each dataset; cross-dataset results from the literature (R-004, R-011, R-017) showing how fragile transfer is. |
| O-003 | Datasets to use, and which are for training, validation and held-out testing (restricted to open-access datasets by D-013) | Visual: FaceForensics++ (R-001), Celeb-DF (R-002), DFDC (R-003), and others on the reading list. Audio: ASVspoof 2019 LA (R-013) and newer editions. Audio-visual: FakeAVCeleb (R-014, which its paper says is gated), DFDC. Generated images: those used in R-011 and others. | Open access without approval (D-013); size fits the 404 GB free on D:; identities or source videos available so that splits can be disjoint (D-012); labels match D-011; generators diverse and not all outdated; results comparable with published work. | Stream B facts in DATA_LOG.md: access route, licence, size, counts, identity metadata, official splits. |
| O-004 (remaining part) | Leakage script and dataset-specific split details | Per-dataset grouping keys (identity, source video, recording); automatic overlap check; storing each split as a file with an ID and hash. | The script proves zero overlap; where identity metadata is missing the fallback is recorded as `not established`. | Identity and source metadata of each chosen dataset. |
| O-005 | Metrics, threshold policy, calibration | Ranking metrics: ROC-AUC, average precision, EER. Threshold metrics: accuracy at a fixed threshold, balanced accuracy. Probability metrics: log loss, Brier score, ECE (R-016). Audio: min t-DCF with EER (R-012, R-013). Threshold policy: fixed 0.5; threshold chosen on validation data; EER threshold on validation. Calibration: none; temperature scaling (R-016). Aggregation: frame-level versus video-level. | The threshold is chosen only on validation data and never on test data (R-011 shows that tuning the threshold on the test set gives an "oracle" number that is not achievable in practice). Class balance in the test set is stated. A metric must be defined mathematically in the document before use. | Class ratios of the chosen datasets; seed-to-seed variance, which can be large (R-012 reports a spread of EER between 1.19 and 2.06 percent over seeds for one baseline); how the DFDC organisers used log loss (R-003). |
| O-006 | MVP design: the simplest thing that runs end to end | (a) Small CNN from scratch on face crops, one dataset. (b) Frozen pretrained features with a linear classifier (R-011 style). (c) Fine-tuned pretrained backbone. (d) Audio: small CNN on a spectrogram. | It runs from raw input file to a score and a metric under the Stage 2 protocol; it needs no tuning to produce a number; it trains within minutes on the measured machine; it is simple enough to explain. | The Stage 2 protocol; measured data-loading speed (not yet measured); a first sizing of each candidate. |
| O-007 | Candidate backbones per modality, including a small network built from scratch as a reference | Visual: small CNN from scratch; ResNet family; Xception and EfficientNet-B4 (both used as strong plain baselines in R-017); frozen CLIP-type encoder with a linear or nearest-neighbour classifier (R-011). Audio: small CNN on a spectrogram; a lightweight model such as AASIST-L (85K parameters, R-012). | The cost axes in this document, measured on one machine under the comparison rules. | Experiments in Stage 4. Prior evidence: in one benchmark a plain Xception was within roughly one to two AUC points of more elaborate detectors under identical settings (R-017), which suggests the backbone and training recipe may matter more than the method name. |
| O-008 | Fine-tuning strategy | Linear probe; full fine-tuning; partial unfreezing of the last blocks; differential learning rates; low-rank adapters (not yet read). | Validation metric at a fixed budget; peak VRAM; stability across seeds; sensitivity to learning rate. | Stage 4 and 5 runs. Prior evidence: ImageNet accuracy correlates with transfer accuracy for both fixed features and fine-tuning across 12 natural-image tasks (R-018), but deepfake data is not natural-image classification, so this must be tested here. |
| O-009 | Fusion strategy (how the fused score of D-011 is formed) | Late fusion of per-modality scores (average or logistic regression); feature-level fusion; cross-modal consistency model (R-015); joint training (R-020, to be read). | Must handle inputs with a missing modality (a still image has no audio, D-011); must be at least as good as the best single modality on the validation data; must be explainable. | Per-modality baselines first. Prior evidence: simple ensembles of unimodal detectors did poorly on FakeAVCeleb in the dataset paper (R-014), which is a reason to test fusion carefully. |
| O-010 | rPPG feasibility test design | (a) Signal sanity: can heart rate be recovered from our data at all, given that most deepfake datasets have no pulse ground truth (R-008). (b) Separation: do rPPG features separate real from fake under an identity-disjoint split, for several generator families (R-009, R-007). (c) Added value: does adding rPPG evidence raise the fused detector's held-out metric beyond the bootstrap interval. (d) Negative controls: time-shuffled or mismatched signals. | The success and failure thresholds are written before the run. rPPG is adopted only if (c) shows a gain larger than the interval of the paired difference. | Stage 6 experiments. Literature: R-005 to R-010 and Stream C3. A dataset subset with enough frames, resolution and lighting stability for rPPG; skin-tone coverage (R-009 notes that rPPG is sensitive to skin tone and lighting). |
| O-011 | Software form | Command line; local web app; both. | Time cost; need to demonstrate to examiners; ease of reproducing results; input handling for image, video and audio files. | Requirements from Stage 0. |

## Contradictions and unclear points

These are visible to the user. They are not decisions.

Resolved in Session 1:
1. CLAUDE.md was missing in Session 0; it is now in the repository root.
2. The CPU: the Session 0 prompt and CLAUDE.md section 6 say "Ryzen 7900K". The measured CPU is an AMD Ryzen 9 7900X. By the user's instruction the measured value stays; CLAUDE.md was not edited.

Still open:
3. CLAUDE.md rule 6 says only papers with status `READ` may be cited. Session 1 adds the status `PARTIAL` (sections listed) and allows PARTIAL papers in Related Work for the parts read. The wording of rule 6 should be updated by the user to match, or this reading confirmed.
4. The Session 0 commits used a provisional message format; CLAUDE.md requires `type(scope): summary`. Not rewritten.
5. D-005 says "comparable with published detectors". Published numbers are measured under different protocols (see RESEARCH.md), so "comparable" must be defined in Stage 0 and 2.
6. The prompt for Session 1 says the GitHub login was done, but `gh auth status` still reports no login on this machine and there is no `origin` remote. See STATUS.md blockers.
7. The folder `Document Template` is inside the repository root, not in its parent folder as the Session 1 prompt describes. It is untracked in git; the PDF and PowerPoint inside it come from an earlier project, so committing them needs the user's confirmation (D-001, D-014).
