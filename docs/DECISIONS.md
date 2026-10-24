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
- Status: Superseded by D-020 (2026-10-24, same day). The text below is kept word for word.
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

### D-020: Data access policy revised, approval-gated datasets allowed again
- Date: 2026-10-24
- Status: Decided (user)
- Question: Whether datasets that need a request form, a signed agreement or manual approval may be used (revisits D-013).
- Options considered: Keep D-013 (open datasets only); allow approval-gated datasets, requested and downloaded manually by the user.
- Criteria: Not recorded beyond the user's reason below.
- Evidence: The implementer read the official access pages and the request forms on 2026-10-24 (summary under "Check of the real access process" below; full facts in DATA_LOG.md).
- Decision and reason: Datasets that need a request form or approval are allowed again. The user will request and download them manually and place them where the implementer says (the data/ layout in DATA_LOG.md). Reason (stated by the user): access to FaceForensics++, FakeAVCeleb and Celeb-DF v2 is, as the user observed, easy and quick. The user has not verified this for every dataset and asked the implementer to check the real access process and wait time on each official page and to report any difference. The implementer's check is below. The Tier 1 and Tier 2 download list that was built under D-013 was withdrawn by the user and is replaced by the proposal in D-021.
- Rejected and why: The exclusion of approval-gated datasets in D-013, because the user now judges the gated datasets quick to obtain.
- Consequences / what could prove this wrong: Approvals may take longer than the user expects; the plan therefore keeps a fallback for the training data (see D-021). Several of the terms restrict derived data and publication (see below).
- Numbering note: the user wrote this decision as "D-017" and the data selection as "D-018". Those numbers were already in use in this log (D-017 environment, D-018 image scope), so the new entries are D-020 and D-021. Nothing else was renumbered.

Check of the real access process (official pages and forms, read 2026-10-24; nothing was submitted):

| Dataset | Route | What the form or page asks | Stated wait time | Differs from "easy and quick"? |
|---|---|---|---|---|
| FaceForensics++ and Google DFD (same download) | Google form, then a download script link by email | Name; lab or affiliation; advisor's name; advisor's email; research purpose; tick to agree the terms of use | The README says that if there is no reply within a week the email is probably bouncing, so a reply within about a week is expected. No faster promise | Not contradicted, but "quick" is not promised: up to about a week is implied |
| Celeb-DF v1 and v2 | Google form or Tencent form, then a download link by email | Name; affiliation; city; country; purpose; version (v1 or v2); a typed signature; an ACADEMIC email address | none stated | Not stated. Terms: non-commercial only; no reproduction or publication of any portion of the videos or of any derived data |
| Celeb-DF++ (a superset that contains the original Celeb-DF fakes) | Tencent form, or a Google form marked "not preferred"; approval then a link | Not read (the Tencent form is not readable by the tools used) | none stated | Not stated |
| FakeAVCeleb | Google form, then a download script link | Name; affiliation; advisor's name and email; a Gmail ID or an email associated with a Google account for sharing; a purpose of at least 100 words; tick to agree; an official academic or organisation email | none stated | Not stated. Terms: non-commercial only; no copying, publication or distribution of any portion of the data or of derived data |
| WildDeepfake | Agreement form; the official README then points to a Hugging Face repository for the download | Agreement and signature; academic email | "we may not reply to your email unless your request is rejected" | Download is on Hugging Face, ungated |
| DFDC | Meta page: an AWS account with an IAM user and access keys, sign-up through dfdc.ai; Kaggle route is only reported by a third party | AWS account number | none stated; the dfdc.ai host had an expired certificate on 2026-10-24 so its terms were not read | Yes: needs an AWS account; not "easy and quick" as far as can be verified |

The implementer could not verify any turnaround time other than the FaceForensics++ statement. Terms that matter for this project: Celeb-DF and FakeAVCeleb forbid publishing any derived data, which includes face crops and probably frames shown as figures in the paper; this has to be settled before any such figure is drafted.

### D-021: Data selection (requirements-driven)
- Date: 2026-10-24
- Status: Decided (user). Proposed by the implementer earlier on 2026-10-24 and approved as proposed by the user the same day.
- Question: Which datasets are downloaded, in which order, and in which subsets, given the requirements of the project, the D-012 split policy and the bandwidth and disk budget (part of O-003).
- Options considered: (A) An availability-driven list built under D-013 (Tier 1 and Tier 2; withdrawn by the user). (B) A requirements-driven list in which every dataset must serve a numbered requirement, using gated datasets requested by the user (this proposal). (C) Open datasets only, accepting a weak video protocol.
- Criteria (stated before the choice): (1) a dataset is kept only if it serves a numbered requirement below; "easy to get" is not a reason; (2) one shared training dataset per modality path, so candidate models are compared fairly (CLAUDE.md rule 4); (3) cross-dataset test sets are never used in training and are grouped by source and identity (D-012); (4) the data needed for the MVP and the first iterations arrives first; (5) total download time and disk use fit the measured bandwidth (4.09 MB/s) and the 404 GB free on D:; (6) the dataset's terms permit our use; (7) nothing is downloaded "just in case".
- Requirements the data must serve:
  - REQ-a. Training data for the MVP and the candidate-model comparison, one shared dataset per modality path.
  - REQ-b. Cross-dataset test sets for the headline generalisation result, with source and identity grouping.
  - REQ-c. Audio-visual data, because audio is part of analysing video, with modality-specific labels (D-011 needs a visual score and an audio score).
  - REQ-d. Audio-only data (a separate mode, D-019).
  - REQ-e. Image data of two kinds only (D-018): face-manipulated images (frames from face-swap or lip-sync video count, as in the user's definition) and fully generated images.
  - REQ-f. Data adequate for the rPPG feasibility test (video quality and compression, subject identity; D-006, O-010).
- Evidence: DATA_LOG.md (facts and what was re-checked), R-017 (the usual protocol trains on FF++ c23 and tests on Celeb-DF, DFD, DFDC and others, which makes our numbers comparable under D-005), R-053 (Celeb-DF++ structure), R-024 and R-014 (modality-specific labels), R-011 (a frozen-feature linear probe works from small training sets; the ablation numbers have not been extracted), R-064 and R-059 (compression matters for rPPG). Counts and sizes of the ForenSynths archives were read from the archives' own central directories by range requests on 2026-10-24.
- Decision and reason: the user approved the proposal as proposed on 2026-10-24. The decision is the keep, drop or defer table in DATA_LOG.md ("Requirements-driven data plan"), summarised here:
  - KEEP for REQ-a: FF++ c23 (originals plus Deepfakes, Face2Face, FaceSwap, NeuralTextures), ASVspoof 2019 LA for audio, the ProGAN validation set (8,000 images) as the small generated-image training set; LAV-DF for audio-visual training.
  - KEEP for REQ-b to REQ-f as tests: Celeb-DF v2, DFD c23 (originals and manipulated), a test-list subset of Celeb-DF++, WildDeepfake test shards, FakeAVCeleb, In-the-Wild, the ProGAN test set, the diffusion test set and an 8-generator subset of the CNN synthesis test set.
  - DEFER: DFDC, FF++ c40, DeepfakeTIMIT with VidTIMIT (fallback only), UBFC-rPPG.
  - DROP: everything else considered (list in DATA_LOG.md).
  - Totals (planning figures; the gated sizes are partly estimates): about 136 GB, about 9.2 hours at 4.09 MB/s, about 200 GB on disk after extraction.
- Rejected and why: Option (A), withdrawn by the user and built from availability. Option (C), because with open data only one modern face-video set and one tiny 2018 set remain (stream B finding), which cannot support the cross-dataset claim of D-012 or comparison with published tables (D-005).
- Consequences / what could prove this wrong: (1) If FF++ is not approved by 2026-10-26 midday, the MVP visual path falls back to LAV-DF, and the cross-dataset comparison with published FF++-trained tables is lost. (2) LAV-DF and FakeAVCeleb both take their real videos from VoxCeleb2, so their identities may overlap; the leakage script has to check this, and a dataset-disjoint claim holds only if the check passes. (3) The Celeb-DF and FakeAVCeleb terms forbid publishing derived data, which limits the figures that can appear in the paper and slides. (4) The gated datasets' sizes are not stated on their pages; the planning figures may be wrong, and the order may need to change. (5) A generated-image training set of 8,000 images is far smaller than the 720,000 used in R-011, so absolute numbers will not match that paper; D-005 is met by re-running under our own protocol. (6) Measured bandwidth was one reading from one server.

### D-022: Gated datasets are fetched on Monday 2026-10-26 on a fast connection and carried over on a drive
- Date: 2026-10-24
- Status: Decided (user), approved 2026-10-24 after the review: the method, the day and the fallback time (2026-10-26 18:00, then the video MVP trains on LAV-DF)
- Question: How and when the approval-gated datasets reach this machine (follow-up to D-020 and D-021).
- Options considered: Download them here as the approvals arrive (the measured speed here is about 4 MB/s from one server); download them on Monday on a connection faster than 970 Mbps and carry them over on a pen drive or a fast SSD.
- Criteria: Time to complete; risk of corrupted or incomplete copies.
- Evidence: Measured download speed here (EXPERIMENTS.md): 4.09 MB/s on one connection, with some hosts as low as 0.1 MB/s per connection. The user's other connection exceeds 970 Mbps (about 120 MB/s). The planned gated data is about 84 GB, plus LAV-DF at 25.6 GB if it is also fetched there.
- Decision and reason: The user will download the gated datasets that the implementer lists on Monday 2026-10-26 on the fast connection and transfer them to this machine on a pen drive or a faster SSD. The datasets in D-021 all remain in the plan; D-021 itself is unchanged. The datasets that can be fetched without approval keep downloading on this machine (ASVspoof 2019 LA, In-the-Wild, the ForenSynths and diffusion test sets, the CNN subset). The drive must use exFAT or NTFS, because FAT32 cannot hold files larger than 4 GB.
- Rejected and why: Downloading the gated datasets here, because of the measured speed.
- Update 2026-10-24 (later the same day, user instruction): FF++ c23 and Google DFD are downloaded on this machine now with the authors' script (scripts/data/fetch_ffpp.sh, server EU2), because they are only about 35 GB in total and the visual MVP depends on FF++; the user is downloading Celeb-DF v2 from the Google Drive link; Celeb-DF++ (about 54 GB, only the test-list videos of the 21 new generators are needed) is deferred; WildDeepfake and FakeAVCeleb wait for the owners' approvals; LAV-DF waits for the user's Hugging Face token. The Monday drive transfer remains for whatever is not on this machine by then.
- Consequences / what could prove this wrong: (1) The fallback date in D-021 (2026-10-26 midday) no longer fits: the gated data cannot be on this machine before Monday afternoon at the earliest. New fallback, decided autonomously and needing the user's review: if FF++ has not passed the completeness checks by Monday 2026-10-26 18:00, the video MVP trains on LAV-DF and FF++ joins the first iteration. (2) The audio MVP and the generated-image MVP do not depend on gated data and run on Monday daytime. (3) A drive copy can corrupt data; the source machine produces a file list with sizes, this machine compares it, runs ffprobe on every video and records SHA-256. (4) If the Monday download fails for a dataset, nothing else is available on this machine for it.

### D-023: Stage 0 to 2 work starts now, in parallel with the downloads
- Date: 2026-10-24
- Status: Decided (user)
- Question: Whether to build the parts of the project that do not need the missing datasets while the downloads continue, which requires lifting the Session 1 rule of no metric choice and no training or detection code.
- Options considered: Finish the Session 1 handoff first and start Stage 0 to 2 in a new session; start now.
- Criteria: The schedule has seven calendar days (D-009, D-016); most of Stage 0 to 2 does not need the gated datasets.
- Evidence: PLAN.md day 2 (Stage 0 to 2) and the data-purpose split in DATA_LOG.md: only the training sets (FF++, ASVspoof 2019, ProGAN, LAV-DF) are needed first; the cross-dataset test sets are touched once per frozen iteration (CLAUDE.md rule 3).
- Decision and reason: Start now. The Session 1 restriction is lifted for Stage 0 to 2 only: the problem statement and requirements, the data manifests, grouping and leakage-check tooling, the evaluation harness (metrics, bootstrap, temperature scaling, the `--final` guard), and the data-preparation pipeline (video decoding, face cropping, audio extraction) with its speed measurements. No model is trained and no model, backbone, fine-tuning method or fusion scheme is chosen (O-006 to O-009 stay open).
- Rejected and why: Waiting for a new session, because the work is independent of the pending data.
- Consequences / what could prove this wrong: Code written before the protocol is frozen could be reworked; the protocol is therefore fixed first (D-024). Subagents write code in their own sub-folders under `src/dfd/` and `tests/` (extension of D-015 for code), and only the main agent commits.

### D-024: Evaluation protocol: metrics, threshold, calibration, uncertainty (settles O-005)
- Date: 2026-10-24
- Status: Decided (user), approved 2026-10-24 after the review: the four choices and the autonomous items (a) onwards
- Question: Which metric is the headline, how the decision threshold is chosen, whether and how probabilities are calibrated, and how uncertainty is reported (O-005).
- Options considered: Headline: ROC-AUC, EER, balanced accuracy. Threshold: validation-chosen maximising balanced accuracy, validation EER threshold, fixed 0.5. Calibration: temperature scaling, report only, Platt scaling. Uncertainty: 3 seeds with 2000 resamples at 95 percent, 5 seeds with 1000 resamples, 3 seeds with 10000 resamples at 99 percent.
- Criteria: The threshold is chosen only on validation data and never on test data (R-011, R-047); the headline must not depend on the threshold or the class balance; the schedule fits 3 seeds per candidate (comparison rule 2).
- Evidence: R-017 and R-053 report AUC; R-016 for temperature scaling and ECE; R-011 and R-047 for the threshold practice; MATH_NOTES.md for the definitions.
- Decision and reason (the user's choices): (1) The headline metric is ROC-AUC, with a bootstrap interval. (2) Secondary metrics: EER, balanced accuracy at the chosen threshold, log loss, ECE (15 equal-width bins, top-label version) and Brier score. For standalone audio on ASVspoof 2019 the EER is reported as well, for comparison with R-012 and R-013. (3) The threshold is chosen on validation data from the training distribution by maximising balanced accuracy; the cross-dataset test uses that same fixed threshold; accuracy at a fixed 0.5 is reported as a reference only. (4) Calibration: temperature scaling fitted on the validation set; ECE and Brier are reported before and after. (5) Uncertainty: 3 seeds per candidate, 2000 bootstrap resamples, 95 percent percentile intervals, paired comparisons (comparison rules 3 to 5); the resampling unit is the source or identity group, never the frame.
- Autonomous items (needs user review): (a) frame-to-video aggregation: the video score is the mean of the per-frame probabilities, a deliberately simple default; R-025 shows that the aggregation rule changes AUC, so alternatives are an iteration hypothesis, not part of the frozen protocol. (b) Validation data for the threshold and for temperature scaling is held-out data from the training dataset, split by the same grouping as D-012. (c) A resample that contains only one class is redrawn.
- Rejected and why: EER and balanced accuracy as the headline (depend on a threshold or are less common for video); fixed 0.5 as the threshold policy (honest only for calibrated, balanced scores); report-only calibration and Platt scaling (not chosen by the user); 5 seeds or 10000 resamples (do not fit the schedule).
- Consequences / what could prove this wrong: A validation set from the training distribution may be too easy, so the chosen threshold may transfer badly across datasets; this will show as a gap between in-dataset and cross-dataset balanced accuracy and is reported, not hidden. Temperature scaling does not fix a shift between datasets.

### D-025: Definition of "comparable with published detectors" (follows from D-005)
- Date: 2026-10-24
- Status: Decided (user), approved 2026-10-24 after the review, amended by the user (replaces the version proposed in REQUIREMENTS.md section 8)
- Question: What "comparable with published detectors" (D-005) means in a checkable way.
- Options considered: Quote paper numbers beside ours; re-run a published detector family under our protocol and compare with a paired interval; match a published number within a tolerance.
- Criteria: The comparison must obey the comparison rules (same data, split, budget and metric; at least 3 seeds; paired bootstrap), and in-dataset and cross-dataset numbers are never mixed (RESEARCH.md rule).
- Evidence: R-017 (the common FF++ to other-datasets protocol); the protocol differences between papers recorded in RESEARCH.md notes 1 and 12.
- Decision and reason (user's amendment): "Comparable" means non-inferior. Ours is comparable to the re-run baseline when the lower end of the paired 95 percent bootstrap interval of (ours minus the baseline), resampled over source or identity groups, is above minus delta. Delta is stated before the final comparison and is chosen from the measured seed-to-seed spread of the baseline. An interval that only contains zero is reported as "not distinguished", never as comparable. Ours is superior when the whole interval is above zero. The baseline is a published detector family re-run under our protocol with the same data, split, budget and metric; quoted paper numbers are context only, with their own protocol named.
- Rejected and why: Quoting paper numbers beside ours (different protocols), and matching a published number (not reproducible under our splits).
- Consequences / what could prove this wrong: Delta has to be stated before the final comparison and derived from measured baseline seed spread (at least 3 seeds), so the baseline must be run before delta exists. If the chosen baseline cannot be trained within the time budget, "comparable" would have to be weakened and said so.

### D-026: Content-duplicate rule for test sets, ProGAN class split, Celeb-DF fake grouping
- Date: 2026-10-24
- Status: Decided (user), approved 2026-10-24 after the review
- Question: Three data-handling points raised by the Stage 1 tooling (stream S2-B): what to do with real images that appear byte for byte in both a training or validation set and a test_x set; whether the ProGAN validation set may be split by class; how to group Celeb-DF fake videos.
- Options considered: (1) Drop the duplicates from test_x; drop the whole stylegan and stylegan2 cat folders from test_x; hold the cat class out of training; accept and document the overlap. (2) Class-disjoint train and val split of the 20 LSUN classes; a random image split. (3) Group Celeb-DF fakes by the first identity in the name; by the second; by connected components of the identity graph.
- Criteria: D-012 (no content shared between train or validation and any test set); groups must be large enough in number for the bootstrap; the choice must be documented and counted.
- Evidence: Main-agent check on 2026-10-24: 200 of the 1,997 real images in each of `stylegan/cat` and `stylegan2/cat` of the CNN subset are byte-identical (SHA-256) to the 200 real cat images of `progan_val`, and the 200 real cat images of `progan_test` are identical to the same ones. Stream S2-B measured 800 content hashes in different splits over the three image sets (all real LSUN images) and 3,604 further repeated hashes inside the CNN subset. After the exclusion was implemented, the final count is 1,200 excluded samples, all in the CNN subset: stylegan/cat 400, stylegan2/cat 400 and stylegan2/horse 400 (the horse folder was not in the first count); the rule also compares against progan_test (test_in), whose reals are the same images. The leakage check with content hashing then passes on 80,803 files. Celeb-DF v2: 59 identities; a strict identity-graph component gives only 3 components (38, 10 and 10 identities), too coarse for a bootstrap.
- Decision and reason: (1) Any sample of a test_x set whose content hash equals the hash of a train or validation sample is excluded from test_x. The exclusions are listed in a file with the reason and counted, and the leakage check, run with content hashing for the image datasets, must then pass. Duplicates inside a single split are a warning, not a failure. (2) The ProGAN validation set is split by LSUN class, 14 classes for training and 6 for validation (group = class, the coarsest source grouping available for images with no identity). The scientific cost is that the threshold and temperature are fitted on classes not seen in training. (3) Celeb-DF fakes take their group from the first identity in the file name, assumed to be the target (the base video); both identities are kept in the notes. Celeb-DF is test-only here, so the grouping affects only the bootstrap resampling units, not leakage.
- Rejected and why: Accepting the overlap (breaks D-012). Dropping whole generator folders or the cat class from training (removes more data than necessary). A random image split of ProGAN (images of one class are not independent sources).
- Consequences / what could prove this wrong: If the assumption that the first Celeb-DF identity is the target is wrong, the bootstrap groups are still valid clusters but differently composed. The stylegan and stylegan2 cat test sets and the stylegan2 horse test set lose 400 real images each (about 20 percent of their reals), which slightly changes their class balance. ASVspoof, FF++ and Celeb-DF have not been content-hashed.

### D-027: Prediction response structure shared by inference and interface
- Date: 2026-10-24
- Status: Decided (user). Proposed by the implementer earlier on 2026-10-24 and approved as proposed by the user the same day.
- Question: What exactly the inference code returns and the interface shows, so that the two can be built in parallel.
- Options considered: A flat result (one score and a label); a structured response with per-modality scores, status, warnings and errors (this proposal).
- Criteria: It must carry what D-011 requires (a score per modality present and a fused score); it must say what the score means; it must make a development stand-in impossible to mistake for a real result.
- Evidence: D-011, D-024 (calibrated probability, validation threshold), PROTOCOL.md (score files), CLAUDE.md rule 8 (claims discipline).
- Decision and reason (proposed): one structured response, schema version 1, with these fields.
  - `schema_version`: "1".
  - `request_id`: unique identifier of the request.
  - `input`: `filename`, `media_type` (image, video or audio), `mime_type`, `size_bytes`, `sha256`, `duration_s` (null for images), `frames_used` (null for audio).
  - `inference_source`: `real` or `mock`. Always present.
  - `model`: `checkpoint_id`, `config_hash`, `preprocess_id`, `git_commit`, `iteration`, `protocol` (the PROTOCOL.md version), `calibration` (method, temperature, the split it was fitted on), `threshold` and how it was chosen. Null when `inference_source` is `mock`.
  - `result.fused`: `score` in [0, 1] and `decision` (fake, real or undetermined). For an image the fused score is the visual score; for an audio file it is the audio score; for a video it follows the combining rule of O-009 (D-011).
  - `result.per_modality`: one entry per modality present (visual, audio, image), each with `available` (true or false, with a reason such as "video has no audio track"), `score`, `n_units` (frames or audio windows used).
  - `score_meaning`: the fixed text "Calibrated probability that this content is fake, from a model fitted on the training data of this project; valid only for the kinds of content in that data; it is not proof."
  - `status`: `state` (ok, partial or failed), `processing_time_ms`, and a list of `stages` with their times (validate, decode, detect, score, calibrate, fuse).
  - `warnings`: a list of `code`, `message` and optional `modality` (for example NO_FACE_DETECTED, SHORT_AUDIO, LOW_RESOLUTION, MULTIPLE_FACES).
  - `errors`: a list of `code` and `message` (for example UNSUPPORTED_MEDIA_TYPE, UNREADABLE_FILE); when present `state` is failed and `result` is null.
- Rejected and why: a flat score and label (cannot represent a missing modality or a warning).
- Consequences / what could prove this wrong: the combining rule for video is not known until O-009 is settled, so `result.fused` for video is provisional; the decision rule (threshold) depends on the model and the calibration; the structure may need a version bump if localisation is ever added (excluded by D-011 in version 1).

### D-028: Shared preprocessing, configuration and checkpoint metadata
- Date: 2026-10-24
- Status: Decided (user). Proposed by the implementer earlier on 2026-10-24 and approved as proposed by the user the same day.
- Question: How training, evaluation and inference avoid duplicate code and silent mismatches.
- Options considered: Each stage with its own preprocessing; one shared preprocessing package and one metadata file (this proposal).
- Criteria: The same function must prepare an input in training, evaluation and inference; a mismatch must be detected, not assumed away.
- Evidence: src/dfd/prep exists as one package; PROTOCOL.md already separates score files from models.
- Decision and reason (proposed): (1) one preprocessing package (`dfd.prep`) is imported by the loaders, the evaluation preparation and the inference library; no other copy. (2) One configuration holds the preprocessing parameters (frame count, crop factor, crop size, audio sampling rate and window) and its hash is the `preprocess_id`. (3) Every checkpoint is written with a metadata file holding the configuration, `preprocess_id`, git commit, manifest and split IDs, seed, library versions, the calibration temperature and the validation threshold. (4) Inference refuses a checkpoint whose `preprocess_id` differs from the one in use. (5) Evaluation reads score files only and never imports a model. The file format of the configuration and metadata is open (O-012).
- Rejected and why: separate copies per stage (a silent mismatch changes results without warning).
- Consequences / what could prove this wrong: a fixed frame count at inference may differ from what a model trained best on; if so the iteration records it as a parameter change with a new `preprocess_id`.

### D-029: Rules for any mock or development adapter
- Date: 2026-10-24
- Status: Decided (user). Proposed by the implementer earlier on 2026-10-24 and approved as proposed by the user the same day.
- Question: How to build the interface before a real model exists without any risk of a fake result being reported.
- Options considered: No stand-in (the interface waits); a clearly labelled stand-in under strict rules (this proposal).
- Criteria: A stand-in may unblock the interface work but must never be mistaken for a measurement (CLAUDE.md rules 1 and 8).
- Evidence: CLAUDE.md rule 1 (nothing unmeasured is a result).
- Decision and reason (proposed): (1) a mock adapter lives in its own module whose name contains `mock`; (2) it always sets `inference_source` to `mock` and leaves `model` empty; (3) every interface shows a visible MOCK banner whenever `inference_source` is not `real`, and a test checks this; (4) the evaluation harness refuses any run whose metadata says mock; (5) the paper, report and slide builds fail if a number cites a run whose source is mock; (6) its scores are deterministic from the file hash so tests are repeatable, and carry no meaning.
- Rejected and why: waiting for real models (blocks two streams for days).
- Consequences / what could prove this wrong: a banner can be removed by mistake, so the test in (3) is part of the acceptance criteria of the interface stream.

### D-030: Workstreams, ownership and integration
- Date: 2026-10-24
- Status: Decided (user). Proposed by the implementer earlier on 2026-10-24 and approved as proposed by the user the same day.
- Question: How three people work in parallel without blocking one another or breaking shared files.
- Options considered: One person at a time; six streams grouped into three roles with owned folders and an integrator (this proposal; details in WORKSTREAMS.md).
- Criteria: No stream starts on an undecided choice; every stream has an interface and acceptance criteria; shared files change through one integrator.
- Evidence: WORKSTREAMS.md sections 4 to 8.
- Decision and reason (proposed): the streams are S1 data pipeline and loaders, S2 modelling and training, S3 evaluation and tracking, S4 inference library, S5 user interface and S6 writing and integration, grouped as Role 1 (S1 and S3), Role 2 (S2) and Role 3 (S4, S5 and the integration of S6). Each stream writes only inside its own folders; the main agent integrates and runs all tests before each commit.
- Rejected and why: a single shared folder with free edits (merge conflicts and unnoticed breakage).
- Consequences / what could prove this wrong: Role 3 carries the integration load near the deadline; the grouping may need rebalancing once the real workload is visible.

### D-031: Software form is a local web page (settles O-011)
- Date: 2026-10-24
- Status: Decided (user)
- Question: What form the user-facing software takes (O-011).
- Options considered: Command line plus local web page (the implementer's recommendation); local web page only; command line only.
- Criteria: Demonstration to examiners; time cost before 2026-10-30.
- Evidence: none yet (a requirement choice).
- Decision and reason: A local web page only: the user uploads an image, video or audio file in a browser and sees the response of D-027. The user chose this over the recommended combination.
- Rejected and why: The command line (as the only form, or as a second form): not chosen by the user.
- Consequences / what could prove this wrong: A browser talks to Python through a small local web service, so O-013 narrows to which service and framework (still open; no framework is chosen here). Evaluation and experiments still run from scripts and the evaluation command, which is not a user interface. The interface must show the MOCK banner of D-029 whenever `inference_source` is not `real`.

### D-032: The MVP starts from frozen pretrained features with a linear head (partly settles O-006)
- Date: 2026-10-24
- Status: Decided (user) for the MVP starting point; the choice of backbone and every later model stays Open (O-007, O-008)
- Question: What the simplest end-to-end first version is for each path (O-006).
- Options considered: Frozen pretrained features with a linear classifier; a small network from scratch; a fine-tuned pretrained backbone.
- Criteria: It runs from raw file to score and metric under the Stage 2 protocol; it needs no tuning to produce a first number; it trains in minutes on the measured machine; it is simple to explain (O-006 criteria).
- Evidence: R-011 (a frozen pretrained feature space with a linear head generalised well across generators for still images; protocol: trained on ProGAN, tested on other generators); R-017 (plain backbones are competitive under a matched recipe). No experiment of ours yet.
- Decision and reason: The MVP (iteration I-0) for each path uses a frozen pretrained feature extractor and a linear classifier on top. This fixes only the form of the MVP. Which pretrained extractor to use for each modality is not decided and is the first item of O-007; the small network from scratch is still run as a reference in Stage 4 (D-004).
- Rejected and why: A network from scratch as the MVP (slowest to a usable number) and a fine-tuned backbone as the MVP (needs the most VRAM and tuning on day 3): not chosen.
- Consequences / what could prove this wrong: A frozen extractor may miss fine artefacts that matter for face video; this becomes the hypothesis of the first iterations (O-008). Pretrained weights need their licence, source and hash in DATA_LOG.md before use (CLAUDE.md rule 7).

### D-033: Image input: face crop for face images, whole image otherwise (settles the image part of O-002)
- Date: 2026-10-24
- Status: Decided (user)
- Question: How a still image is preprocessed (O-002, image path).
- Options considered: Face crop for face images and the whole image otherwise; always the whole image; always a face crop.
- Criteria: Must serve both image kinds of D-018: face-manipulated images (frames from face-swap or lip-sync video count) and fully generated images.
- Evidence: R-001 (face crop with an enlargement factor for face forgery); R-011 (whole-image generated-image detection).
- Decision and reason: If a face is found, the detector works on the face crop; otherwise on the whole image. The choice made is reported in the response (D-027, a warning or a field) so that nobody reads a whole-image score as a face score.
- Rejected and why: Always the whole image and always a face crop: each fails one of the two image kinds.
- Consequences / what could prove this wrong: The two image kinds may need separate models; that is reported as separate test conditions already (D-018). Face detection errors decide which path an image takes; the detector's rate on real data is a Stage 1 measurement. The video and audio paths are not decided by this entry.

### D-034: Face detector and default preprocessing parameters
- Date: 2026-10-24
- Status: Decided (user), approved 2026-10-24 after the review
- Question: Which face detector the preparation pipeline uses, and the default values of its parameters (frames per video, crop enlargement, crop size, JPEG quality, handling of frames without a face and of several faces).
- Options considered: Detectors: YuNet 2023mar, UltraFace RFB-320, SCRFD 500M (all CPU, through OpenCV). Parameter values are listed with the decision.
- Criteria: Runs on CPU under Python 3.13 without a package that clashes with the project environment; speed; behaviour on rotated, small and face-free images; a licence that allows our use (CLAUDE.md rule 7); the detector must not be the bottleneck.
- Evidence: Measurement M-001 (EXPERIMENTS.md, stream S2-C, caveats there). On 7,488 sampled FF++ frames all three detectors found a face in 100 percent, so that rate does not separate them; YuNet 12 ms per frame, UltraFace 5 ms, SCRFD 30 ms; hits on 500 face-free images: YuNet 6.0, UltraFace 20.2, SCRFD 4.6 percent; decoding takes 73 percent of the time per video and detection 24 percent. Licences: YuNet is MIT (OpenCV Zoo folder, read 2026-10-24); the InsightFace SCRFD weights are non-commercial research only; the UltraFace licence is not verified. MediaPipe was not installable without a clash with the headless OpenCV package.
- Decision and reason: (1) The detector is YuNet 2023mar (`yunet_2023mar.onnx`, input long side at most 640 px) through `cv2.FaceDetectorYN`: small, MIT licensed, fast enough, and the detector choice changes the total time little. SCRFD and UltraFace were used only as benchmark comparisons and are not used by the project. (2) Defaults of the shared preprocessing configuration (D-028), each a parameter and not a frozen choice: 16 uniformly spaced frames per video; a square crop around the largest face enlarged by a factor of 1.3 (the value of R-001); crop size 224 by 224; JPEG quality 95. Changing any of them gives a new `preprocess_id`. (3) A frame with no detection is recorded in the index and left out of the clip score; a clip with no detected face at all returns the warning NO_FACE_DETECTED (D-027); an image with no face goes through the whole-image path (D-033). (4) With several faces the largest is used and the warning MULTIPLE_FACES is returned; 11 to 15 percent of sampled FF++ frames have more than one box, so this rule is revisited if the measurements show a problem.
- Further evidence (M-002, added the same day at the user's request): on 300 FF++ val frames and 300 Celeb-DF dev_x frames all three detectors found a face in every frame, never disagreed, and their largest boxes agree (mean IoU 0.89 to 0.92); the difference that matters is speed and licence. D-034 was confirmed by the user on 2026-10-24; the crops are not redone.
- Rejected and why: SCRFD (research-only weights, 2.5 times slower), UltraFace (20 percent hits on face-free images, weakest on small faces), MediaPipe (dependency clash). Carrying the previous box or a centre crop for frames without a face (invents a face position).
- Consequences / what could prove this wrong: The frame count, enlargement and JPEG quality were chosen for speed and simplicity, not because they give the best detector accuracy; their effect on accuracy is not measured and is a candidate iteration. The 100 percent detection rate on FF++ says nothing about hard faces; no face ground truth exists here. Sequential decoding is used because seeking is about 4 times slower on FF++.

### D-035: Development slices (split `dev_x`) for ranking and hyperparameters
- Date: 2026-10-24
- Status: Decided (user), approved 2026-10-24 after the review, including amendment 1
- Question: Which cross-dataset data may be used to rank candidates and choose hyperparameters before the final tests. The user's instruction for O-007 refers to "the dev-set decision"; no such decision existed in this log, and ranking candidates on the final cross-dataset test sets would break CLAUDE.md rule 3.
- Options considered: Rank on the in-dataset validation split only; rank on the final test_x sets (not allowed); carve a development slice out of each cross-dataset source (this decision).
- Criteria: The ranking must reflect generalisation to data not used in training, not in-distribution fit; the slice must be group-disjoint from the final test_x (D-012); the rule must be fixed and seeded before any extraction.
- Evidence: R-017 and R-053 (in-dataset and cross-dataset numbers differ greatly: for the same detectors above 0.93 within FF++ and about 0.77 across datasets, R-017); R-011 (threshold and selection on test data give optimistic numbers).
- Decision and reason: A new split name `dev_x`. It is a slice of a dataset that is otherwise `test_x`, chosen by a seeded draw at the level of the group, and removed from `test_x`. It may be used to rank candidates, choose hyperparameters and measure preprocessing tools (for example face detectors, using the frames only, no scores); it is never reported as the headline and it is not a final split, so the `--final` guard does not apply to it. The rule is: sort the group names, draw with `numpy.random.default_rng(20261024).permutation`, and take the first k groups; every sample of a drawn group (real and fake) goes to `dev_x`. The slices are: Celeb-DF v2, 15 of its 59 identity groups; In-the-Wild, 12 of its speakers (from `meta.csv`; 54 speakers, see Amendment 1); the CNN synthesis subset, 2 of its 8 generator folders. In-dataset validation (`val`) keeps its role in D-024: the threshold, the temperature and early stopping. Content duplicates between `dev_x` and `test_x` are excluded from `test_x` (D-026 rule extended).
- Amendment 1 (2026-10-24, same day, Decided (user), approved the same day; made before any model score exists and prompted only by the structure of the data): (a) In-the-Wild has 54 speakers in its `meta.csv` (the dataset paper says 58 celebrities), so its slice is 12 of 54. (b) The first draw over the 8 CNN-synthesis generator folders gave stylegan and whichfaceisreal, but stylegan and stylegan2 share byte-identical real images; after content de-duplication stylegan2 kept only 3,593 reals against 7,988 fakes in test_x (its car and cat fakes had no reals), which makes a per-generator AUC unreliable. The draw unit is therefore changed: generators that share real-image content form one unit (stylegan with stylegan2), and the same seeded rule draws 2 of the resulting 7 units. The seed and the rule are otherwise unchanged. (c) The 300 YouTube-real videos of Celeb-DF v2 have no identity and are not drawn; they stay in test_x, so the Celeb-DF dev_x slice has 146 real and 1,582 fake videos; AUC does not depend on the class ratio but the interval over the few reals is wider.
- Rejected and why: Ranking on in-dataset validation only (selects for in-distribution fit, not generalisation); ranking on test_x (breaks rule 3).
- Consequences / what could prove this wrong: The final test_x sets shrink (Celeb-DF v2 to 44 identity groups plus its YouTube-real videos, In-the-Wild to 42 speakers, the CNN subset to 5 generator units), and the official Celeb-DF test list is not used. With 15 identity groups the bootstrap interval of a dev_x AUC will be wide, so the rule "where intervals overlap the cheaper candidate wins" is likely to apply often. The ranking is only as good as these small slices. Added on the user's review (2026-10-24): with 15 identity groups the percentile bootstrap undercovers, so ties will partly reflect noise; an exact paired sign-flip permutation test over groups is therefore run as a sensitivity check beside the bootstrap (to be implemented in dfd.eval before the first ranking is reported; the ranking rule of D-036 itself uses the bootstrap). The official Celeb-DF test list is not used, so published Celeb-DF numbers are context only, never a comparison.

### D-036: Protocol and selection rule of the O-007 encoder sweep (iteration I-0)
- Date: 2026-10-24
- Status: Decided (user), approved 2026-10-24 after the review, with amendment to item (2)
- Question: How the candidate encoders are compared and chosen for fine-tuning (O-007), designed as a measured comparison instead of a list to pick from.
- Options considered: The user picks from a list; a measured comparison under a pre-declared protocol and rule (this proposal).
- Criteria: The same crops and splits for every candidate; features extracted once; 3 seeds; an interval over sources; a ranking on a development slice, not on the final tests (D-035); a cost record for every candidate; a rule that prefers the cheaper candidate when the evidence does not separate them (comparison rule 5).
- Evidence: none yet (design only); docs/O007_SWEEP.md has the full protocol, E-001 to E-003 in EXPERIMENTS.md hold the hypotheses and criteria written before any extraction.
- Decision and reason: (1) The sweep is iteration I-0: frozen encoders, linear head, three tasks (T-V face video frames, T-I generated images, T-A speech audio). (2) Ranking by the mean dev_x AUC with the paired 95 percent bootstrap over groups; the tie set is every candidate whose paired interval against the top-ranked candidate includes zero, and the winner is the cheapest in that set (lower end-to-end extraction time per 1,000 images or clips, then fewer parameters); for the top three entering I-1 the tie set comes first, cheapest first, then the remaining candidates by mean dev_x AUC (this ordering is Decided autonomously - needs user review); the from-scratch reference is a reported baseline and is excluded from the selection. (3) The top three of each task enter iteration I-1, a fine-tuning comparison under one budget with three learning rates, measuring steps to a fixed dev target, learning-rate sensitivity, peak VRAM, time per epoch and instability events; the claim "easiest to fine-tune" is defined operationally in O007_SWEEP.md section 9. (4) The shortlist comes from the licence-verified candidates in docs/_agents/O7IMG.md and O7AUD.md once the user approves it.
- Rejected and why: Choosing the encoder by the user from a list (the user asked for a measured comparison); ranking on the final test sets (rule 3).
- Consequences / what could prove this wrong: The dev slices are small, so many candidates will tie and the cost rule will decide often; frozen features may be inadequate for face forgery (failure criterion f2 of E-001); the final model may differ from the sweep winner after fine-tuning, which is exactly what I-1 measures.

## Open decisions

Settled by Session 1 decisions: O-001 (D-011); O-005 (D-024); O-011 (D-031); O-006 for the MVP starting point (D-032); O-002 for the image path (D-033). O-004 is settled as an outline by D-012 (script and details pending). O-002 is partly answered by D-018 and D-019 (what the modalities are); how each input type is handled remains open. O-003 is constrained by D-020 (access policy) and settled by D-021 (approved 2026-10-24). None of the questions below is otherwise answered. For each one, the options are those the implementer would compare, the criteria are stated before any result exists, and the evidence column says what must be collected. Paper IDs refer to RESEARCH.md. Pointers to papers show where options come from; they are not choices.

| ID | Question | Options to compare | Criteria (stated before results) | Evidence needed |
|---|---|---|---|---|
| O-002 | How image-only, video and audio inputs are each handled, and which claims each supports (the scope is fixed by D-018 and D-019) | Images: face crop then classifier (as R-001) versus whole-image detector (as R-011). Video: per-frame scores aggregated over time versus a clip-level temporal model. Audio: waveform model (R-012) versus spectral-feature model. Cross-modal: consistency between audio and lips (R-015). | A claim is made only for the kind of input the training and test data cover. Each path must have a held-out test with its own protocol. | The generators present in each dataset; cross-dataset results from the literature (R-004, R-011, R-017) showing how fragile transfer is. |
| O-003 | Datasets to use, and which are for training, validation and held-out testing (approval-gated datasets allowed by D-020; the proposal is D-021) | Visual: FaceForensics++ (R-001), Celeb-DF (R-002), DFDC (R-003), and others on the reading list. Audio: ASVspoof 2019 LA (R-013) and newer editions. Audio-visual: FakeAVCeleb (R-014, which its paper says is gated), DFDC. Generated images: those used in R-011 and others. | Serves a numbered requirement of D-021; terms permit our use; size fits the 404 GB free on D:; identities or source videos available so that splits can be disjoint (D-012); labels match D-011; generators diverse and not all outdated; results comparable with published work. | Stream B facts in DATA_LOG.md: access route, licence, size, counts, identity metadata, official splits. |
| O-004 (remaining part) | Leakage script and dataset-specific split details | Per-dataset grouping keys (identity, source video, recording); automatic overlap check; storing each split as a file with an ID and hash. | The script proves zero overlap; where identity metadata is missing the fallback is recorded as `not established`. | Identity and source metadata of each chosen dataset. |
| O-005 (settled by D-024; row kept for the options considered) | Metrics, threshold policy, calibration | Ranking metrics: ROC-AUC, average precision, EER. Threshold metrics: accuracy at a fixed threshold, balanced accuracy. Probability metrics: log loss, Brier score, ECE (R-016). Audio: min t-DCF with EER (R-012, R-013). Threshold policy: fixed 0.5; threshold chosen on validation data; EER threshold on validation. Calibration: none; temperature scaling (R-016). Aggregation: frame-level versus video-level. | The threshold is chosen only on validation data and never on test data (R-011 shows that tuning the threshold on the test set gives an "oracle" number that is not achievable in practice). Class balance in the test set is stated. A metric must be defined mathematically in the document before use. | Class ratios of the chosen datasets; seed-to-seed variance, which can be large (R-012 reports a spread of EER between 1.19 and 2.06 percent over seeds for one baseline); how the DFDC organisers used log loss (R-003). |
| O-006 | MVP design: the simplest thing that runs end to end | (a) Small CNN from scratch on face crops, one dataset. (b) Frozen pretrained features with a linear classifier (R-011 style). (c) Fine-tuned pretrained backbone. (d) Audio: small CNN on a spectrogram. | It runs from raw input file to a score and a metric under the Stage 2 protocol; it needs no tuning to produce a number; it trains within minutes on the measured machine; it is simple enough to explain. | The Stage 2 protocol; measured data-loading speed (not yet measured); a first sizing of each candidate. |
| O-007 (design Proposed as D-036) | Candidate backbones per modality, including a small network built from scratch as a reference | Visual: small CNN from scratch; ResNet family; Xception and EfficientNet-B4 (both used as strong plain baselines in R-017); frozen CLIP-type encoder with a linear or nearest-neighbour classifier (R-011). Audio: small CNN on a spectrogram; a lightweight model such as AASIST-L (85K parameters, R-012). | The cost axes in this document, measured on one machine under the comparison rules. | Experiments in Stage 4. Prior evidence: in one benchmark a plain Xception was within roughly one to two AUC points of more elaborate detectors under identical settings (R-017), which suggests the backbone and training recipe may matter more than the method name. |
| O-008 | Fine-tuning strategy | Linear probe; full fine-tuning; partial unfreezing of the last blocks; differential learning rates; low-rank adapters (not yet read). | Validation metric at a fixed budget; peak VRAM; stability across seeds; sensitivity to learning rate. | Stage 4 and 5 runs. Prior evidence: ImageNet accuracy correlates with transfer accuracy for both fixed features and fine-tuning across 12 natural-image tasks (R-018), but deepfake data is not natural-image classification, so this must be tested here. |
| O-009 | Fusion strategy (how the fused score of D-011 is formed) | Late fusion of per-modality scores (average or logistic regression); feature-level fusion; cross-modal consistency model (R-015); joint training (R-020, to be read). | Must handle inputs with a missing modality (a still image has no audio, D-011); must be at least as good as the best single modality on the validation data; must be explainable. | Per-modality baselines first. Prior evidence: simple ensembles of unimodal detectors did poorly on FakeAVCeleb in the dataset paper (R-014), which is a reason to test fusion carefully. |
| O-010 | rPPG feasibility test design | (a) Signal sanity: can heart rate be recovered from our data at all, given that most deepfake datasets have no pulse ground truth (R-008). (b) Separation: do rPPG features separate real from fake under an identity-disjoint split, for several generator families (R-009, R-007). (c) Added value: does adding rPPG evidence raise the fused detector's held-out metric beyond the bootstrap interval. (d) Negative controls: time-shuffled or mismatched signals. | The success and failure thresholds are written before the run. rPPG is adopted only if (c) shows a gain larger than the interval of the paired difference. | Stage 6 experiments. Literature: R-005 to R-010 and Stream C3. A dataset subset with enough frames, resolution and lighting stability for rPPG; skin-tone coverage (R-009 notes that rPPG is sensitive to skin tone and lighting). |
| O-011 | Software form | Command line; local web app; both. | Time cost; need to demonstrate to examiners; ease of reproducing results; input handling for image, video and audio files. | Requirements from Stage 0. |
| O-012 | Format of the shared configuration and checkpoint metadata (D-028), and whether experiment tracking uses plain files or a tracking library | Plain JSON files under version control; YAML; a tracking library | Must be hashable, human readable and need no extra service; no duplicate source of truth | A small trial in the loader work |
| O-013 | How the web page calls the inference library (D-031 implies a small local web service); which framework serves it | A local web service in Python (framework to be compared); in-process call from a page served by the same program | Same response structure in every case (D-027); lowest integration risk before 30 October | Settled: form by D-031, framework by D-043 (Flask, decided autonomously - needs user review) |

## Contradictions and unclear points

These are visible to the user. They are not decisions.

Resolved in Session 1:
1. CLAUDE.md was missing in Session 0; it is now in the repository root.
2. The CPU: the Session 0 prompt and CLAUDE.md section 6 say "Ryzen 7900K". The measured CPU is an AMD Ryzen 9 7900X. By the user's instruction the measured value stays; CLAUDE.md was not edited.

Resolved on 2026-10-24 on the user's review (D-037):
3. (resolved, rule 6 now allows PARTIAL papers for the parts read) CLAUDE.md rule 6 says only papers with status `READ` may be cited. Session 1 adds the status `PARTIAL` (sections listed) and allows PARTIAL papers in Related Work for the parts read. The wording of rule 6 should be updated by the user to match, or this reading confirmed.

Still open:
4. The Session 0 commits used a provisional message format; CLAUDE.md requires `type(scope): summary`. Not rewritten.
5. (resolved by D-025, amended) D-005 says "comparable with published detectors". Published numbers are measured under different protocols (see RESEARCH.md), so "comparable" must be defined in Stage 0 and 2.
6. (resolved: gh is logged in and origin exists) The prompt for Session 1 says the GitHub login was done, but `gh auth status` still reports no login on this machine and there is no `origin` remote. See STATUS.md blockers.
7. (resolved: the folder is in .gitignore, no file of it is tracked, and it is not committed) The folder `Document Template` is inside the repository root, not in its parent folder as the Session 1 prompt describes. It is untracked in git; the PDF and PowerPoint inside it come from an earlier project, so committing them needs the user's confirmation (D-001, D-014).
8. (resolved: mapping D-017 -> D-020 and D-018 -> D-021 confirmed by the user) The Session 1 user instruction numbered the access-policy change D-017 and the data selection D-018, which were already used in this log. They were recorded as D-020 and D-021 (see D-020). The user should confirm.

### D-037: Housekeeping on the user's review of 2026-10-24
- Date: 2026-10-24
- Status: Decided (user)
- Question: Close the open contradictions and confirm the Session 1 decisions that were tagged autonomous or proposed.
- Decision and reason: (1) D-022 (fallback time), D-024 (autonomous items), D-026, D-034, D-035 with amendment 1 and D-036 are Decided (user); D-025 and D-036 item (2) are amended as written in their entries; D-035 gains the permutation-test sensitivity check and the Celeb-DF note. (2) CLAUDE.md rule 6 is changed to allow papers with status PARTIAL for the parts read (the Session 1 practice). (3) `Document Template/` stays in .gitignore and is not committed. (4) The mapping of the user's D-017 and D-018 to D-020 and D-021 is confirmed. (5) GitHub is logged in and origin exists, so contradiction 6 is closed.
- Rejected and why: none.
- Consequences / what could prove this wrong: rule 6 now needs a per-citation statement of which part was read, otherwise it is as loose as the rule it replaced.

### D-038: Encoder shortlist of the O-007 sweep, conditional licences, Xception variant, DINOv3 exclusion
- Date: 2026-10-24
- Status: Decided (user), except the Xception licence wording, which is Decided autonomously - needs user review
- Question: Which verified encoders enter the I-0 sweep (D-036 item 4) and how the open licence points are handled.
- Options considered: the proposed shortlist of docs/O007_SWEEP.md section 3.1 as written; the same with the fallbacks IMG-14 and another audio model; a changed list.
- Criteria: the five criteria of O007_SWEEP.md section 3; every weight logged in DATA_LOG.md with source, revision, licence and hash before use (rule 7).
- Evidence: docs/_agents/O7IMG.md and O7AUD.md (verification from source, 2026-10-24); the merge in O007_SWEEP.md section 3.1.
- Decision and reason: (1) The shortlist is approved: images IMG-01 EfficientNet-B4, IMG-04 Xception (see 3), IMG-07 and IMG-08 DINOv2 small and base, IMG-13 OpenAI CLIP ViT-L/14, IMG-15 LAION CLIP ViT-L/14, IMG-17 SigLIP base, IMG-20 from-scratch reference; audio AU-1 wav2vec2-base, AU-4 WavLM base+, AU-6 Whisper-small encoder, AU-8 HuBERT base, AU-REF from-scratch reference. (2) The two conditional licences, IMG-13 (no licence stated in its repo, card says research use) and AU-4 (no tag, CC BY-SA 3.0 upstream), are accepted as research-only. This is a licence limitation that the paper must state: results with these weights are research-only and no claim of commercial usability is made. (3) On the user's instruction, the original Xception is used if timm ships it with a licence that allows our use. timm 1.0.30 ships it as `legacy_xception` with `license: apache-2.0` in its config; the licence of the upstream Keras weights is NR. It is used as research-only, so IMG-04 replaces IMG-05 and the aligned variant is out; if IMG-05 is ever used it is named "Xception41 (aligned variant)" in every document. (4) DINOv3 is excluded on purpose: its Hugging Face gate is manual and the licence carries an acknowledgement duty. It can be added later as an iteration if the user accepts the access form. The same exclusion is in DATA_LOG.md.
- Rejected and why: fallbacks IMG-14 and a replacement for AU-4 (the user accepted the conditional licences); plain Xception41 (not the architecture the FF++ literature uses).
- Consequences / what could prove this wrong: the research-only label limits where the final model may be reused; if the upstream Keras weight licence turns out to forbid use, IMG-04 is dropped and the sweep rerun for that row only.

### D-039: The LAV-DF tar is kept until the freeze
- Date: 2026-10-24
- Status: Decided (user)
- Question: When the 25.6 GB LAV-DF.tar is deleted after extraction and verification (replaces the "delete after verification" step of the user's instruction of 2026-10-24).
- Options considered: delete as soon as the checks pass; keep until the freeze on 2026-10-29 and delete earlier only if free space on D: drops below 100 GB.
- Criteria: re-extraction must stay possible if a problem shows up late; disk space must not run out (301 GB free when decided).
- Decision and reason: keep the tar until the freeze on 2026-10-29; delete earlier only if free space on D: is below 100 GB. The job (scripts/data/lavdf_job.py) checks free space once at its end and deletes in that case only when the hash, file-count and ffprobe checks passed (an addition of the implementer: otherwise the tar would be the only good copy). Deleting at the freeze is done by hand.
- Rejected and why: automatic deletion after the checks (loses the cheap way to repair a bad extraction).
- Consequences / what could prove this wrong: about 25.6 GB of disk stays in use for five days; if D: fills up through other work, the 100 GB rule triggers a deletion only at the job's end, so free space must be watched by hand afterwards.

### D-040: Definition of a working MVP (iteration I-0)
- Date: 2026-10-24
- Status: Decided (user), approved as written on 2026-10-24
- Question: What exactly counts as "the MVP works", so that the claim can be checked.
- Options considered: a notebook that prints a score; a trained model with metrics only; the full chain from file to web page (this proposal).
- Criteria: it must exercise the whole path a user sees (D-027, D-028, D-031, D-032) and obey the frozen protocol (D-024, D-035).
- Evidence: none yet; the pieces exist separately (loaders, harness, crops, sweep code).
- Decision and reason (proposed): the MVP works when all of the following hold. (1) A file goes in (image, video or audio) through the local web page. (2) The encoder of each task is the winner of the I-0 sweep under the D-036 rule (T-V, T-I, T-A), frozen, with a linear head trained on the training split. (3) The score is calibrated by temperature scaling fitted on validation (D-024) and the checkpoint metadata of D-028 is checked at load (a `preprocess_id` mismatch is refused). (4) Metrics are on the in-dataset validation split and on dev_x, 3 seeds, mean, standard deviation and bootstrap interval, in the ledger; no test split is touched. (5) The web page shows a real result with `inference_source` set to real and the warnings of D-034 (NO_FACE_DETECTED, MULTIPLE_FACES). (6) A video's visual and audio scores are shown separately; no fused score is claimed until O-009 is decided (the implementer's addition). (7) The tests pass and the run is committed with its experiment entry. It makes no accuracy claim beyond the intervals.
- Rejected and why: a score printed in a notebook (does not test the path); metrics without the web page (the deliverable is software).
- Consequences / what could prove this wrong: if the sweep winners tie, the cheapest is used (D-036) and the MVP may be a weak model; that is acceptable for an MVP and is what the later iterations measure. The time estimate is in STATUS.md; it can slip with unmeasured encoder speed, E-002 and E-003 extraction, head training time, the web page not yet built against a real checkpoint, GPU contention with the LAV-DF job, and winners that need a different preprocessing size.

### D-041: Iteration ladder I-0 to I-8 (docs/ITERATIONS.md)
- Date: 2026-10-24
- Status: Decided (user), approved on 2026-10-24 with the six changes listed in the decision text
- Question: Which iterations follow the MVP, in what order, and how each lever is chosen.
- Options considered: a fixed list of levers decided now; levers chosen after each diagnosis within fixed families, with rPPG second-last and freeze last (this proposal).
- Criteria: each iteration must start from a measured diagnosis (CLAUDE.md rules 1 and 2); at least 7 iterations after the MVP (D-016); the freeze is 2026-10-29.
- Evidence: the diagnostic battery (EXPERIMENTS.md) and the derivations (MATH_NOTES.md sections 10 to 17); no results yet.
- Decision and reason (proposed): seven experimental iterations after the MVP (I-1 to I-7) plus I-8, integration and freeze, which runs no experiments. I-1 to I-6 take their lever from the families fine-tuning depth, frame aggregation, loss and class weighting, augmentation and regularisation, combining encoders, late fusion and calibration, chosen after the diagnosis of the previous one; I-7 is the rPPG experiment (O-010) with thresholds written in ITERATIONS.md and a signal-sanity check S-1 started earlier. Changes made by the user on approval: (1) the wording above; (2) verdicts on dev_x are exploratory (sign, size, interval), each frozen iteration is evaluated once on the sealed test sets at the freeze (a staircase), and the confirmatory claims are two pre-declared paired comparisons on sealed data: final versus I-0, and final versus the re-run baseline of D-025 (ITERATIONS.md and PROTOCOL.md); (3) the I-1 budget is redone from measured step times and D-036 item 3 is replaced by D-042; (4) PLAN.md and ITERATIONS.md carry one schedule; (5) the rPPG adopt or do-not-adopt decision is made on dev_x and S-1's shuffled control is pooled across clips with several shuffles per clip; (6) I-6 states the data that trains the heads and the fused label. D-036 item 3 is replaced by D-042.
- Rejected and why: a fixed list now (pre-judges the diagnosis).
- Consequences / what could prove this wrong: six diagnosis-chosen slots in two days is tight; slots that cannot finish by 2026-10-29 are reported as not done, not cut silently. I-1 as specified in D-036 (three encoders, three learning rates, three seeds) may not fit the GPU time.

### D-042: Budget and scope of iteration I-1 from measured step times (replaces D-036 item 3)
- Date: 2026-10-25
- Status: Decided (user) for the structure (video task first, then extra seeds for the winner only, a per-run time cap, a stated list of what does not fit, D-036 item 3 replaced); the cap, the step count and the fit rule below are numbers of the implementer: Decided autonomously - needs user review. The step times are those of M-003, which was partly overlapped with other jobs; the rule is re-checked against the clean rerun (data/_logs/step_time_alone.json) before the first I-1 run.
- Question: How much fine-tuning I-1 can afford on one 12 GB GPU in the time available, and which tasks and encoders it covers.
- Options considered: D-036 item 3 as written (top three of each task, three learning rates, 3 seeds at the best rate, 15 runs per task at about 90 minutes per task, O007_SWEEP.md section 9); video task only with one seed per learning rate and extra seeds for the winner; fewer learning rates.
- Criteria: measured step times (M-003); the budget of Tuesday 2026-10-27 (ITERATIONS.md); one shared budget per task (comparison rule 4); a run must leave the GPU usable by nothing else only when unavoidable.
- Evidence: M-003 at effective batch 32 and 57,600 training frames (3,600 FF++ clips x 16 frames) per epoch: DINOv2-S 700 frames/s (82 s per epoch), DINOv2-B 239 (241 s), Xception 289 (199 s), EfficientNet-B4 227 (253 s), CLIP ViT-L/14 about 58 to 65 frames/s at batch 16 (886 to 990 s) with 8.4 to 9.1 GiB, and 13.75 GiB (spill, 19 s per step) at batch 32.
- Decision and reason: (1) Scope: I-1 fine-tunes the video task (T-V) only. T-I and T-A keep their I-0 frozen heads in I-1 and are fine-tuned later only if time remains (a new decision then). (2) Stage 1: the top three encoders of E-001 that fit, each at the learning rates 1e-5, 3e-5 and 1e-4, one seed (seed 0): 9 runs. Stage 2: seeds 1 and 2 at the best rate of the best encoder only (2 further runs). Other settings as in O007_SWEEP.md section 9 (AdamW, cosine with 5 percent warm-up, weight decay 0.05, mixed precision, effective batch 32, no augmentation). (3) One budget: 1,800 optimiser steps (57,600 frames, one pass over the training frames) for every run. (4) Per-run time cap: 10 minutes of training; a run that would exceed it, or whose evaluation pushes the wall time above 15 minutes, is stopped and recorded as exceeded. (5) Fit rule: an encoder enters I-1 only if its peak VRAM at batch 32 is at most 10 GiB and one budget takes at most 10 minutes. By M-003 this admits DINOv2-S, DINOv2-B, Xception and EfficientNet-B4; the two CLIP ViT-L/14 models (IMG-13 and IMG-15) do not fit (13.75 GiB at batch 32, about 15 minutes per budget even at batch 16, which also blocks the GPU for other jobs). If a CLIP ViT-L/14 is among the top three of E-001, the next ranked encoder that fits takes its place and the substitution is reported; the CLIP model stays frozen (a low-rank adapter variant is a later option, not measured). SigLIP base (IMG-17) is not yet measured; the rule is applied when it is. (6) Expected cost on the measured numbers: nine runs of at most 4.2 minutes of training plus evaluation, about one hour, and two further runs; this is far below the 90 minutes per task of D-036 and leaves room for the evaluation passes. (7) The claim "easiest to fine-tune" (O007_SWEEP.md section 9) is made for the video task and for the encoders that fit only, and is worded that way.
- Rejected and why: D-036 item 3 as written (three tasks at 90 minutes each do not fit the day, and the CLIP models do not fit the card at batch 32); fewer learning rates (the sensitivity measure needs three).
- Consequences / what could prove this wrong: one pass over the frames may under-train the larger models and bias the comparison toward the cheap ones; if the clean rerun changes a step time by more than 20 percent the rule is re-applied; evaluation passes (validation and dev_x, about 39k frames) are extra time not yet measured.

### D-043: The framework that serves the local web page (settles O-013)
- Date: 2026-10-25
- Status: Decided (user), approved 2026-10-25 after three confirmations: the server binds to 127.0.0.1 only (src/dfd/web/__main__.py, no option to change it, tested); uploads are limited to 512 MB (MAX_CONTENT_LENGTH, 413 answer in the D-027 form, tested); uploaded files are deleted after every request, also after an aborted upload, and leftovers are removed at start (tested)
- Question: Which Python framework serves the local web page of D-031 and calls the inference library.
- Options considered: (1) Flask 3.1; (2) FastAPI with uvicorn and python-multipart; (3) the standard library `http.server` with hand-written multipart parsing; (4) Streamlit or Gradio; (5) Bottle; (6) an in-process page without a server (rejected by D-031, which needs a browser).
- Criteria (written before the choice): (a) installs on Python 3.13 with the project's package manager and no compiled build; (b) runs as one process with no extra service (no database, no message broker, no separate front-end build); (c) handles file upload of up to several hundred megabytes without extra code that we must test ourselves; (d) the page is plain HTML, CSS and a little JavaScript that the user can restyle without a build step; (e) few packages, so that the environment stays small; (f) the response of D-027 is returned unchanged as JSON.
- Evidence: installation of Flask into the project environment on 2026-10-25 added 4 packages (flask 3.1.3, werkzeug 3.1.9, blinker, itsdangerous) on top of jinja2, click and markupsafe already present; the multipart parser is werkzeug's.
- Decision and reason: Flask. It meets (a) to (f): one process, upload handling by werkzeug, templates and static files in plain HTML/CSS/JS, no build step, four new packages. Jobs run in a worker thread and the page polls a status route, which gives stage progress without a second service.
- Rejected and why: FastAPI (also meets the criteria, but needs three packages plus an ASGI server and python-multipart, more to explain and no gain for a one-user local page); standard library `http.server` (the `cgi` module was removed in Python 3.13, so multipart parsing would be our own code to write and test, which breaks criterion (c)); Streamlit and Gradio (heavy dependency trees, a fixed look that is hard to restyle, breaks criteria (d) and (e)); Bottle (smaller community and less upload handling than Flask, no benefit).
- Consequences / what could prove this wrong: Flask's built-in server is for local use only; it is bound to 127.0.0.1 and is not a deployment. If the demonstration machine cannot install Flask, fall back to FastAPI behind the same library, because the library does not import the web code. The user can overrule the choice at no cost until the page is restyled.

### D-044: Checkpoint folder format and what the real adapter checks (settles O-012 for checkpoints)
- Date: 2026-10-25
- Status: Decided autonomously - needs user review
- Question: The file format of the configuration and metadata that D-028 left open (O-012), and what the real adapter does at load time.
- Options considered: one JSON metadata file plus one NumPy head file per task; a single pickled object; a framework checkpoint with the metadata inside.
- Criteria: readable without running code (a viva question can be answered by opening the file); no pickle (code execution on load); every D-028 item present; a changed head file or preprocessing must be detected.
- Evidence: D-028 items 3 and 4; the hash identity of a crop folder (params.json, `dfd.data.loaders.preprocess_id`) already exists, so the video `preprocess_id` is the same value as the crop folder's.
- Decision and reason: A checkpoint is a folder with `metadata.json` (version 1: checkpoint_id, iteration, git_commit, protocol, created, seed, library_versions, manifest_ids, split_ids, and per task T-V, T-I or T-A: encoder cid, name, revision, dim and input_size; head file name and SHA-256; preprocess_id and preprocess_config; calibration method, temperature and the split it was fitted on; threshold value and how it was chosen; training data) and one `head_<task>.npz` (mean, std, w, b of the linear head). `src/dfd/infer/checkpoint.py` loads it and refuses: no metadata (NoCheckpointError), missing fields, a head file whose hash differs, a head whose width differs from the encoder's (CheckpointError), and a `preprocess_id` different from the one in use (PreprocessMismatchError, D-028 item 4). The preprocessing ids come from `src/dfd/prep/config.py` (video: the hash of the crop-folder identity; image and audio: small configurations of their own). The real adapter also checks that the encoder weights on disk are the hash-verified ones of the recorded revision. Scoring follows training exactly: 16 frames, YuNet, enlargement 1.3, 224 px, a JPEG quality-95 round trip, mean of frame probabilities, temperature scaling; an image with a face goes through the T-V head and one without through the T-I head, with the path reported in the field `path` (D-033). Audio is scored from the first 64,600 samples at 16 kHz (repeat-padded, `dfd.prep.audio.window_first_repeat`, the same function the extraction uses), the speech encoder's own feature extractor, mean and standard deviation over time, the T-A head and temperature scaling; a checkpoint without a T-A head reports the audio modality unavailable (reason given) and no score is ever invented. Amendment 2026-10-25: the whole-image path (T-I) applies the encoder's own model-card processor to the full-resolution image, exactly as the E-002 extraction does, and the `preprocess_id` of image and audio now describe those rules (the first versions described a plain resize and a centre crop that the extraction does not use).
- Rejected and why: a pickled object (executes code on load); metadata only inside a framework checkpoint (cannot be read without the framework).
- Consequences / what could prove this wrong: tested only with a test double for the encoder and detector and a made-up head; no real checkpoint exists yet, so the first real run (the MVP) may expose a mismatch between training-time and inference-time preprocessing (for example the JPEG round trip or colour order), which the MVP acceptance run must check by scoring a training-split clip both ways.

### D-045: Celeb-DF++ de-duplication, identity exclusion and the resulting test balance
- Date: 2026-10-25
- Status: Decided autonomously - needs user review
- Question: How Celeb-DF++ enters the family `celebdf` without counting any source video or identity of the development slice twice (PROTOCOL.md group-disjoint rule, D-012, D-035, D-020).
- Options considered: keep the 150 byte-identical real videos in both datasets (a duplicate warning only); drop the Celeb-DF++ copies and keep one group id per source video (this decision); drop the Celeb-DF v2 copies instead.
- Criteria: no source video counted twice in any test or dev slice (user instruction 2026-10-25); the reference dataset keeps its official grouping; the test_x of the family stays untouched.
- Evidence: leakage check with content hashing: 150 content hashes occurred twice inside test_x (Celeb-DF v2 and Celeb-DF++ reals); 15 dev_x identities of Celeb-DF v2 appear as target or celebrity source in 2,079 of the 5,078 extracted Celeb-DF++ videos.
- Decision and reason: (1) 2,079 Celeb-DF++ samples are excluded from test_x because their target identity (1,477) or only their celebrity source identity (602, the implementer's addition) is one of the 15 dev_x identities; they are not used as dev_x. (2) 150 Celeb-DF++ real videos whose content equals a Celeb-DF v2 sample are dropped; Celeb-DF v2 keeps them under its own group id. (3) Result: Celeb-DF++ test_x has 2,849 fake videos and no real video in 44 identity groups (split celebdf_pp-3f9df077b9eb); with Celeb-DF v2 the family test_x holds 744 real and 6,906 fake videos. The earlier counts (150 real, 2,849 fake) counted the 150 real videos twice. (4) Any AUC on Celeb-DF++ is computed against the Celeb-DF v2 real videos; pooled analyses use the identity as group across both datasets.
- Rejected and why: keeping the duplicates (a source video counted twice narrows intervals falsely); dropping the Celeb-DF v2 copies (breaks the official Celeb-DF v2 test list).
- Consequences / what could prove this wrong: the test balance is 89 percent fake, so balanced accuracy and the threshold carry more noise than the AUC; the independent units are about 44 identities, so intervals stay wide whatever the video count (DATA_LOG.md). The identity-only exclusion removes about 600 samples that show a dev_x face only as the swap source; a narrower rule (target only) would keep them and is a one-line change if the user prefers it.
