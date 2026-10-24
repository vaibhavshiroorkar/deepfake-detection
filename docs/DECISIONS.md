# DECISIONS

Last updated: 2026-10-24 (Session 0)

This document records every decision taken in the project, why it was taken, what evidence supported it and what could prove it wrong. It is written to be read by a reviewer with a strong mathematical background, so claims are tied to stated criteria and to measurable evidence. A decision without evidence is labelled as such.

## How decisions are made

### Entry template

Every decision uses this template. Fields are never left out; "none yet" is written where there is nothing to say.

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

When a decision is replaced, the old entry is not edited or deleted. Its status changes to `Superseded by D-NNN` and the new entry cites it. This keeps the history auditable: a reader can see what was believed at each date and why it changed.

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

## Decisions taken by the user

All ten entries below were decided by the user on 2026-10-24 and were given to the implementer in the Session 0 prompt. The prompt did not state the reasons or the rejected alternatives, so those fields say so rather than inventing them. They should be confirmed with the user and filled in.

### D-001: Build the whole system from zero in this repository
- Date: 2026-10-24
- Status: Decided (user)
- Question: Whether to reuse earlier work or start again.
- Options considered: Reuse earlier code or text; build from zero.
- Criteria: Not recorded.
- Evidence: none yet
- Decision and reason: Build the whole system from zero in this repository. No code or text is carried over from earlier work. Reason not recorded in the prompt.
- Rejected and why: Reuse of earlier work. Reason not recorded.
- Consequences / what could prove this wrong: All components must be built and verified here, which costs time. Session 0 observed that the repository folder was empty at the start, which is consistent with this decision.

### D-002: Input scope is images, videos and audio
- Date: 2026-10-24
- Status: Decided (user) for the scope; how each input type is handled is Open (see O-002)
- Question: Which input types the system must accept.
- Options considered: Images only; images and video; images, video and audio.
- Criteria: Not recorded.
- Evidence: none yet
- Decision and reason: The scope is images, videos and audio. Audio is needed because analysing a video requires it. How image, video and audio inputs are each handled is not decided; it is logged as O-002.
- Rejected and why: Narrower scopes. Reason not recorded.
- Consequences / what could prove this wrong: A wide scope divides the effort between modalities. Datasets with audio labels are fewer and smaller than visual ones (see RESEARCH.md, R-013 and R-014), which could limit what claims the audio branch can support.

### D-003: Deliverables
- Date: 2026-10-24
- Status: Decided (user)
- Question: What the project must produce.
- Options considered: Not recorded.
- Criteria: Not recorded.
- Evidence: none yet
- Decision and reason: The deliverables are the software, an IEEE-format research paper (LaTeX, written in Overleaf), a technical report and a presentation, all produced in this repository.
- Rejected and why: Not recorded.
- Consequences / what could prove this wrong: The paper and report are assembled from the living documents, so the logs must be complete and honest from the first session.

### D-004: Method is MVP first, then at least 7 hypothesis-driven iterations
- Date: 2026-10-24
- Status: Decided (user)
- Question: How the work is organised.
- Options considered: Not recorded.
- Criteria: Not recorded.
- Evidence: none yet
- Decision and reason: Build the simplest MVP first, then at least 7 further iterations. Each iteration starts from a written hypothesis and ends with a measured result, including failures. Candidate models are run in the same pipeline from the MVP stage so that the final choice has supporting evidence.
- Rejected and why: Not recorded.
- Consequences / what could prove this wrong: The comparison rules above must hold from the MVP onward, so the shared pipeline and evaluation protocol (Stages 1 and 2) must exist before the MVP.

### D-005: Goal is comparability on our own protocol, not the best model
- Date: 2026-10-24
- Status: Decided (user)
- Question: What counts as success.
- Options considered: Best-in-world performance; documented path to results comparable with published detectors.
- Criteria: Not recorded.
- Evidence: none yet
- Decision and reason: Show a documented path to results comparable with published detectors on a protocol we define and run ourselves. The goal is not the best model in the world.
- Rejected and why: Best-in-world performance as a goal. Reason not recorded.
- Consequences / what could prove this wrong: Published numbers come from different protocols and are not directly comparable (RESEARCH.md shows this repeatedly). "Comparable" therefore has to be defined precisely in Stage 0 and 2, for example by re-running a published detector on our protocol rather than quoting its paper's number.

### D-006: rPPG is a candidate improvement, not an adopted component
- Date: 2026-10-24
- Status: Decided (user)
- Question: Whether remote photoplethysmography (rPPG, the heartbeat-in-skin-colour idea behind Intel FakeCatcher) becomes part of the system.
- Options considered: Adopt; do not adopt; evaluate first.
- Criteria: Adoption depends on a feasibility test in Stage 6.
- Evidence: Literature is mixed (R-005 to R-010). Older work reports high accuracy on 2019-era data; 2025 and 2026 papers report that modern face-swap fakes can carry a plausible pulse (R-008) and that an older rPPG detector degrades from AUC 0.999 to 0.622 on newer talking-face fakes in one reproduction (R-009).
- Decision and reason: rPPG will be evaluated as a candidate improvement. It is not adopted. Adoption depends on the Stage 6 feasibility test.
- Rejected and why: Immediate adoption, because the literature does not support assuming it works on modern fakes.
- Consequences / what could prove this wrong: If the feasibility test fails, rPPG is reported as a negative result and not included. See O-010.

### D-007: Hardware is local and fine-tuning is local
- Date: 2026-10-24
- Status: Decided (user)
- Question: Where training runs.
- Options considered: Local GPU; rented cloud GPU.
- Criteria: Not recorded.
- Evidence: ENV-DESKTOP-GPU in EXPERIMENTS.md (measured 2026-10-24).
- Decision and reason: Hardware is local; fine-tuning is done locally; plans must fit inside the measured VRAM. Measured: one NVIDIA GeForce RTX 5070 with 12 GB (about 11 GiB usable while the desktop is running).
- Rejected and why: Cloud GPUs. Reason not recorded.
- Consequences / what could prove this wrong: Large backbones or long video clips may not fit. Mixed precision reduced peak memory by about a third in the sizing probe, which helps, but each candidate must be measured (cost axes).

### D-008: Hosting is GitHub with small commits and a tag per frozen iteration
- Date: 2026-10-24
- Status: Decided (user)
- Question: Where the code and documents are hosted and how history is kept.
- Options considered: Not recorded.
- Criteria: Not recorded.
- Evidence: none yet
- Decision and reason: Hosting is GitHub, committed in small steps, tagged per frozen iteration.
- Rejected and why: Not recorded.
- Consequences / what could prove this wrong: At the end of Session 0 no git remote exists and the GitHub CLI is not logged in, so nothing has been pushed yet. This is listed as a blocker in STATUS.md.

### D-009: Timeline
- Date: 2026-10-24
- Status: Decided (user) for the principle; exact dates Open
- Question: How long the project may take.
- Options considered: Not recorded.
- Criteria: Not recorded.
- Evidence: none yet
- Decision and reason: More than two weeks are available and the work should finish as soon as is sound. Exact dates are open.
- Rejected and why: Not recorded.
- Consequences / what could prove this wrong: Without dates, scope has to be controlled by the stage gates in PLAN.md. The user should supply a submission date so that the plan can be checked against it.

### D-010: Documentation method
- Date: 2026-10-24
- Status: Decided (user)
- Question: How progress, decisions and evidence are recorded.
- Options considered: Not recorded.
- Criteria: Not recorded.
- Evidence: none yet
- Decision and reason: Use the living-document set listed in CLAUDE.md. The implementer found no CLAUDE.md in the repository during Session 0, so the set used is the eight documents named in the Session 0 prompt: PLAN, DECISIONS, FILES, EXPERIMENTS, DATA_LOG, RESEARCH, VIVA_PREP and STATUS. This needs the user's confirmation.
- Rejected and why: Not recorded.
- Consequences / what could prove this wrong: If CLAUDE.md contains rules (for example a commit message format) that differ from what was assumed, earlier commits will need to be reconciled.

## Open decisions

None of the questions below is answered. For each one, the options are those the implementer would compare, the criteria are stated before any result exists, and the evidence column says what must be collected. Paper IDs refer to RESEARCH.md. Pointers to papers show where options come from; they are not choices.

| ID | Question | Options to compare | Criteria (stated before results) | Evidence needed |
|---|---|---|---|---|
| O-001 | Exact problem statement and output | (a) One real/fake score per input. (b) Per-modality scores plus a fused score. (c) Per-modality scores, a fused score and localisation of manipulated time segments (R-015 does temporal localisation). (d) Add a label for which modality is fake (FakeAVCeleb separates the cases, R-014). | The output must be evaluable with labels that exist in the chosen datasets; must support the claims in D-005; must be implementable within the cost axes and time. | Which labels each candidate dataset provides (DATA_LOG.md); what published detectors output; what the user wants to demonstrate. |
| O-002 | How image-only, video and audio inputs are each handled, and which claims each supports | Images: face crop then classifier (as R-001) versus whole-image detector (as R-011). Video: per-frame scores aggregated over time versus a clip-level temporal model. Audio: waveform model (R-012) versus spectral-feature model. Cross-modal: consistency between audio and lips (R-015). | A claim is made only for the kind of input the training and test data cover: a detector trained on face-swap video makes no claim about fully generated images or cloned voices without evidence. Each path must have a held-out test with its own protocol. | The generators present in each dataset; cross-dataset results from the literature (R-004, R-011, R-017) showing how fragile transfer is. |
| O-003 | Datasets to use, and which are for training, validation and held-out testing | Visual: FaceForensics++ (R-001), Celeb-DF (R-002), DFDC (R-003), and others on the reading list. Audio: ASVspoof 2019 LA (R-013). Audio-visual: FakeAVCeleb (R-014), DFDC (has audio manipulation in some clips, R-015). | Licence and access terms permit our use; size fits the 404 GB free on D:; identities or source videos are available so that splits can be disjoint; labels match O-001; generators are diverse and not all outdated; results are comparable with published work. | Access terms and licences (not yet checked, to be recorded in DATA_LOG.md); sizes; counts of identities; what each dataset's license says about redistribution. |
| O-004 | Split and leakage policy | (a) Random frame split (listed only as a known-leaky reference). (b) Video-level split. (c) Source-video-disjoint (a real video and all fakes made from it stay together). (d) Identity-disjoint (R-009 and R-007 do this). (e) Generator-disjoint (cross-manipulation, R-004). (f) Dataset-disjoint (cross-dataset, R-002, R-017). | No identity, source video, or near-duplicate frame appears in more than one partition; every split is stored as a file with an ID and a hash; the policy is tested by an automatic check. | Identity and source metadata of each dataset; a leakage check script (not written in Session 0); whether published splits are identity-disjoint (R-001 uses a fixed video split; whether identities overlap is not established). |
| O-005 | Metrics, threshold policy, calibration | Ranking metrics: ROC-AUC, average precision, EER. Threshold metrics: accuracy at a fixed threshold, balanced accuracy. Probability metrics: log loss, Brier score, ECE (R-016). Audio: min t-DCF with EER (R-012, R-013). Threshold policy: fixed 0.5; threshold chosen on validation data; EER threshold on validation. Calibration: none; temperature scaling (R-016). Aggregation: frame-level versus video-level. | The threshold is chosen only on validation data and never on test data (R-011 shows that tuning the threshold on the test set gives an "oracle" number that is not achievable in practice). Class balance in the test set is stated. A metric must be defined mathematically in the document before use. | Class ratios of the chosen datasets; seed-to-seed variance, which can be large (R-012 reports a spread of EER between 1.19 and 2.06 percent over seeds for one baseline); how the DFDC organisers used log loss (R-003). |
| O-006 | MVP design: the simplest thing that runs end to end | (a) Small CNN from scratch on face crops, one dataset. (b) Frozen pretrained features with a linear classifier (R-011 style). (c) Fine-tuned pretrained backbone. (d) Audio: small CNN on a spectrogram. | It runs from raw input file to a score and a metric under the Stage 2 protocol; it needs no tuning to produce a number; it trains within minutes on the measured machine; it is simple enough to explain. | The Stage 2 protocol; measured data-loading speed (not yet measured); a first sizing of each candidate. |
| O-007 | Candidate backbones per modality, including a small network built from scratch as a reference | Visual: small CNN from scratch; ResNet family; Xception and EfficientNet-B4 (both used as strong plain baselines in R-017); frozen CLIP-type encoder with a linear or nearest-neighbour classifier (R-011). Audio: small CNN on a spectrogram; a lightweight model such as AASIST-L (85K parameters, R-012). | The cost axes in this document, measured on one machine under the comparison rules. | Experiments in Stage 4. Prior evidence: in one benchmark a plain Xception was within roughly one to two AUC points of more elaborate detectors under identical settings (R-017), which suggests the backbone and training recipe may matter more than the method name. |
| O-008 | Fine-tuning strategy | Linear probe; full fine-tuning; partial unfreezing of the last blocks; differential learning rates; low-rank adapters (not yet read). | Validation metric at a fixed budget; peak VRAM; stability across seeds; sensitivity to learning rate. | Stage 4 and 5 runs. Prior evidence: ImageNet accuracy correlates with transfer accuracy for both fixed features and fine-tuning across 12 natural-image tasks (R-018), but deepfake data is not natural-image classification, so this must be tested here. |
| O-009 | Fusion strategy | Late fusion of per-modality scores (average or logistic regression); feature-level fusion; cross-modal consistency model (R-015); joint training (R-020, to be read). | Must handle inputs with a missing modality (a still image has no audio); must be at least as good as the best single modality on the validation data; must be explainable. | Per-modality baselines first. Prior evidence: simple ensembles of unimodal detectors did poorly on FakeAVCeleb in the dataset paper (R-014), which is a reason to test fusion carefully. |
| O-010 | rPPG feasibility test design | (a) Signal sanity: can heart rate be recovered from our data at all, given that most deepfake datasets have no pulse ground truth (R-008). (b) Separation: do rPPG features separate real from fake under an identity-disjoint split, for several generator families (R-009, R-007). (c) Added value: does adding rPPG evidence raise the fused detector's held-out metric beyond the bootstrap interval. (d) Negative controls: time-shuffled or mismatched signals. | The success and failure thresholds are written before the run. rPPG is adopted only if (c) shows a gain larger than the interval of the paired difference. | Stage 6 experiments. Literature: R-005 to R-010. A dataset subset with enough frames, resolution and lighting stability for rPPG; skin-tone coverage (R-009 notes that rPPG is sensitive to skin tone and lighting). |
| O-011 | Software form | Command line; local web app; both. | Time cost; need to demonstrate to examiners; ease of reproducing results; input handling for image, video and audio files. | Requirements from Stage 0. |

## Contradictions and unclear points found in Session 0

These are recorded here so that they are visible to the user. They are not decisions.

1. The Session 0 prompt says to start by reading CLAUDE.md, which it calls binding, and refers to a commit message format and a document set "listed in CLAUDE.md". No CLAUDE.md exists in the repository, and the repository folder was empty and was not a git repository. A provisional commit message format was used (see STATUS.md).
2. The prompt describes the CPU as a Ryzen "7900K". The measured CPU is an AMD Ryzen 9 7900X. This does not affect the plan.
3. D-002 says audio is needed "because analysing video requires it", and also says how each input type is handled is not decided. These are consistent only if "needed" means "in scope", which is how this document reads it.
4. D-005 says "comparable with published detectors". Published numbers are measured under different protocols (see RESEARCH.md), so "comparable" must be defined in Stage 0 and 2.
5. The prompt asks the implementer to "tag setup" and push, but no GitHub account is logged in on the machine. See STATUS.md blockers.
