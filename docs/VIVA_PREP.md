# VIVA_PREP

Last updated: 2026-10-24 (Session 1) (Session 0)

A bank of questions an examiner may ask, grouped by theme. Each question is linked to the decision (D or O), experiment (E) or paper (R) that will answer it. Answers marked "pending evidence" stay that way until the evidence exists; nothing here is a prepared claim. Section 8 holds standard mathematical definitions, which do not depend on experiments, with the source used.

## 1. Models and backbones

| # | Question | Answered by | Answer |
|---|---|---|---|
| 1.1 | Why this backbone and not others? | O-007, Stage 4 comparison table | pending evidence |
| 1.2 | Why not build our own network? | O-007 (a small network from scratch is included as a reference), R-017 | pending evidence |
| 1.3 | What did other models score and how were they trained? | R-001 to R-004, R-011, R-017; our re-runs under D-005 | partly available in RESEARCH.md; our own comparison pending evidence |
| 1.4 | Could the other models have been trained better? | R-017 (recipe matters), our cost-axis study | pending evidence |
| 1.5 | How much compute does the chosen model need? | Cost axes in DECISIONS.md, EXPERIMENTS.md | pending evidence |

## 2. Training, fine-tuning and loss

| # | Question | Answered by | Answer |
|---|---|---|---|
| 2.1 | Why this fine-tuning method? | O-008, R-018, R-011 | pending evidence |
| 2.2 | Why this loss function? | Not yet an open decision; to be added when the MVP design (O-006) fixes it | pending evidence |
| 2.3 | How sensitive is the result to the learning rate and the random seed? | Comparison rules, R-012 (seed variance example) | pending evidence |
| 2.4 | Why mixed precision? | ENV-DESKTOP-GPU sizing probe (memory and speed measured) | The probe showed about 34 percent less peak memory and about 42 percent less time per step for a tiny network; whether it affects accuracy for the real model is pending evidence |

## 3. Data, splits and leakage

| # | Question | Answered by | Answer |
|---|---|---|---|
| 3.1 | Why this split? | O-004 | pending evidence |
| 3.2 | How do you know there is no leakage? | O-004, the leakage check script (not yet written), DATA_LOG.md | pending evidence |
| 3.3 | Why these datasets and what are their licences and biases? | D-021, D-020, DATA_LOG.md, R-002, R-014 | Selection rule is on record: a dataset is kept only if it serves a numbered requirement (D-021). Licence terms are in DATA_LOG.md. Bias analysis is pending evidence |
| 3.4 | Why is the test set only used once? | DECISIONS.md rule on frozen choices | The rule is stated in DECISIONS.md; its application is pending evidence |
| 3.5 | What is the difference between in-dataset and cross-dataset results and why does it matter? | RESEARCH.md notes, R-017 | Available: in R-017 the mean AUC of strong detectors is above 0.93 within FF++ and about 0.77 to 0.79 across datasets. Our own numbers are pending evidence |
| 3.6 | Why FaceForensics++ as the shared training set? | D-021, R-017 | The usual protocol in the literature trains on FF++ c23 and tests on other datasets (R-017), so our numbers can be set beside published ones (D-005). Whether it worked here is pending evidence |
| 3.7 | Why did you drop datasets such as OpenForensics, ASVspoof 5, GenImage or DFDC? | D-021 keep, drop or defer table | Each was dropped or deferred because no numbered requirement needed it, or because of size or an unverified access route; the one-line reasons are in DATA_LOG.md |
| 3.8 | Two of your audio-visual datasets take real videos from VoxCeleb2. Could identities appear in both training and test? | D-012, D-021, the leakage-check script (not yet written) | pending evidence: the identity overlap between LAV-DF and FakeAVCeleb is to be checked before any dataset-disjoint claim |
| 3.9 | Your generated-image training set is only 8,000 images. Is that enough and is it comparable with the literature? | D-021 consequences, R-011 | It is far smaller than the 720,000 images of R-011, so that paper's table is not reproduced; adequacy is pending evidence (a training-size comparison would answer it) |
| 3.10 | Can you show example frames in the paper? | D-020, DATA_LOG.md rule 4 | Celeb-DF and FakeAVCeleb forbid publishing derived data; what may be shown is to be settled before drafting figures |

## 4. Metrics, threshold and calibration

| # | Question | Answered by | Answer |
|---|---|---|---|
| 4.1 | Why this metric, and what are its mathematical definitions? | O-005, section 8 below | Definitions in section 8; the choice is pending evidence |
| 4.2 | What do ROC-AUC, EER, ECE, the Brier score and a bootstrap interval measure, and how are they computed? | Section 8 below | See section 8 |
| 4.3 | Why 0.5 as the threshold? | O-005, R-011 (threshold tuning), R-003 (log loss near 0.69) | pending evidence (a 0.5 threshold is justified only if scores are calibrated and the classes are balanced) |
| 4.4 | Is the model calibrated? | O-005, R-016 | pending evidence |
| 4.5 | How wide is the uncertainty on your reported numbers? | Comparison rules, bootstrap procedure fixed in Stage 2 | pending evidence |

## 5. Modalities and fusion

| # | Question | Answered by | Answer |
|---|---|---|---|
| 5.1 | What does the system claim for an image, a video and an audio file separately? | O-001, O-002 | pending evidence |
| 5.2 | Why this fusion strategy? | O-009, R-014, R-015 | pending evidence |
| 5.3 | What happens when a modality is missing? | O-009 | pending evidence |

## 6. rPPG

| # | Question | Answered by | Answer |
|---|---|---|---|
| 6.1 | Why consider rPPG at all? | D-006, R-005 | The idea comes from FakeCatcher (R-005): the claim that fake faces do not carry a consistent pulse. Whether this holds for modern fakes is the subject of the feasibility test |
| 6.2 | What did rPPG show on our data? | O-010, Stage 6 experiments | pending evidence |
| 6.3 | Why was rPPG adopted or not adopted? | D-006 and the Stage 6 decision | pending evidence |
| 6.4 | Could an attacker defeat a pulse check? | R-008, R-010 | Literature: modern face swaps can carry a plausible pulse (R-008) and pulse signals can be forged deliberately (R-010). Our own test is pending evidence |

## 7. Limits and process

| # | Question | Answered by | Answer |
|---|---|---|---|
| 7.1 | What are the limits of the system? | Stage 7 report, all experiments | pending evidence |
| 7.2 | What does the system not detect? | O-002 and the held-out generator tests | pending evidence |
| 7.3 | Is there demographic bias? | Subgroup evaluation, R-009 (skin tone caveat), R-014 (ethnicity balance) | pending evidence |
| 7.4 | What went wrong during the project and what was corrected? | STATUS.md, Mistakes and corrections | Session 0 entries exist; later sessions pending |
| 7.5 | Why is "comparable with published detectors" a fair claim? | D-005 and how comparability is defined in Stage 0 and 2 | pending evidence |
| 7.6 | What would you do with more time or compute? | D-007, final report | pending evidence |

## 8. Standard definitions (background, not experiment-dependent)

Notation: n test examples with labels y_i in {0, 1} (1 = fake) and model scores s_i in [0, 1] for the fake class. Let n_1 be the number of fakes and n_0 the number of real examples.

- ROC-AUC. The probability that a randomly chosen fake receives a higher score than a randomly chosen real example, counting ties as one half: AUC = (1 / (n_1 n_0)) * sum over fake i and real j of [ 1(s_i > s_j) + 0.5 * 1(s_i = s_j) ]. It equals the area under the curve of true positive rate against false positive rate as the threshold sweeps from 1 to 0. It uses ranking only, so it does not depend on calibration and is unchanged by any increasing transformation of the scores.
- Equal error rate (EER). The value of the false positive rate (equal to the false negative rate) at the threshold where those two rates are equal. It is read from the ROC curve and summarises ranking quality at one operating point. The threshold at which it occurs depends on the data it is computed on.
- Log loss. The mean of -[ y_i ln p_i + (1 - y_i) ln(1 - p_i) ], where p_i is the predicted probability of fake. A model that always says 0.5 scores ln 2, about 0.693; R-003 uses this as the no-information level.
- Brier score. The mean of (p_i - y_i)^2 over the test set. It is a proper scoring rule, so it is minimised in expectation only by reporting the true probabilities. The definition is standard; the original source (Brier, 1950) has not been read in this project.
- Expected calibration error (ECE), as defined in R-016. Divide the confidence range into M equal-width bins B_1, ..., B_M, where the confidence of an example is the predicted probability of its predicted class. Then ECE = sum over m of (|B_m| / n) * | acc(B_m) - conf(B_m) |, where acc is the fraction of correct predictions in the bin and conf is the mean confidence. R-016 uses M = 15. ECE depends on M and on how the bins are formed.
- Temperature scaling, as in R-016. Replace the logits z by z / T with a single scalar T > 0 chosen to minimise the negative log likelihood on a validation set. It does not change which class is predicted, so accuracy and AUC are unchanged; only the confidence changes.
- Bootstrap confidence interval. Resample the test units (identities or source videos, not frames) with replacement B times; recompute the metric on each resample; the interval is formed from the lower and upper quantiles of the B values, for example the 2.5th and 97.5th percentiles for a 95 percent interval. A paired comparison of two models uses the same resampled units for both and takes the interval of the difference of their metrics. B and the level are fixed in Stage 2.
- Minimum tandem detection cost function (min t-DCF), used in audio anti-spoofing (R-012, R-013). It measures the cost of a detector when it is used together with a speaker verification system, so it combines the detector's errors with the verification system's. It is not used for visual detection.
