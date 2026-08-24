# Model selection

This document defines how the project selects components. It prevents model
choices based on reputation, one lucky seed, or final-test performance.

## Selection rules

1. Use training data and validation data only.
2. Keep the split, seed set, training budget, and evaluation code fixed.
3. Change one component at a time.
4. Give candidates the same input information and comparable tuning budgets.
5. Run accepted model comparisons with three fixed seeds.
6. Report the mean, spread, confidence interval, runtime, and memory use.
7. Prefer the cheaper candidate when confidence intervals overlap materially.
8. Record rejected candidates and failure reasons.
9. Never reopen a frozen choice after seeing final-test results.

Source-grouped cross-validation is the primary selection evidence. The locked
test partitions measure generalization after selection. They do not select it.

## Candidate matrix

| Stage | Baseline | Challengers | Primary evidence |
|---|---|---|---|
| Face detector | MTCNN | YuNet | Reviewed recall, false detections, landmarks, track stability, and runtime |
| Face tracker | Greedy IoU | Motion-aware association | Identity switches, stable coverage, and abstention rate |
| Visual branch | EfficientNet-B0 plus GRU | ConvNeXt-Tiny; frozen DINOv2 if compute allows | Source-grouped ROC-AUC, worst-method ROC-AUC, calibration, and cost |
| Audio branch | Wav2Vec2 Base | WavLM; AASIST | Cue-specific ROC-AUC, EER, calibration, and cost |
| Sync branch | Current temporal model | Published SyncNet-style baseline; AV-HuBERT if compute allows | Offset accuracy, mismatch detection, temporal localization, and cost |
| Fusion | Calibrated logistic regression | Small MLP | Out-of-fold macro ROC-AUC, Brier score, coverage, and stability |
| Calibration | Platt scaling | Isotonic regression when sample size supports it | Validation Brier score and expected calibration error |

These candidates are a controlled project scope. They are not a claim that no
other model could perform better.

## Detector benchmark

Build the detector review set from training identities only. Sample at least
500 frames from at least 100 clips. Stratify across real and fake clips,
manipulation families, compression, pose, lighting, and multi-person scenes.
Use supplied demographic fields for coverage checks when available.

Annotate the intended speaking face, its box, and five facial landmarks. Record
frames where no suitable face exists. Review a shared subset twice to estimate
annotation disagreement.

Measure:

- Target-face recall at an intersection over union of at least 0.5.
- False detections per frame.
- Five-point landmark error normalized by inter-eye distance.
- Identity switches per 1,000 tracked frames.
- Stable-track coverage and resulting abstention rate.
- Mouth-region jitter after compensating for face motion.
- Median and 95th percentile processing time on the target CPU and GPU.

Reject a detector if its target-face recall is more than one percentage point
below the best candidate. Among the remaining candidates, select the fastest
one that does not worsen identity switches or landmark error materially.
Use landmark-aligned downstream validation as the final tie-breaker.

Published speed figures are background evidence only. Different hardware,
resolutions, and thresholds make them unsuitable for project selection.

## Branch comparison protocol

Use the same frozen view cache unless the view itself is under study. Match the
optimizer search space, early-stopping rule, augmentation budget, and maximum
training steps. Report parameter count, peak memory, and wall-clock time.

Use macro performance across manipulation families as the primary branch
score. Report the worst family beside it. A higher pooled score does not win if
one large family hides a major failure on another family.

If the paired confidence interval includes no meaningful difference, retain
the cheaper or simpler model. Define the meaningful effect size before running
the comparison and store it with the experiment group.

## Fusion comparison protocol

Train every fusion candidate from identical out-of-fold rows. Do not impute a
missing modality probability as real or fake. Keep coverage and abstention in
the result table.

Logistic regression remains the primary fusion model unless the MLP improves
the predeclared validation target across all three seeds. A single best seed
does not justify the MLP.

## Decision record

Every accepted comparison must record:

- Question and candidates.
- Git commit and environment lock hash.
- Dataset, split, preprocessing, and feature-store hashes.
- Run IDs and fixed seeds.
- Primary and secondary metrics with intervals.
- Runtime and peak memory.
- Selected candidate and exact reason.
- Known weaknesses and rejected alternatives.

Store full runs in MLflow after its integration. Keep the short accepted
decision in this document or a later architecture decision record.

## Background sources

- [facenet-pytorch MTCNN](https://github.com/timesler/facenet-pytorch)
- [OpenCV YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- [DINOv2](https://arxiv.org/abs/2304.07193)
- [WavLM](https://huggingface.co/docs/transformers/main/model_doc/wavlm)
- [AASIST](https://arxiv.org/abs/2110.01200)
- [AV-HuBERT](https://arxiv.org/abs/2201.02184)
