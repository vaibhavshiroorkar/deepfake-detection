# Predictions recorded before the results

Written before the measurements they refer to, so the next round is scored
against what was claimed rather than what can be rationalised afterwards. Each
row names the artifact that will settle it.

The reason this file exists: predictions made during this project have been wrong
often enough to need a record. Freezing BatchNorm was called a clean win and was
a trade. Motion was claimed to cause false alarms and does the opposite. A lower
encoder learning rate was expected to fix the lip-sync collapse and changed
nothing. Training on FaceForensics++ was expected to improve transfer and made it
worse. Four wrong calls is a reason to write the next ones down.

## Leave-one-family-out, five arms still running

Settled by `runs/ffpp-20260913/evaluation/by-family-unseen-*.json`. The seen
column is measured. The unseen column is the prediction, made after seeing only
the Deepfakes arm.

| Family | Seen, measured | Unseen, predicted | Reasoning |
| --- | --- | --- | --- |
| Deepfakes | 0.8912 | **0.7771 measured** | Settled. Drop of 0.1141, intervals do not overlap. |
| Face2Face | 0.8600 | **0.7839 measured**, predicted 0.74 to 0.78 | Point estimate landed just above the band; the interval covers it. |
| FaceShifter | 0.8531 | **0.7077 measured**, predicted 0.76 to 0.80 | **Wrong, and the reasoning was wrong.** Category membership does not predict transfer. FaceShifter is designed for better blending, so it does not leave the seams the older swaps leave, and sharing the "swap" label bought nothing. Fifth wrong prediction in this project. |
| FaceSwap | 0.8110 | 0.68 to 0.74 | Graphics-based rather than learned, so the least like anything left in training. |
| NeuralTextures | 0.8486 | 0.66 to 0.73 | Subtlest family, usually the hardest in published tables. |
| DeepFakeDetection | 0.9817 | above 0.92 | **This is a shortcut test, not an accuracy prediction.** It is the Google actor set, filmed separately from the 1,000 YouTube originals. If an arm that never saw it still scores above 0.92, the model is separating it on capture conditions rather than on manipulation, and that margin is not detection skill. A fall to the 0.75 range would mean the opposite. |

Predicted mean drop across the five unseen arms: 0.10 to 0.13. Measured mean
across the three arms that ran: 0.1119.

The NeuralTextures and DeepFakeDetection arms were not run. The sweep was
stopped after three arms because FaceForensics++ carries no audio and the
project's focus is audiovisual. The DeepFakeDetection prediction stands as an
open shortcut test for whenever it runs.

## Leave-one-generator-out on DF26, the only metric that counts

Predicted before running, under the protocol where the scored generator is never
in training.

| Claim | Prediction |
| --- | --- |
| Macro mean across seven unseen DF26 generators | 0.78 to 0.86 |
| Best single unseen generator | above 0.93 |
| Worst single unseen generator | below 0.65 |
| Spread between best and worst | at least 0.25 |
| Any unseen generator above 0.99 | would indicate a leak, not success |

The spread prediction is the one worth checking. Per-generator collapse is the
norm in this problem, and a narrow spread would be more surprising than a low
mean.

## The wider system

| Question | Prediction | Settled by |
| --- | --- | --- |
| Cross-corpus DFDC after the architecture rebuild | No better than 0.60, and 0.55 is the honest central estimate, unless capture-diverse training data is added. Every intervention tried so far failed to move it. | A rebuilt system scored on the same 2,410 clips |
| A generated-media specialist on seen generators | Above 0.95. Detecting generators you trained on is not the hard problem. | Leave-one-generator-out arm, seen column |
| The same specialist on a fully held-out generator | 0.65 to 0.85, and the interval will be wide. Published per-generator results range from 0.73 on Sora down to 0.15 on Kling for a detector that scored 0.70 on another generator. | Held-out generator arm |
| In-domain FakeAVCeleb after laundering-matched training | Falls from 0.9990 to between 0.95 and 0.98, and this is not a regression. Part of the current number is the compression shortcut, measured at 86 percent against 23 percent between re-encoded and raw clips. | Rebuilt system on the same test partition |
| Calibration error cross-corpus | Falls from 0.6366 to below 0.15 with per-branch calibration and reliability weighting. This is the easiest win on the list. | ECE on the DFDC partition |
| Abstention rate on generated media | Above 60 percent of clips flagged out of distribution before any specialist is trained, using the motion gate alone. | The distribution gate on a generated-clip set |

## What would falsify the architecture's central claim

The claim is that Level 1 should rest on capture-chain statistics rather than on
cross-modal consistency. It is wrong if a cross-modal branch, on jointly
generated media, separates real from synthetic better than the forensic residual
branch does. That comparison needs generated audiovisual clips and is the first
experiment to run once they exist.
