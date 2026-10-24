# O-007 SWEEP (iteration I-0)

Status: DESIGN, awaiting the user's approval of the shortlist and the rule. Written 2026-10-24, before any feature extraction. Nothing in this file is a result. The hypotheses and criteria are in EXPERIMENTS.md as E-001 to E-003 and are not edited after the run.

## 1. What the sweep decides

D-032 fixes the form of the MVP: frozen pretrained features with a linear head. O-007 asks which encoder. The sweep is a measured comparison of the candidates under one protocol, per task. It chooses which candidates enter the fine-tuning iteration I-1 (section 8); it does not choose the final model, and it makes no claim about fine-tuned performance.

## 2. Tasks and data (D-012, D-021, D-034, D-035)

| Task | Train | Val (threshold, temperature, early stopping) | Dev slice for ranking (`dev_x`) | Input to the encoder |
|---|---|---|---|---|
| T-V: face video frames | FF++ c23 official train | FF++ official val | Celeb-DF v2, 15 of 59 identity groups (D-035) | The D-034 crops: 16 frames per video, YuNet, enlarged 1.3, 224 by 224, JPEG 95 |
| T-I: generated images | ProGAN validation set, 14 of 20 classes | the other 6 classes | 2 of the 8 CNN synthesis generators (D-035) | Whole images (no face detection), resized by the encoder's own rule |
| T-A: speech audio | ASVspoof 2019 LA train | ASVspoof 2019 LA dev | In-the-Wild, 12 of 58 speakers (D-035) | First 4 seconds (64,600 samples at 16 kHz), shorter clips repeat-padded; the Whisper encoder gets its fixed 30 second window with the 4 seconds first and only the frames of those 4 seconds pooled |

The final test sets (`test_in`, `test_x`) are not touched. Every split comes from the split files of D-012 and D-035; the leakage check must pass before extraction.

Class balance: FF++ has four fakes per real video and ASVspoof about nine spoofed per bona fide utterance, so the head uses class-balanced loss weights. This is a declared setting, not a tuned one.

## 3. Shortlist (criteria first, candidates verified from source)

Criteria, applied to every candidate: (1) the licence is verified from the weights' own repository (model card and LICENSE text), and whether use is commercial-allowed or research-only is stated; (2) weights are downloadable and usable offline (file names, sizes, SHA-256, a loading library that installs on Python 3.13 with torch 2.14.1); (3) full inference fits in 12 GB, and full fine-tuning is plausible on 12 GB or marked as not; (4) feature size is stated with the exact tensor; (5) the families differ. Anything not verified from the source is NR and does not enter the shortlist.

Seed candidates (to be replaced by what the verification finds): images: EfficientNet, Xception, DINOv2, DINOv3 only if its licence permits our use, OpenAI CLIP ViT-L/14 (the encoder of R-011) and one open-licence CLIP-style alternative, plus a small network trained from scratch as a reference; audio: wav2vec 2.0, WavLM, a Whisper encoder, plus a small network trained from scratch as a reference.

The verified table is in docs/_agents/O7IMG.md and docs/_agents/O7AUD.md and is merged below once both are checked.

SHORTLIST TABLE: pending verification.

## 4. Feature extraction (once per candidate and task)

- The encoder runs in evaluation mode with mixed precision, no augmentation and no test-time augmentation, using the preprocessing stated on its own model card (resize, normalisation).
- The feature is the standard pooled embedding before any classifier head: the global-average-pooled map for the CNNs, the class-token embedding for DINOv2 and DINOv3, the image embedding for the CLIP-style encoders, and the mean over time of the last hidden layer (mean and standard deviation concatenated) for the audio encoders. Using other layers is a later hypothesis, not part of I-0.
- Features are stored once as 16-bit floats under data/_features/<candidate>/<task>/ with an index to the manifest sample IDs. The same crops and the same split files are used for every candidate.

## 5. The head

A single linear layer on features standardised with the training mean and standard deviation. Binary cross-entropy with class weights; AdamW with learning rate 1e-3 and weight decay 1e-4; batches of 1,024 features; at most 200 epochs with early stopping on the validation AUC (patience 10). Three seeds (0, 1 and 2) change the initialisation and the batch order. No other hyperparameter is tuned in I-0.

## 6. Metrics and uncertainty (D-024)

The ranking metric is the clip-level AUC on `dev_x` (video: mean of the frame probabilities, D-024; image: per image; audio: per clip), averaged over the 3 seeds, with the 2000-resample 95 percent group bootstrap and the multi-seed rule of PROTOCOL section 6. Also reported: the in-dataset validation AUC and the gap to the dev AUC, EER, balanced accuracy at the validation-chosen threshold, and ECE and Brier before and after temperature scaling. The resampling unit is the identity, speaker, class or generator group.

## 7. Cost record for every candidate

Parameters; FLOPs of one forward pass at the native input size (with the PyTorch FLOP counter); peak VRAM during inference at batch 64 for images and batch 32 of 4 second clips for audio, in half precision; extraction time per 1,000 images or clips, as model-only throughput on preloaded batches and as end-to-end including decoding (median of 3 repeats); feature size; licence (commercial-allowed or research-only); weights SHA-256.

## 8. Selection rule

Per task, rank the candidates by the mean `dev_x` AUC. The leader is the highest. A candidate is tied with a higher-ranked one when the paired 95 percent interval of their AUC difference includes zero; among tied candidates the cheaper wins, where cheaper means the lower end-to-end extraction time per 1,000 images or clips and, if equal, fewer parameters, and the decision text says when this rule decided. The ordering produced by this rule gives the top three, which enter I-1. The from-scratch reference is a baseline trained end to end on the same split for a fixed budget (10 epochs, AdamW 1e-3, no augmentation); it is reported in the same table with its protocol difference stated and is excluded from the selection. If the best candidate's lower interval bound on `dev_x` is 0.55 or below, the sweep is declared inconclusive for that task and said so; I-1 still proceeds with the top three, flagged.

## 9. Fine-tuning evidence, iteration I-1 (planned)

The top three candidates of each task from section 8 are fine-tuned under one shared budget, to support or reject any later claim of which is easiest to fine-tune.
- Budget: the same number of optimiser steps for every candidate of a task, the same data, the same augmentation (none to begin with), the same effective batch of 32 (with gradient accumulation if the real batch is smaller), AdamW with a cosine schedule and 5 percent warm-up, weight decay 0.05, mixed precision, full fine-tuning (gradient checkpointing where needed). The step count is set from the time per step measured in the sweep so that each task fits about 90 minutes, and is written into EXPERIMENTS.md before the run.
- Learning rates: three, 1e-5, 3e-5 and 1e-4, the same grid for all candidates; one run per rate for the sensitivity measure, then 3 seeds at the best rate (15 runs per task).
- Measured for every candidate: the steps needed to reach a fixed `dev_x` target (the leader's frozen `dev_x` AUC from I-0, rounded down to 0.01; a run that never reaches it counts as the worst); sensitivity to the learning rate (largest minus smallest final `dev_x` AUC over the three rates); peak VRAM; time per 1,000 steps and per epoch; instability events (a loss that is not finite, or a training loss above five times its running minimum after warm-up).
- The operational definition of "easiest to fine-tune": fewest steps to the target (median over runs), ties broken by the smaller learning-rate sensitivity, then by fewer instability events. A claim of this kind holds only under this protocol and is worded that way.
- A candidate that cannot be fine-tuned on 12 GB at an effective batch of 32 even with gradient checkpointing is recorded as not fine-tunable here, with the measured memory.

## 10. Preprocessing notes

D-034 (YuNet, the crop defaults) stays decided autonomously until the user confirms it. A short measurement of YuNet against two other face detectors is made on 300 random frames from FF++ val and from the Celeb-DF `dev_x` slice (share with a face found, speed, box agreement, stream S2-C); the crops are not redone unless the user agrees. JPEG quality 95 for the stored crops is a candidate preprocessing iteration and is not changed now.

## 11. Risks

- The `dev_x` slices are small (15 identity groups, 12 speakers, 2 generators), so many intervals will overlap and the cost rule will decide often.
- Frozen features may be weak for fine face artefacts; the sweep can only show that, not fix it.
- The generated-image path trains on 8,000 images, far fewer than R-011; absolute numbers will not match that paper.
- Extraction time is not yet measured for any candidate; I-1 step counts are set from the sweep measurements.
