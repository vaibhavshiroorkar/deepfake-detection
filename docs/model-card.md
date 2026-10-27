# Model card

The fusion, LAV-DF and PE/CLIP results below are historical measurements with known protocol defects.
Outer-fold labels selected the checkpoints that produced the original OOF features.
Explicit zero-start LAV-DF windows could shift into annotated manipulation.
PE/CLIP features used the wrong input normalization.
Affected registry rows require reruns. Their old scores do not establish the corrected pipeline's performance.

For the Design A late-fusion system trained in `program-20260906`. Every number
here resolves to a row in [result traceability](research/result-traceability.md)
and every claim is drawn from [findings](research/findings.md). Nothing in this
card was measured for the card.

Read the headline first: **this model should not be deployed.** It is research
evidence about what cue-specific fusion does and does not buy. The sections
below say why in measured terms rather than as a disclaimer.

## What it is

Three cue-specific branches over one clip, combined by a calibrated late head.

| Part | Model | Cue |
| --- | --- | --- |
| Visual | EfficientNet-B0 with a GRU over 16 sampled frames | Face artifacts |
| Audio | Wav2Vec2 Base with attentive pooling | Speech spoofing |
| Sync | ResNet-18 mouth encoder with a separate Wav2Vec2 | Mouth-audio alignment |
| Head | Logistic regression over Platt-calibrated branch logits plus three quality features | Combination |

The head is fitted only on out-of-fold rows from source-grouped cross-fitting,
but the historical fold checkpoints selected epochs on those same holdout rows.
The head therefore requires retraining on nested OOF features. The logistic head was
selected over a small MLP by measurement, not preference: see F18.

A clip missing any branch is abstained on rather than imputed. The head demands
every column, which is a property of the head and is reported as coverage.

## Intended use

Research into detector generalization and into the validity checks that catch
shortcuts. That is the whole of it.

The project's contribution is the protocol, not the accuracy. Four shortcuts
were caught before they produced a reported result, each by a cheap check run
before training rather than after, and that practice is what transfers.

## Out of scope

Taken from the [threat model](threat-model.md), and each line is a measured
limit rather than a caution.

- **Any decision about a person.** Identity, law enforcement, employment, or
  moderation outcomes. The score does not support them.
- **Proving media authentic.** The system detects manipulation of a real
  recording. A low score is not evidence of authenticity.
- **Media generated whole.** A face-crop pipeline is at chance on person-free
  generated scenes, and a full-frame probe is at chance on face manipulation
  (F13). Neither path covers both.
- **Any corpus unlike the training data.** See the next section.
- **Image-only or audio-only claims**, real-time guarantees, white-box
  adversarial robustness, clips with no stable speaking face, and attribution to
  a generator.

## How well it works, and where it stops

In-domain, on the held-out FakeAVCeleb test partition:

| Metric | Value |
| --- | --- |
| Fusion ROC-AUC | 0.9990 |
| Fusion gain over the strongest single branch | +0.0159 [+0.0129, +0.0190] |
| Expected calibration error | 0.0022 |
| Errors | 3 of 1,648, all on `faceswap` |

Cross-corpus, which is the number that matters:

| Corpus | ROC-AUC | What happens |
| --- | --- | --- |
| DFDC | 0.6873 mean, 0.6422 to 0.7500 | Three seeds, each with its own head. Fusion never beats its own visual branch, and is measurably worse at two of three seeds |
| Celeb-DF-v2 | 0.6682 | 141 of 155 genuine videos called manipulated |
| MNW (locked, external) | detection 24 of 84 fakes | 0 of 10 on `vasa_1`; 6 of 7 likely-manipulated wild clips, and the one likely-authentic clip also called fake |

The in-domain figure is not evidence of generalization. A strong in-domain
number has four times in this project turned out to be a shortcut rather than
skill, which is why leave-one-generator-out is the only headline metric on the
open-world track.

### The five limits that decide whether this is usable

**It accuses genuine video, at a rate set by capture conditions.** False alarms
on real clips run 0.00 percent in-domain, 4.14 on DFDC and 90.97 on Celeb-DF
(F16). Forcing zero false alarms on Celeb-DF needs a 0.9998 threshold, at which
nothing is caught at all. The per-media-kind thresholds are chosen by Youden's
J, which weights sensitivity and specificity equally, and that is the wrong rule
for any deployment that cannot accuse a genuine video.

**It is wrong confidently.** Cross-corpus the model is more sure when it is
wrong than when it is right: 0.8742 against 0.8299 on DFDC, and 951 of 1,590
fusion errors sit past 0.9 confidence (F22). No confidence threshold abstains
its way out of this, because confidence does not separate right from wrong once
the corpus changes.

**Its calibration does not transport.** Expected calibration error is 0.0022
in-domain and 0.6366 on DFDC. An unweighted mean of the calibrated branches,
with nothing learned, is better calibrated cross-corpus than the fitted head by
a factor of eleven (F18). The head's learned weights are the part that fails to
cross corpora.

**Most of its errors are beyond any head.** On DFDC the fused score and the
visual branch are both wrong on 1,585 of 2,351 compared clips (F23). Fusion
repairs 314 and breaks 5, which is real and asymmetric, but 67 percent of the
corpus is not recoverable by combining these branches at all.

**Ordinary recompression destroys its operating point.** JPEG at quality 75,
lighter than what a messaging app applies, costs 0.0054 of ROC-AUC and takes
false alarms from 0 to 76 of 150 genuine clips. By quality 50 every genuine clip
is called fake while ROC-AUC still reads 0.9187 (F24). Light background noise
does the same to the audio branch in the opposite direction, missing 857 of 859
fakes at 20 dB SNR (F25). Both figures hold the face track fixed and are
therefore optimistic.

## Fairness

No gap by race or gender is visible on the in-domain test partition, and no
genuine clip is called fake in any subgroup (F20).

That is not a fairness claim. Every slice sits at or near ROC-AUC 1.0, where a
real gap could not become visible, and the denominators are 14 to 16 genuine
clips per race group. The partition cannot answer the question. Neither
cross-corpus set carries demographic annotation, so the question is unanswered
where the model actually fails.

## Uncertainty

Every interval is a 95 percent bootstrap clustered on source identity, because
the question is what happens on different speakers and FakeAVCeleb puts many
clips on one.

Most results are `provisional (1 seed)` against a protocol that asks for three.
Where three seeds do exist, they show what that costs, and cross-corpus it is
large:

| System | Seeds | In-domain spread | DFDC spread |
| --- | --- | ---: | ---: |
| visual branch (F21) | 17, 29, 43 | 0.0005 | 0.0733 |
| fused system (F27) | 17, 29, 43 | 0.0000 | **0.1078** |
| sync branch (F26) | 17, 29, 43 | 0.0567 | not scored |

In-domain the seeds agree to three or four decimal places. Cross-corpus the
fused system ranges over 0.1078, which is thirteen times the seed-17
fusion-against-visual difference it is used to discuss, and **the seed this repository reports is the
best of the three in both cases**. The honest form of a cross-corpus number here
is the mean and range, or the seed stated beside it.

Paired comparisons are unaffected: they hold the seed fixed and resample clips,
so seed variance sits outside their intervals by construction.

## Training data

FakeAVCeleb, split 70/15/15 by source identity with a frozen split hash and no
source overlap between partitions. Real clips are the scarce resource in every
partition: the full training partition holds 14,391 fake against 700 real
videos, because FakeAVCeleb contains only 500 genuine identities in total.

DFDC, Celeb-DF-v2 and MNW are evaluation only. MNW cannot select a model or a
threshold. See the [data card](data-card.md) for the full inventory, the
handling rules, and the `FakeVideo-RealAudio` schema caveat.

## Provenance

The view cache is content-addressed under a hash covering both the view settings
and the code version, so a server whose preprocessing differs from the cache a
model was trained on is refused rather than silently served. Checkpoint hash,
run ID, split hash, commit and seed are checked before the dashboard will score
anything.

## Known defects

- Padded audio reached the attentive pooling unmasked in every trained
  checkpoint, which is the padding shortcut named in the threat model. Masking
  is now implemented and covered by tests, but no shipped checkpoint was trained
  with it, and the caches those checkpoints used record no sample count.
- The silence shortcut, the other padding-adjacent risk in the threat model, was
  measured and is **not** present: shifting the waveform half a second changes
  nothing (F25).
- The two cross-modal streams never measured correspondence. Diagonal attention
  mass sat at chance in every epoch of three runs while accuracy rose (F1).
- The branch architectures were barely contested. `VIS-01` kept EfficientNet-B0
  over ConvNeXt-Tiny on stability, not accuracy: ConvNeXt diverged at one seed
  of three (F29). `AUD-01` is undecided, because Wav2Vec2 and WavLM both read
  1.0000 in-domain and no cross-corpus audio partition exists (F28).
