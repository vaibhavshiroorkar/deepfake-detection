# Validity checks catch a detector that looks right and is not

Draft. Every number here resolves to a row in
[result traceability](result-traceability.md), and the rows that depend on the
Design B visual streams are regenerated after the live-BatchNorm retrain. No row
has reached `accepted`: the frozen protocol asks for three training seeds and
these streams were trained on one, so every result is `provisional (1 seed)`.
Section 7 states that and the rest of the limits plainly.

## Abstract

We ask whether cue-specific fusion of visual artifacts, audio spoofing and
mouth-audio alignment generalizes better than a strong visual baseline, under
source-disjoint and shortcut-controlled evaluation. It does not, on our data.
The contribution is not the architecture and not the accuracy. It is a set of
validity checks that caught a detector which looked right and was not, and the
measurements those checks produced.

Three results carry the paper. A cross-attention stream reached 0.9991 ROC-AUC
while a correspondence measure recorded every epoch and never optimised sat at
chance, so the accuracy is real and the stated mechanism is absent. Freezing
BatchNorm during fine-tuning, a change that looked like a clear improvement on
validation, moved cross-corpus ROC-AUC by 0.1276 in the wrong direction while
moving in-domain ROC-AUC by 0.5430 in the right one, against an epoch budget
worth 0.005. And selecting a checkpoint on validation
loss kept one that scored below chance, on a run where loss and ROC-AUC moved in
opposite directions on the same epoch.

## 1. Introduction

A deepfake detector that reads 0.999 in-domain and 0.75 cross-corpus is a
familiar result, and the usual response is to reach for more cues. That response
assumes the cues do what their names say. This paper is about checking that
assumption, and about what the checks found.

The research question was fixed before the experiments and anticipated a
negative answer: under source-disjoint and shortcut-controlled evaluation, does
cue-specific fusion of visual artifacts, audio spoofing and mouth-audio
alignment generalize better than a strong visual baseline? The exit criteria
stated that fusion does not need to win, and that a negative result is reported
with its confidence interval.

We report that negative result. We also report something we did not anticipate,
which is that three of the changes made along the way were wrong in ways that
only a validity check could see:

1. A stream can reach near-perfect accuracy with attention that never moved.
2. A regulariser that improves validation can destroy transfer.
3. A selection rule can prefer a checkpoint that scores below chance.

None of these is visible in a loss curve or an accuracy table. Each was caught
by a measurement recorded beside the training objective and never optimised
against it. That is the method this paper contributes.

## 2. Related work

Audio-visual feature fusion for deepfake detection is established, as is
shortcut-controlled evaluation, synchronization modelling, and holistic
coherence learning. We do not claim multimodal fusion is new. We claim that the
evidence practices around it are underspecified in a way that lets a stream with
an inert mechanism pass as a working one, and we give an instance.

## 3. Method

### 3.1 Two systems, one evaluation path

Design A is a three-branch pipeline: EfficientNet-B0 with a GRU for visual
artifacts, Wav2Vec2 with attentive pooling for audio spoofing, and a framewise
ResNet-18 mouth encoder with a separate Wav2Vec2 for mouth-audio alignment.
Fusion is regularized logistic regression over Platt-calibrated branch logits
plus three quality features, fitted only on out-of-fold rows.

Design B replaces the branches with configurable streams and the late head with
a feature-level head. Visual streams take a backbone and a temporal model.
Audiovisual streams are cross-attention over temporal tokens. `StreamFusion`
projects each stream's embedding to a common width, masks absent streams after
the projection so an absent stream contributes exactly zero including its bias,
and fuses in feature space.

Both are scored through the same code. This matters more than it sounds: the
largest gap in our results, Design A's visual branch at 0.7611 against Design
B's at 0.5067 on the same clips, was first suspected to be an evaluation
artifact. Scoring the Design A checkpoint through the Design B path returned
0.7611, matching to four decimals, which established the regression as real
before anything was concluded from it.

### 3.2 Validity checks, recorded and never optimised

The central instrument is `diagonal_mass`: the fraction of attention mass near
the diagonal of a cross-attention map. A stream that aligns mouth motion with
sound should concentrate mass there. It is computed every epoch, written to the
history file, and never enters the loss. That is the whole design. A measure
that is optimised stops being evidence about the mechanism and becomes another
thing the model fits.

Its baseline depends on the query length, which is a trap we fell into. Uniform
attention scores 0.0592 at 50 query steps and 0.3438 at 8. Comparing a stream at
8 steps against the 50-step baseline makes chance look like six times chance.

### 3.3 Abstention rather than imputation

A clip with no stable face track produces no visual view. The pipeline reports
it as unavailable and counts it, rather than substituting a full-frame crop or
dropping it from an inner join. Coverage and abstention rate are reported beside
every metric. Section 5.9 measures what this policy costs and buys.

### 3.4 Provenance

Splits are frozen by source identity and carry a split hash. The view cache is
content-addressed under a `preprocessing_config_hash` covering both the view
settings and the code version, so a server whose preprocessing differs from the
cache a model was trained on is rejected rather than silently served. Every
reported number resolves to a registry row with the artifact's SHA-256 and an
MLflow run.

### 3.5 Serving

The served system routes by media kind. A video drives every stream, an image
drives the visual stream alone and is marked limited, a sound file drives the
audio stream alone. Routing is decided by a table, not by which views happen to
exist, because an image produces a visual view and the emotion stream reads that
same view: without the table an image would reach a model that also expects a
voice and score on half its input.

Decision thresholds are per media kind, chosen by Youden's J on holdout rows the
head was not fitted on, and stored in the checkpoint.

## 4. Experimental setup

Training data is FakeAVCeleb, split 70/15/15 by source identity. The
cross-corpus test set is DFDC, the only audio-bearing corpus here not built on
VoxCeleb2. Celeb-DF-v2 and the evaluation-only MNW benchmark are used for
external checks; MNW is never tuned on.

Confidence intervals are 95 percent bootstrap intervals clustered on source
identity, not on the clip. The question a reader has is what happens on a
different set of speakers, not on a different draw of clips from these speakers.
On DFDC the identity-clustered interval is narrower than a clip-level one,
0.0776 against 0.1065, because identities there carry balanced class
proportions. Narrower is not why we chose it.

FaceForensics++ is absent, so no number here is comparable to a published table.

## 5. Results

### 5.1 Neither cross-modal stream measures correspondence

The emotion stream reads 0.9991 ROC-AUC on holdout and 0.9978 in-domain, the
best of five streams on both. Chance `diagonal_mass` depends on the query
length: 0.3438 at 8 steps, 0.0592 at 50.

| Stream | Query steps | Chance | Measured range |
| --- | --- | --- | --- |
| emotion | 8 | 0.3438 | 0.3434 to 0.3457, 9 epochs |
| lip-sync | 50 | 0.0592 | 0.0592 to 0.0595, 4 epochs |
| lip-sync, lower encoder LR | 50 | 0.0592 | 0.0592 to 0.0595, 6 epochs |

Both sit on chance and stay there across three separate training runs. Emotion's
ROC-AUC rose from 0.9847 to 0.9991 across its nine epochs without its attention
moving.

The accuracy is real and the mechanism both streams are named for is not
operating. An accuracy table ranks emotion first and says nothing about the cue
it is named for being absent.

A second, smaller result sits underneath. Lip-sync degrades from 0.7593 every
epoch after the first, which reads as the two pretrained encoders destabilising
on unfreeze. Retraining at a fifth of the encoder learning rate, with an extra
frozen epoch and more patience, changed nothing: validation loss quadruples at
epoch 2, while the encoders are still frozen. The head alone overfits, on a cue
its attention map says it is not reading.

### 5.2 Freezing BatchNorm trades transfer for in-domain fit

Both visual streams trained twice on the full training partition, one flag
apart. Labels are `video_fake`, the objective the visual stream optimises:

| Arm | Stream | Validation | In-domain | DFDC |
| --- | --- | --- | --- | --- |
| frozen | EfficientNet-B0 | 0.9997 | 0.9987 | 0.5067 |
| live | EfficientNet-B0 | 0.4909 | 0.4557 | 0.6343 |
| frozen | DINOv3, frozen ViT | 0.9559 | 0.9546 | 0.5147 |
| live | DINOv3, frozen ViT | 0.9559 | 0.9546 | 0.5147 |

Live BatchNorm moves EfficientNet by +0.1276 on DFDC and -0.5430 in-domain. The
DINOv3 rows are identical to four decimals on every epoch of both runs, which
rules out run-to-run variation: a frozen ViT has no running statistics, so the
flag must change nothing there, and it changes nothing.

Live BatchNorm statistics adapt to the data the model sees, which is what AdaBN
does deliberately, and freezing them removed an accidental domain adapter.

The trade is too steep to take. An in-domain ROC-AUC of 0.4557 is below chance,
so the live arm is not a detector that generalizes better, it is a broken
detector that ranks slightly above chance out of domain. We ship the frozen arm
and report the trade. The measurement establishes the size of the lever, not a
setting to adopt.

One confound remains in the table above: the live arm stopped at epoch 4 on
patience while the frozen arm ran 10, so the epoch budget is not held fixed. A
controlled 1,400-clip sweep holds it fixed and finds the epoch budget moves DFDC
by 0.005 against BatchNorm's 0.148.

The same checkpoint also has two legitimate in-domain numbers, 0.9987 against
`video_fake` and 0.9719 against `clip_fake`, because a clip with a real video
track and spoofed audio is one and not the other. They coincide on DFDC, where
every manipulated clip has a manipulated video track. We state which label each
number uses wherever both appear.

### 5.3 Selection on loss kept a checkpoint below chance

`visual-efficientnet` reached its lowest BCE at epoch 1 with its backbone still
frozen. That checkpoint measured 0.4668 in-domain, below chance. On another run
epoch 2 had the worse loss, 0.3931 against 0.3296, and the better ROC-AUC,
0.9547 against 0.9428.

BCE measures calibration and punishes a confident mistake hard, so a model that
ranks well while overconfident loses to one that hedges and ranks badly. Every
objective in this project is stated in ROC-AUC. Selecting on loss while
reporting ROC-AUC is a mismatch that produces a wrong answer silently.

### 5.4 Per-stream accuracy

Five streams, in-domain and cross-corpus. Intervals are regenerated with the
retrained visual streams.

| Stream | holdout | in-domain | DFDC |
| --- | --- | --- | --- |
| emotion | 0.9991 | 0.9978 | 0.5951 |
| visual, EfficientNet-B0 | 0.9733 | 0.9719 | 0.5067 |
| visual, DINOv3 frozen | 0.9216 | 0.9199 | 0.5147 |
| audio, Wav2Vec2 | 0.7919 | 0.7739 | 0.5070 |
| lip-sync | 0.7593 | 0.7587 | 0.4890 |

Four of five sit within 0.02 of chance on DFDC. The in-domain column says the
streams learned something; the DFDC column says most of what they learned does
not survive a change of corpus.

### 5.5 Fusing a subset beats fusing everything

All 31 subsets, each with its own head fitted on the same holdout rows:

| Partition | Best combination | Best single | All five |
| --- | --- | --- | --- |
| in-domain | audio + emotion + visual, 0.9990 | emotion, 0.9978 | 0.9988 |
| DFDC | emotion + lip-sync, 0.6076 | emotion, 0.5991 | 0.5411 |

Fusion beats the best single stream on both partitions. The full fusion does
not: on DFDC it falls below three of its own inputs. A system that always fuses
everything it holds is choosing the worse of two options already in its hands.

### 5.6 The negative answer to the research question

The protocol's comparison is a paired source bootstrap: identities are resampled
and the difference between the two systems is taken within each resample, so
they are never compared across different draws.

| Partition | Fusion minus visual | 95% interval |
| --- | --- | --- |
| in-domain | +0.0247 | [+0.0220, +0.0280] |
| DFDC | -0.0083 | [-0.0185, +0.0024] |

Fusion helps in-domain, and the interval is clear of zero. Cross-corpus the
difference is negative and the interval includes zero, so the honest statement
is not that fusion is worse but that no difference is detectable. The
in-domain advantage does not survive the change of corpus. Design B's best
cross-corpus combination reads 0.6076, below both.

### 5.6b Calibration collapses further than accuracy does

| Partition | ROC-AUC | Precision | Recall | FPR at 95% TPR | Brier | Calibration error |
| --- | --- | --- | --- | --- | --- | --- |
| in-domain | 0.9990 | 1.0000 | 0.9981 | 0.0000 | 0.0018 | 0.0022 |
| DFDC | 0.7500 | 0.9904 | 0.2820 | 0.8690 | 0.5466 | 0.6366 |

ROC-AUC falls by a quarter. Expected calibration error rises by a factor of 289.
At the fixed threshold the pipeline catches 28 percent of manipulated clips
cross-corpus while keeping precision at 0.99, so it is not accusing genuine
clips, it is failing to accuse manipulated ones and reporting confident
probabilities while it does.

For a reader deciding whether to deploy such a system, the calibration row
matters more than the ROC-AUC row: a ranking metric says the scores are ordered
usefully, and the Brier score says the numbers attached to them are not.

### 5.7 Redundancy was not where it was predicted

Lip-sync shares 82.5 percent of its errors with the audio branch in-domain,
correlation 0.820, and is anti-correlated with the visual stream at -0.160. Read
alone, that says lip-sync is a redundant audio duplicate and should be dropped.
It is also in the best DFDC combination. In-domain redundancy and cross-corpus
usefulness are different questions and the error-overlap table answers only the
first.

### 5.8 The deep head does not beat the late head

| Partition | Late fusion | Deep fusion |
| --- | --- | --- |
| in-domain | 0.9989 [0.9974, 1.0000] | 0.9988 [0.9971, 1.0000] |
| DFDC | 0.5687 [0.5279, 0.6052] | 0.5411 [0.4984, 0.5828] |

Indistinguishable in-domain, and on DFDC the simpler head reads higher with
heavily overlapping intervals. Nothing is separated. What the comparison does
rule out is the premise the feature-level head was built on: that the embedding
carries something the scalar logit does not.

### 5.9 Abstention buys no accuracy

On DFDC the head reads 0.5418 refusing every clip with a missing stream,
covering 97.6 percent, and 0.5411 answering from the streams that ran, covering
all of them. The abstention policy stays, because a verdict with no evidence
behind it is worse than a refusal. It cannot be argued for on accuracy.

### 5.10 One threshold cannot serve three media kinds

Chosen by Youden's J on held-back rows: 0.50 for a fused video, 0.76 for an
image through the visual stream alone, 0.26 for sound alone, at J of 0.994,
0.939 and 0.513. The same weights with different streams present put the fused
probability on a different scale.

### 5.11 How generalization fails

0.9990 in-domain, 0.7500 on DFDC, and 25 of 85 manipulated clips detected on
MNW, which is fake-only and so has no ROC-AUC. The per-generator breakdown is
where the shape shows:

| MNW generator | Detected |
| --- | --- |
| vasa_1 | 0 of 10 |
| raskai | 0 of 4 |
| diff2lip | 1 of 6 |
| sadtalker_video | 1 of 5 |
| wav2lip_gfpgan | 1 of 6 |
| echo_mimic | 3 of 10 |
| the remaining named generators | 2 of 4 to 2 of 7 each |
| unnamed in-the-wild clips | 7 of 8 |

The direction of the failure is measurable. Frame-to-frame motion, computed on
the exact view the model is handed, correlates negatively with the fake
probability: -0.316 among manipulated DFDC clips and -0.202 among manipulated
in-domain clips, while barely moving the score on genuine clips in either
corpus, -0.035 and -0.010. DFDC's top motion quartile carries 19 percent more
motion than FakeAVCeleb's, so the clips it adds are the ones the model is least
willing to call fake.

That is consistent with the operating point rather than with the intuition.
Precision on DFDC is 0.9904 and recall is 0.2820: the pipeline misses 72 percent
of manipulated clips and almost never accuses a genuine one. An earlier draft of
this paper asserted the opposite mechanism, that motion looks like forgery
flicker and produces false alarms. False alarms on genuine clips rise only from
0.0 to 2.7 percent between the calmest and most moving quartiles. The claim was
wrong and the measurement that corrected it is one line of analysis over
artifacts that already existed.

The two generators missed completely are the ones whose artifacts are temporally
smooth, and the clips the detector does find are the in-the-wild ones, which
carry compression and editing artifacts as well.

## 6. Failures that changed a result

The failure log holds about 50 entries. These are the ones that changed a
number rather than costing time, and in several the first diagnosis was wrong,
which is the part worth reading.

**Selection on loss.** Covered in 5.3. Two checkpoints were published and
withdrawn.

**BatchNorm frozen.** Implementing a dead configuration field took validation
from 0.5154 to 0.9058 and was recorded as a clean win. It was a trade, and
validation was the only thing measured at the time.

**Attention compared against the wrong baseline.** `diagonal_mass` 0.3439 was
called six times chance using a baseline computed at a different query length.
It is at chance.

**A cache loader that materialised every view.** Loading all four views per clip
cost about 20 MB where 9.19 MB was needed and killed a run after two and a half
hours.

**A checkpoint picker that walked the cache.** A recursive glob over 62,000
entries added ten seconds to every page rerun.

**Windows commit charge.** A series of failures that read as "out of memory with
15 GB free" were commit-charge exhaustion, 126.7 of 127.9 GB, with about 99 GB
held outside any live process after repeated forced kills of CUDA processes.
Free physical RAM is not the ceiling on this platform.

**Two withdrawn readings of our own results.** The emotion stream was called a
duplicate of the visual stream; it is the best stream on both partitions.
Lip-sync was called a redundant audio duplicate to drop; it carries the best
cross-corpus combination.

## 7. Limitations

Trained on FakeAVCeleb, not FaceForensics++, so no number here is comparable to
a published table.

One seed for every Design B stream where the protocol asks for three. Every
result is `provisional` for that reason alone.

No in-the-wild training data.

The cross-modal streams do not measure correspondence (5.1), so the
architecture's stated mechanism is unverified and the streams should be
described by what they do rather than by what they are named.

Several required evaluations in the protocol have not been run: the
identity-strict stress subset, leave-one-method-family-out, and the compression,
noise and resolution stress tests.

## 8. Conclusion

Cue-specific fusion did not beat the visual baseline cross-corpus on our data,
and the full fusion was worse than three of its parts. That is the answer to the
question as asked.

The more useful result is how close the project came to reporting the opposite.
A stream at 0.9991 with an inert mechanism, a regulariser that bought validation
accuracy by removing a domain adapter, and a selection rule that preferred a
below-chance checkpoint would each have passed unnoticed under an evaluation
protocol built on accuracy tables alone. Each was caught by a quantity recorded
next to the objective and never optimised against it. We think that practice,
rather than any architecture here, is what transfers.
