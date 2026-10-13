# Validity checks catch a detector that looks right and is not

Draft. Every number here resolves to a row in
[result traceability](result-traceability.md). Of 36 rows, 5 are `accepted`, 30
are `provisional (1 seed)` and 1 is `pending`. The protocol asks for three
training seeds; Design A's three branches now have them, every Design B stream
has one, and the fusion rows stay provisional because the head reads out-of-fold
features exported from seed-17 fold models. Section 7 states that and the rest of
the limits plainly, and section 5.13 measures what a seed is worth.

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
validation, cost 0.5430 of in-domain ROC-AUC and bought 0.1276 cross-corpus in an
uncontrolled comparison; with the epoch budget held fixed that cross-corpus
figure is **0.0177**, so the lever is real, one-seventh the size, and the
in-domain cost is what survives the control. And selecting a checkpoint on
validation loss kept one that scored below chance, on a run where loss and
ROC-AUC moved in opposite directions on the same epoch.

A fourth result frames the other three. Cross-corpus ROC-AUC on this pipeline
moves by **0.1057** with the training seed alone, against 0.0001 in-domain, and
the seed this paper reports is the best of three. Every unpaired cross-corpus
comparison below is therefore stated with its range, and the paired comparisons,
which hold the seed fixed and resample clips, are unaffected.

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

Both figures are seed 17. Design A's branch spans 0.6878 to 0.7611 across three
seeds (5.13), so the gap is between the most favourable Design A draw and a
single Design B arm. It remains far larger than the seed spread, so the conclusion
holds, but the margin is smaller than one seed each suggests.

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

Seeds. The protocol fixes 17, 29 and 43. All three Design A branches have all
three; every Design B stream has seed 17 alone. **Unless a number below says
otherwise it is seed 17**, and seed 17 is the best of the three on DFDC, so
single-seed cross-corpus figures here are the most favourable draw rather than a
central one. Section 5.13 measures what that is worth.

A resample that lands on one class is skipped and counted out of the reported
sample count, rather than scored as 0.5. Scoring it 0.5 places a point mass in
the middle of the percentile distribution, which can make an interval contain
chance because of the fallback rather than because of the data.

Confidence intervals are 95 percent bootstrap intervals clustered on source
identity, not on the clip. The question a reader has is what happens on a
different set of speakers, not on a different draw of clips from these speakers.
On DFDC the identity-clustered interval is narrower than a clip-level one,
0.0776 against 0.1065, because identities there carry balanced class
proportions. Narrower is not why we chose it.

FaceForensics++ c23 is also present and is used as a second training corpus, so
one result here follows the protocol published cross-dataset tables use: train
on FF++, evaluate zero-shot on Celeb-DF-v2 and DFDC. Its provenance is weaker
than the rest, a third-party mirror rather than the TUM EULA route, which the
data card records. It carries no audio track at all, so it trains the visual
stream alone.

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

The obvious objection is that FakeAVCeleb does not require correspondence: its
fakes are whole-clip manipulations, so a model can separate them by recognising
the generator and never needs to compare mouth to sound. LAV-DF removes that
escape by construction. Its adapter cuts two windows from the same file, one
containing the manipulated span and one avoiding every manipulated span, so
speaker, lighting, microphone, codec and re-encoding are all held fixed and the
windows differ in one thing.

A lip-sync stream trained on those pairs reads 0.9969 ROC-AUC on 1,161 held-out
windows with `diagonal_mass` at 0.0592 against a chance of 0.0600. At chance, on
the corpus built to make correspondence the only available cue. The model is
therefore not failing to learn correspondence because the data let it cheat: on
data designed to forbid cheating it still found another route, and reached
near-perfect accuracy on it.

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
controlled 1,400-clip sweep holds it fixed, and **it does not reproduce the size
of the effect**:

| Arm | Epochs | Validation | DFDC |
| --- | ---: | ---: | ---: |
| frozen BatchNorm | 10 | 0.9072 | 0.5508 |
| live BatchNorm | 10 | 0.5500 | 0.5686 |
| frozen BatchNorm | 3 | 0.8728 | 0.5466 |

With the budget held at 10, BatchNorm moves DFDC by **+0.0177**, not the +0.1276
of the uncontrolled comparison. Varying epochs with BatchNorm frozen moves it by
0.0042, so the epoch budget is not the explanation either. Neither lever is worth
what the uncontrolled numbers implied.

What survives the control is the direction and the in-domain cost: live BatchNorm
still collapses validation from 0.9072 to 0.5500. The trade is real and still too
steep to take, and its cross-corpus half is roughly seven times smaller than the
uncontrolled comparison suggested. These are 1,400-clip arms, so read the
difference between them rather than the absolute values.

An earlier version of this section reported the controlled sweep as giving 0.148
for BatchNorm against 0.005 for epochs. That was quoted from a run whose output
was never written under `runs/`. The sweep has since been run and recorded, the
epoch figure held, and the BatchNorm figure did not.

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

The baseline here is the Platt-calibrated visual score alone. Against a head
fitted on the visual branch **plus the three quality features**, a stronger
baseline, the in-domain gain is +0.0159 [+0.0129, +0.0190]. Both are paired on the
same clips under the clip-level label; they differ only in what "visual" means,
and we report both rather than the more flattering one.

The in-domain gain is also narrower than it looks in kind, not just in size. It is
attributable almost entirely to the 75 `RealVideo-FakeAudio` clips, where the
video is genuine and only the audio was replaced, so no visual branch can detect
them even in principle. Fusion's measured in-domain benefit here is the cue
coverage it was built for rather than a general improvement.

Fusion helps in-domain, and the interval is clear of zero. Cross-corpus the
difference is negative and the interval includes zero, so the honest statement
is not that fusion is worse but that no difference is detectable. The
in-domain advantage does not survive the change of corpus. Design B's best
cross-corpus combination reads 0.6076, below both.

### 5.6a Training on the standard cross-dataset corpus did not improve transfer

The visual stream trained on FF++ c23 and scored on corpora it never saw,
beside the same stream trained on FakeAVCeleb and scored on the same clips:

| Trained on | FF++ test | Celeb-DF-v2 | DFDC | FakeAVCeleb test |
| --- | --- | --- | --- | --- |
| FF++ c23 | 0.8723 [0.8425, 0.9008] | 0.6348 [0.5766, 0.6920] | 0.4920 [0.4478, 0.5350] | 0.8253 [0.7921, 0.8638] |
| FakeAVCeleb | not scored | 0.6682 | 0.7611 | 0.9988 |

Transfer to DFDC gets worse, from 0.7611 to 0.4920, an interval spanning chance.
Celeb-DF is a tie with heavily overlapping intervals. Whatever FF++'s six
manipulation families buy, it is not cross-corpus transfer here.

Both rows are one seed each, and the FakeAVCeleb row is seed 17, the best of its
three on DFDC. Against that model's three-seed mean of 0.7151 the drop is 0.7151
to 0.4920, and against its worst draw 0.6878 to 0.4920. The conclusion is
unchanged at every seed, because 0.4920 sits below the whole range, but the
headline difference is 0.22 rather than the 0.27 a best-draw comparison gives.

The penalty is not a fixed cost of leaving the training corpus, though. The FF++
model reads 0.8253 on FakeAVCeleb, far above its 0.4920 on DFDC, so each model
is best on its own corpus and second best on the other's while DFDC is hard for
both. Both training corpora are studio-grade manipulations of reasonably clean
footage; DFDC is consumer video, and neither contains anything resembling it.

One seed, and FF++'s no-audio constraint means this compares visual streams
only, against a Design A number trained with a GRU rather than an LSTM.

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

0.9990 in-domain, 0.7500 on DFDC, and 24 of 84 manipulated clips detected on
MNW, 29 percent [20, 39], which is almost fake-only and so has no ROC-AUC. The
breakdown is where the shape shows:

| MNW group | Detected | 95% interval |
| --- | --- | --- |
| named generators | 18 of 77, 23% | [15%, 34%] |
| in-the-wild clips | 6 of 7, 86% | [49%, 97%] |
| worst named: vasa_1 | 0 of 10 | [0%, 28%] |
| worst named: raskai | 0 of 4 | [0%, 49%] |

The per-generator counts are 4 to 10 clips, so no single row carries weight on
its own and the interval on 0 of 10 reaches 28 percent. What does carry is the
separation between the two groups, whose intervals do not overlap: the detector
finds in-the-wild clips, which have been re-encoded and edited, and misses the
raw output of named generators.

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

### 5.12 The identity-strict condition does not change the number

Result: `A-visual-dfdc` and the identity-strict subset. Scoring the frozen visual
branch on the identity-strict stress subset, where no target identity appears in
more than one partition, leaves the number where the source-disjoint split put
it. Source-disjointness was already doing the work; the stricter condition adds
nothing to detect.

### 5.13 A seed moves the cross-corpus number more than any effect we report

Results: `A-visual-seeds`, `A-fusion-seed-spread`. Three seeds, everything else
held fixed.

| System | Seed 17 | Seed 29 | Seed 43 | In-domain spread | DFDC spread |
| --- | ---: | ---: | ---: | ---: | ---: |
| visual branch, DFDC | 0.7611 | 0.6963 | 0.6878 | 0.0005 | **0.0733** |
| fused system, DFDC | 0.7500 | 0.6814 | 0.6442 | 0.0001 | **0.1057** |

In-domain the seeds agree to three or four decimal places. Cross-corpus the fused
system ranges over 0.1057, which is twelve times the fusion-against-visual
difference of -0.0085 that 5.6 reports, and six times the in-domain gain. **The
seed we report, 17, is the best of the three in both rows**, against a fused mean
of 0.6919.

Fusing three branches did not average the noise away, it compounded it: 0.1057
for the fused system against 0.0733 for the visual branch alone.

This does not touch the paired comparisons in 5.6, 5.8 or 5.15, which hold the
seed fixed and resample clips, so seed variance sits outside their intervals by
construction. It bounds what an unpaired cross-corpus comparison between two
differently seeded systems can claim.

### 5.14 Corruption destroys the operating point while ROC-AUC barely moves

Results: `A-corruption-visual`, `A-corruption-audio`. Corruption applied to the
cached view, so the face track is held fixed and only the pixels the model sees
degrade.

| Condition | ROC-AUC | Genuine called fake | Fakes missed |
| --- | ---: | ---: | ---: |
| clean | 0.9988 | 0 of 150 | 3 of 1,498 |
| JPEG 75 | 0.9934 | **76 of 150** | 1 |
| JPEG 50 | 0.9187 | **150 of 150** | 0 |
| resize 0.50 | 0.4316 | 150 of 150 | 0 |

JPEG at quality 75 is lighter than a messaging app applies. It costs 0.0054 of
ROC-AUC, which any table reads as robustness, while taking the false-alarm rate
from 0 to 51 percent. At quality 50 the model calls every genuine clip fake and
misses nothing, which is a detector that has stopped discriminating, and its AUC
there is still 0.9187.

The audio branch fails in the opposite direction. Noise at 20 dB SNR costs 0.0227
of ROC-AUC and misses 857 of 859 fakes. Visual degradation calls everything fake;
audio degradation calls everything genuine. A fused system inherits both and they
do not cancel.

One clean negative: shifting the waveform by half a second changes nothing,
1.0000 against 1.0000, so the leading-silence shortcut the threat model names is
absent from this checkpoint.

Scope. These numbers hold the face track fixed, so they isolate the classifier
from the detector. Corrupting a source video would also break face detection and
produce abstentions this cannot see, and JPEG artifacts at 224 pixels are not
those of compressing a 720p source and then cropping. Every figure above is an
optimistic bound rather than an estimate.

### 5.15 The model is more confident when it is wrong, cross-corpus

Result: `A-error-analysis`. Confidence here is how sure the model was of whatever
answer it gave.

| Evaluation | Errors | Sure when right | Sure when wrong | Errors past 0.9 |
| --- | ---: | ---: | ---: | ---: |
| fusion, in-domain | 3 of 1,648 | 0.9990 | 0.9949 | 3 |
| fusion, DFDC | 1,590 of 2,351 | 0.8299 | **0.8742** | 951 |
| visual, DFDC | 1,942 of 2,410 | 0.8517 | **0.9244** | 1,482 |

On both cross-corpus sets the model is more confident when wrong than when right.
This closes an escape route 5.9 leaves open: abstaining on partial coverage buys
no accuracy, and abstaining on low confidence cannot work either, because
confidence does not separate right from wrong once the corpus changes. There is no
threshold on this score that drops errors faster than correct answers.

The same analysis bounds what any fusion head could do here. On DFDC the fused
score and its own visual branch are both wrong on **1,585 of 2,351** compared
clips, so 67 percent of that corpus is not recoverable by combining these
branches at all. Fusion repairs 314 of the branch's errors and breaks 5, an
asymmetric trade that ROC-AUC does not show because it changes threshold
decisions without improving the ranking.

### 5.16 The sync branch, and an exit criterion never measured

Results: `A-sync-seeds`, `A-research-design-audit`.

The synchronization branch had no evaluation path: `evaluate branch` takes visual
or audio only, because sync predicts an offset class rather than a label. Scoring
the anomaly logit the fusion head actually reads, against the clip-level label:

| Seed | ROC-AUC | 95% interval |
| --- | ---: | --- |
| 17 | 0.4364 | [0.3888, 0.4832] |
| 29 | 0.4570 | [0.4136, 0.4999] |
| 43 | 0.4931 | [0.4341, 0.5524] |

Every seed is below 0.5 and two intervals exclude it. The scalar fusion reads
from this branch carries no usable signal at any seed. The label is clip-level and
therefore wrong for a re-synchronised fake, which is genuinely fake and may be
well aligned, so these are a floor rather than a measurement of alignment
detection.

The research design states an exit criterion for this branch: the sync offset
confidence interval must exclude chance. **It has never been measured.** The
branch's history records only a combined loss, and that loss is offset
cross-entropy plus 0.1 times a contrastive term, so the offset task's own
performance cannot be recovered from what was written. The best-epoch validation
loss is 2.1843 against a chance cross-entropy of ln(8) = 2.079 for the 8-way task;
subtracting an estimated contrastive contribution puts offset cross-entropy near
2.05, marginally under chance. That is suggestive and it is not a measurement.

### 5.17 One branch transfers and two do not

Results: `A-visual-seeds`, `A-audio-crosscorpus-gap`, `A-sync-seeds`.

| Branch | In-domain | Cross-corpus |
| --- | --- | --- |
| visual | 0.9988 | 0.7151 mean over three seeds |
| audio | 0.7739 | **0.5054 [0.4747, 0.5371]**, contains chance |
| sync | 0.4364 to 0.4931 over three seeds | not scored |

The audio branch's cross-corpus number was long absent and read as work nobody
had done. It is measurable at clip level and it is chance. A cue-specific audio
evaluation is not available: DFDC does not publish which modality each forgery
touched, so its audio-only label is a placeholder, and the corpora that would
resolve it either carry no audio or are built on the same VoxCeleb2 footage as the
training set.

So the system this paper studies has one branch that transfers and two that do
not. The in-domain fusion gain is entirely attributable to the 75
`RealVideo-FakeAudio` clips a visual branch cannot detect in principle, and 5.15's
finding that 1,585 of 2,351 cross-corpus errors are beyond any head follows
directly rather than needing a separate explanation.

### 5.18 Open-world generated video, and four shortcuts caught before training

Results: `C-df26-audio-audit`, `C-df26-logo`.

Everything above detects manipulation of a real recording. Video generated
whole is a different problem, and 5.11 shows this pipeline is close to blind on
it: 0 of 10 on `vasa_1`. So the second half of this paper asks a narrower
question on a different corpus. Can a frozen encoder with a linear probe detect
a video generator it has never seen?

DF26 holds 2,691 clips: 271 real and 2,420 from seven generators. Four are
commercial (Veo 3.1, Kling 3.0, Grok Imagine, Wan 2.6) and three are open source
(HunyuanVideo 1.5, LTX 2.3, Wan 2.2). The probe trains on six generators and is
scored on the seventh, once per generator. There is no in-domain column. Source
videos are split before generators, so no source appears on both sides, and a
model's image-to-video and text-to-video modes count as one generator.

Each of those rules exists because a check found the leak it closes, and all
four checks ran before any number was reported:

1. **Content.** Veo 3 lab clips scored against talking-head reals came out
   near perfect. That was a classifier for what was in the scene.
2. **Shared real clips.** A generator-only split put the same real clips in
   training and test. A smoke test scored perfectly on every arm, and no
   probe is that good.
3. **Sibling modes.** Holding out one mode of a model while training on its
   other mode leaks the model's weights.
4. **Audio presence.** Every sampled commercial clip carries an audio track, no
   open-source clip does, and 3 of 30 real clips do. Any audio branch scored on
   DF26 would read near 1.0 while measuring a container property, so DF26 is
   treated as visual-only.

Intervals in 5.18 to 5.20 resample clips, or source videos for paired
differences. Generated clips have no speaker to cluster on, so these intervals
are narrower than the identity-clustered ones above and should not be compared
with them.

### 5.19 Frame rate leaked through one generator

Results: `C-df26-logo`, `C-df26-logo-controlled`.

The first DF26 table, a frozen DINOv3 probe, read macro 0.6942. Real clips were
the only ones at 25 or 29.97 frames per second, and the feature vector includes
a term that responds to motion, so frame rate was a plausible path into the
score. Re-encoding every clip to 640x360 at 24 frames per second gives 0.6786.
The loss is concentrated: Wan 2.6, the only generator at 30 fps, fell 0.094 and
HunyuanVideo fell 0.045, while the other five moved by less than 0.014. Every
DF26 number below uses the re-encoded clips.

### 5.20 The encoder is worth 0.07, against the argument for the one we kept

Results: `A-encoder-comparison`, `C-df26-logo-pe`.

| Encoder | Macro ROC-AUC | Paired difference against DINOv3 |
| --- | ---: | --- |
| Perception Encoder, PE-Core-B/16 | **0.7527** | +0.0741 [+0.0121, +0.1317] |
| CLIP ViT-B/16 | 0.7454 | +0.0668 [+0.0049, +0.1289] |
| DINOv3 ViT-S/16 | 0.6786 | reference |

Same clips, same split, same probe; only the frozen encoder changes. Perception
Encoder leads on all seven held-out generators, and PE against CLIP is not
separated, -0.0073 [-0.0603, +0.0463].

| Held-out generator | Perception Encoder |
| --- | --- |
| Veo 3.1 | 0.8496 [0.7886, 0.9015] |
| Wan 2.6 | 0.8164 [0.7472, 0.8785] |
| Grok Imagine | 0.7905 [0.7158, 0.8552] |
| Kling 3.0 | 0.7821 [0.7073, 0.8447] |
| Wan 2.2 | 0.6996 [0.6340, 0.7603] |
| HunyuanVideo 1.5 | 0.6764 [0.6071, 0.7401] |
| LTX 2.3 distilled | 0.6542 [0.5866, 0.7166] |

This comparison has a history that belongs in a paper about validity checks. A
figure of 72.7 for PE against 57.0 for DINOv3 changed the project's encoder
choice for weeks with no artifact behind it. Its DINOv3 number was eleven points
below this pipeline's own measurement, so the claim was withdrawn and DINOv3
kept. The reason given for keeping it was mechanistic: a vision-language
objective pulls features toward semantics and away from the texture detail
forensics reads. Re-run under the protocol, the withdrawn claim had the right
direction at half the size, and the mechanism argument was wrong for generated
video: both vision-language encoders beat the self-supervised one.

The commercial generators score higher than the open-source ones, 0.81 against
0.68. The DF26 authors report the opposite for their detectors. Theirs transfer
from face manipulation, while this probe transfers between generators, and all
four commercial generators were collected through one platform and may resemble
each other more than the open-source three do. That explanation is untested, and
the gap should not be quoted until it is.

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

Design A and Design B are trained on FakeAVCeleb, which almost nobody uses as a
training corpus, so their numbers are comparable only against each other. The
FF++ result in 5.6a follows the published protocol and is the exception, at one
seed and visual-only.

One seed for every Design B stream where the protocol asks for three, so every
Design B result is `provisional` for that reason alone. Design A now has three
seeds on all three branches, and what they show is in 5.13: the fused
cross-corpus number moves 0.1057 with the seed. The fusion rows remain
`provisional` because the head is fitted on out-of-fold features exported from
seed-17 fold models, and cross-fitting the folds per seed is eighteen further
training runs that have not been done.

No in-the-wild training data.

The cross-modal streams do not measure correspondence (5.1), so the
architecture's stated mechanism is unverified and the streams should be
described by what they do rather than by what they are named.

Required evaluations still not run: leave-one-method-family-out, the
authentic-only against global-fake sync label comparison, and the
leading-silence-removal ablation. The identity-strict stress subset and the
compression, noise and resolution stress tests have since been run and are
reported in 5.12 and 5.14.

One exit criterion of the research design has never been measured at all: "the
sync offset confidence interval must exclude chance". The sync branch's history
records only a combined loss, and that loss adds a contrastive term to the offset
cross-entropy, so the offset task's own performance cannot be recovered from the
artifacts. Section 5.16 reports what can be said.

Fully generated video is out of scope and the system is close to blind on it: 0
of 10 on `vasa_1` and 0 of 4 on `raskai`. Every corpus here manipulates a real
recording, so the traces the models learned are seams and boundary artifacts
that generated video does not have. The motion result compounds it, since
smoothness reads as authenticity and generated video is smooth. Detecting that
class needs the whole frame, frequency-domain features and a
leave-one-generator-out protocol, and is a separate question from the one this
paper asks.

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

Two limits on that claim, since the same practice was applied unevenly to this
work. One of the three levers above shrank by a factor of seven once its control
was actually run, having been reported for months from a run whose output was
never written down. And the branch decomposition is thinner than the framing
suggests: of three cue-specific branches, one transfers cross-corpus, one is at
chance and one is below it, so this is a study of one working detector and two
that do not rather than of fusion between three. Both are in section 7.

A final observation the seed result forces. Reporting a single cross-corpus
number to four decimal places invites comparisons the measurement cannot support:
here the seed alone moves that number by 0.1057, more than any architectural
choice we tested. A validity protocol that records mechanism metrics but not
seed spread is still incomplete, and ours was.
