# Findings

## Finding contract

Every finding cites a result ID from
[result traceability](result-traceability.md). Report negative and positive
results with the same uncertainty and coverage rules.

A finding recorded here is not a claim the paper makes. It is a claim the
measurements support at the status its result rows carry. Every Design B row is
`provisional (1 seed)` because the frozen protocol asks for three training seeds
and these streams were trained on one, so every finding below inherits that
limit and says so once here rather than in every line.

## Accepted findings

None. No result row has reached `accepted`, for the seed reason above.

## Provisional findings

### F1. Neither cross-modal stream measures correspondence, and one of them is the most accurate model in the project

Result: `B-stream-scores` and `B-training-histories`. The emotion stream reads
0.9991 on holdout and 0.9978 in-domain, the best of the five on both.

`diagonal_mass` is the fraction of attention mass near the diagonal, recorded
every epoch and never optimised, which is what makes it evidence rather than a
target. Its chance value depends on the query length: uniform attention scores
0.0592 at 50 query steps and 0.3438 at 8.

Measured across every epoch of three separate training runs:

| Stream | Query steps | Chance | Measured range |
| --- | --- | --- | --- |
| emotion | 8 | 0.3438 | 0.3434 to 0.3457 over 9 epochs |
| lip-sync | 50 | 0.0592 | 0.0592 to 0.0595 over 4 epochs |
| lip-sync, lower encoder LR | 50 | 0.0592 | 0.0592 to 0.0595 over 6 epochs |

Both streams sit on chance and stay there. Emotion's ROC-AUC rose from 0.9847 to
0.9991 across its nine epochs without its attention moving at all.

The accuracy is real and the stated mechanism is not operating in either stream.
This is the paper's central instance: an accuracy table ranks emotion first and
says nothing about the fact that the cue it is named for is absent.

This also corrects an earlier reading of the same number. Emotion's 0.3439 was
first called six times chance, against lip-sync's 0.0592 baseline. The two
streams use different query lengths, so that comparison was wrong: at 8 steps
0.3438 is chance, and emotion sits on it.

### F1a. A shortcut-controlled corpus does not rescue the mechanism either

Result: `B-lavdf-lipsync`. This is the strongest form of F1 and it rules out the
comfortable explanation.

The obvious defence of F1 is that FakeAVCeleb does not require correspondence:
its fakes are whole-clip manipulations, so a model can separate them by
recognising the generator and never needs to compare mouth to sound. LAV-DF is
built to remove exactly that escape. Its adapter cuts two windows from the same
file: one containing the manipulated span, one avoiding every manipulated span.
Same speaker, same lighting, same microphone, same codec, same re-encoding pass.
The two windows differ in one thing only.

A lip-sync stream trained on those pairs reads **0.9969 ROC-AUC** on 1,161
held-out windows, and its `diagonal_mass` is **0.0592 against a chance of
0.0600**. At chance, on the corpus designed to make correspondence the only
available cue.

So the model is not failing to learn correspondence because the data let it
cheat. On data built to forbid cheating it still solved the task by some other
route, and reached near-perfect accuracy doing it. Whatever is diagnostic inside
a manipulated window, whether the TTS audio's texture or the reenactment's
rendering, is being read instead of the alignment between the two streams.

This is the finding to lead with. A validity check that only fired on a
convenient dataset would be weak evidence. One that fires at 0.9969 on the
dataset specifically constructed to prevent the shortcut is the paper's claim in
a single measurement.

### F1b. Lip-sync's collapse is head overfitting, not encoder instability

Result: `B-training-histories`. The lip-sync stream reaches 0.7593 at epoch 1
and degrades every epoch after. The obvious reading was that unfreezing the two
pretrained encoders destabilised it, so it was retrained at an encoder learning
rate of 1e-6 instead of 5e-6, with three frozen epochs instead of two and
patience 5 instead of 3.

It made no difference, and the reason is visible in the epoch where it breaks:

| Epoch | Encoders | Train loss | Validation loss | ROC-AUC |
| --- | --- | --- | --- | --- |
| 1 | frozen | 0.5628 | 0.5746 | 0.7593 |
| 2 | frozen | 0.4710 | 2.1972 | 0.5077 |
| 3 | frozen | 0.2620 | 2.2333 | 0.5217 |
| 4 | on | 0.1620 | 3.8077 | 0.4090 |
| 5 | on | 0.1030 | 2.9206 | 0.4838 |
| 6 | on | 0.0589 | 3.0748 | 0.4474 |

Validation loss quadruples at epoch 2, while the encoders are still frozen. The
encoder learning rate cannot be the cause of a collapse that happens before the
encoders train. The head alone overfits, on a cue the attention map says it is
not reading.

Both runs select epoch 1, so the shipped model is unchanged. The hypothesis was
wrong and the experiment that tested it is worth reporting for that reason.

### F2. Freezing BatchNorm while fine-tuning trades transfer for in-domain fit

Result: `B-batchnorm-arms`, at full scale, and `B-batchnorm-trade` for the
controlled 1,400-clip sweep that isolates the epoch budget.

Both visual streams trained twice on the full training partition, changing one
flag. Labels are `video_fake`, the objective the visual stream was trained on:

| Arm | Stream | Validation | In-domain | DFDC |
| --- | --- | --- | --- | --- |
| frozen | EfficientNet-B0 | 0.9997 | 0.9987 | 0.5067 |
| live | EfficientNet-B0 | 0.4909 | 0.4557 | 0.6343 |
| frozen | DINOv3, frozen ViT | 0.9559 | 0.9546 | 0.5147 |
| live | DINOv3, frozen ViT | 0.9559 | 0.9546 | 0.5147 |

Live BatchNorm moves EfficientNet by +0.1276 on DFDC and -0.5430 in-domain. The
DINOv3 rows are identical to four decimals on every epoch of both runs, which is
what rules out run-to-run variation: a frozen ViT has no running statistics, so
the flag must change nothing there, and it changes nothing.

Live BatchNorm statistics adapt to the data the model sees, which is what AdaBN
does deliberately, and freezing them removed an accidental domain adapter.

The trade is too steep to take. An in-domain ROC-AUC of 0.4557 is below chance,
so the live arm is not a better-generalizing detector, it is a broken detector
that happens to rank slightly above chance out of domain. The project ships the
frozen arm and reports the trade. What the measurement establishes is the size
of the lever, not a setting to adopt.

The 1,400-clip sweep adds the control the full-scale comparison lacks: the live
arm stopped at epoch 4 on patience while the frozen arm ran 10, so the epoch
budget is not held fixed above. In the sweep it is, and it moves DFDC by 0.005
against BatchNorm's 0.148.

This supersedes the earlier reading that freezing BatchNorm was a clean win. It
was recorded as one because it took a visual stream from 0.5154 to 0.9058 on
validation, and validation was the only thing measured at the time.

### F2b. The same checkpoint has two in-domain numbers, and both are correct

Result: `B-batchnorm-arms` against `B-stream-scores`. The visual stream reads
0.9987 in-domain when scored against `video_fake`, the label it was trained on,
and 0.9719 when scored against `clip_fake`, the label the fusion store carries.

A FakeAVCeleb clip with a real video track and spoofed audio is `clip_fake` and
not `video_fake`, so the two labels disagree on exactly the clips a visual model
cannot be expected to catch. They coincide on DFDC, where every manipulated clip
has a manipulated video track, which is why the DFDC column agrees to four
decimals between the two scoring paths.

Neither number is wrong. Quoting them in the same table without saying which
label each uses would be.

### F3. Selecting a checkpoint on validation loss kept one scoring below chance

Result: superseded rows under
`runs/design-b-20260910/checkpoints-loss-selected/`, kept for the comparison.

`visual-efficientnet` reached its lowest BCE at epoch 1 with the backbone still
frozen, and that checkpoint measured 0.4668 in-domain, below chance. On one run
loss and AUC moved in opposite directions on the same epoch: epoch 2 had the
worse loss, 0.3931 against 0.3296, and the better AUC, 0.9547 against 0.9428.

BCE measures calibration and punishes a confident mistake hard. A model that
ranks well while being overconfident loses to one that hedges and ranks badly.
Every objective in this project is stated in ROC-AUC, so both trainers now
select on it.

### F4. Fusing a subset beats fusing everything, cross-corpus

Result: `B-deep-ablation`. All 31 subsets of the five streams, each with its own
head fitted on the same holdout rows and scored on the same two partitions.

| Partition | Best combination | Best single | All five |
| --- | --- | --- | --- |
| in-domain | audio + emotion + visual-efficientnet, 0.9990 [0.9976, 1.0000] | emotion, 0.9978 | 0.9988 |
| DFDC | emotion + lip-sync, 0.6076 [0.5666, 0.6488] | emotion, 0.5991 | 0.5411 |

On DFDC the best combination's interval, [0.5666, 0.6488], contains the best
single stream's 0.5991, so the two are not separated. The claim that survives is
the negative one: the full five-stream fusion at 0.5411 sits below three of its
own inputs.

Fusion beats the best single stream on both partitions. The full fusion does
not: on DFDC it falls below three of its own inputs. More streams is not the
lever, and a system that always fuses everything it has is choosing the worse
of two options it already holds.

### F5. Redundancy is measurable and was not where it was predicted

Result: `B-stream-correlation`. Lip-sync shares 82.5 percent of its errors with
the audio branch in-domain, correlation 0.820, and is anti-correlated with the
visual stream at -0.160 with 26.2 percent shared errors.

The first reading of this was that lip-sync is a redundant audio duplicate and
should be dropped. That was wrong: lip-sync is in the best DFDC combination
(F4). Redundancy in-domain and usefulness cross-corpus are different questions,
and the error-overlap table answers only the first.

### F6. Motion hides manipulation, it does not create false alarms

Result: `A-motion`. Measured as the mean absolute difference between consecutive
frames of the cached view, which is what the model is handed, against the fake
probability it returns.

| Corpus | Mean motion | Top quartile | Correlation, manipulated clips | Correlation, authentic clips |
| --- | --- | --- | --- | --- |
| FakeAVCeleb, in-domain | 0.2429 | 0.3038 | -0.202 | -0.010 |
| DFDC, cross-corpus | 0.2710 | 0.3627 | -0.316 | -0.035 |

The sign is negative and it lives entirely in the manipulated class. More motion
means a lower fake score, on both corpora, and motion barely moves the score on
genuine clips at all. DFDC's upper quartile carries 19 percent more motion than
FakeAVCeleb's, so the clips that corpus adds are exactly the ones the model is
least likely to call fake. That is a mechanism for the transfer gap and it lines
up with the operating point: on DFDC at threshold 0.5 the pipeline has precision
0.9904 and recall 0.2820, so it misses 72 percent of manipulated clips while
almost never crying wolf.

**This supersedes the mechanism written into earlier drafts.** The claim there
was that sampling 16 frames from across a clip makes genuine camera motion look
like the flicker of a forgery, producing false alarms. Measured, false alarms on
genuine clips rise only from 0.0 percent in the calmest quartile to 2.7 percent
in the most moving one on DFDC, and not at all in-domain. The dominant failure
on moving video is the opposite of the one claimed: missed detections, not false
positives.

### F7. One decision threshold cannot serve three media kinds

Result: `B-gate-thresholds`. Thresholds chosen by Youden's J on holdout rows the
head was not fitted on: 0.50 for a fused video, 0.76 for an image scored through
the visual stream alone, 0.26 for sound alone. Youden's J at those cut-offs is
0.994, 0.939 and 0.513.

An image and a video reach the head through the same weights with different
streams present, so their fused probabilities sit on different scales. Serving
one threshold would systematically under-call one kind and over-call another.

### F8. The deep head does not beat the late head, cross-corpus

Result: `B-fusion-ablations`. Both heads fitted on the same holdout rows and
scored on the same partitions. The late head calibrates each stream's logit and
fits a logistic regression over the calibrated scores plus three quality
features; the deep head projects each stream's embedding and fuses in feature
space.

| Partition | Late fusion | Deep fusion |
| --- | --- | --- |
| in-domain | 0.9989 [0.9974, 1.0000] | 0.9988 [0.9971, 1.0000] |
| DFDC | 0.5687 [0.5279, 0.6052] | 0.5411 [0.4984, 0.5828] |

In-domain they are indistinguishable. On DFDC the late head reads 0.0276 higher,
with intervals that overlap heavily, so this separates nothing. What it does
rule out is the reason the deep head exists: the embedding was expected to carry
something the scalar logit does not, and on these two partitions it does not
show.

### F9. Abstaining on partial coverage buys 0.0007 AUC and costs 2.4 percent of the clips

Result: `B-fusion-ablations`. On DFDC the deep head reads 0.5418 when it refuses
every clip missing a stream, covering 97.6 percent, and 0.5411 when it answers
from the streams that ran, covering all of them.

The abstention policy is still the right default, because a verdict with no
evidence behind it is worse than a refusal. But it should not be argued for on
accuracy: it does not buy accuracy here. The 2.4 percent of DFDC clips with
partial coverage are not the clips the head is getting wrong.

### F10. Training on the standard cross-dataset corpus did not improve transfer

Result: `C-ffpp-zeroshot`. The visual stream trained on FaceForensics++ c23 and
scored on corpora it never saw, against the same stream trained on FakeAVCeleb
and scored on the same clips:

| Trained on | FF++ test | Celeb-DF-v2 | DFDC | FakeAVCeleb test |
| --- | --- | --- | --- | --- |
| FF++ c23 | 0.8723 [0.8425, 0.9008] | 0.6348 [0.5766, 0.6920] | 0.4920 [0.4478, 0.5350] | 0.8253 [0.7921, 0.8638] |
| FakeAVCeleb, Design A | not scored | 0.6682 | 0.7611 | 0.9988 |

Three things in that table.

**The DFDC column goes the wrong way.** The FF++ model reads 0.4920 with an
interval spanning 0.5, which is chance. The FakeAVCeleb model reads 0.7611 on
the same clips. Training on the corpus the literature treats as the
cross-dataset standard made transfer to DFDC worse, not better.

**Celeb-DF is a tie.** 0.6348 against 0.6682, intervals overlapping heavily. The
expected benefit of FF++'s six manipulation families over FakeAVCeleb's does not
appear.

**Each model is best on its own corpus and second best on the other's.** The
FF++ model reads 0.8253 on FakeAVCeleb, well below FakeAVCeleb's own 0.9988 but
far above its 0.4920 on DFDC. So the transfer penalty is not a fixed property of
leaving the training corpus; it depends on which corpus you leave it for, and
DFDC is the hard one for both.

This is the direct answer to whether more varied training data fixes
generalization, and on this evidence it does not. The practical reading is that
the failure is not a shortage of manipulation families in training. Both corpora
are studio-grade face manipulations; DFDC is consumer video, and neither
training set contains anything like it.

Caveats, all of which cut against over-reading the table. One seed. FF++ c23
carries no audio, so this is the visual stream alone against a Design A number
that is also visual-only but was trained with a GRU rather than an LSTM. The
DFDC subset here is 94 percent manipulated across 150 identities and is not the
standard DFDC test set, so the absolute numbers are comparable within this
project and only roughly comparable with published ones.

### F11. The identity-strict condition does not change the number

Result: `C-ffpp-zeroshot`. The protocol requires an identity-strict stress test
and it had never been run. FF++ names a fake `target_source`, and the ordinary
split groups on the target, the identity being replaced, so a clip's face donor
can sit in another partition. The strict subset keeps only clips where both
identities fall inside the same partition, so nothing in it shares a source or a
target with anything the model trained on.

| Test set | Clips | ROC-AUC |
| --- | --- | --- |
| FF++ test, ordinary split | 954 | 0.8723 [0.8425, 0.9008] |
| FF++ test, identity-strict | 292 | 0.8761 [0.8039, 0.9356] |

The point estimates are within 0.004 of each other and the intervals overlap
almost entirely. The strict interval is wider because the subset is a third of
the size, which is the cost of the stricter condition rather than a result.

The finding is the absence of a finding, and it is worth stating: the ordinary
split's number was not inflated by the face donor leaking across partitions.
That was a live risk, it is what the strict subset exists to detect, and on this
corpus it did not happen.

## Superseded findings

| Claim | Replaced by | Why it was wrong |
| --- | --- | --- |
| Emotion's diagonal mass is six times chance | F1 | Compared against lip-sync's baseline; the two streams use different query lengths, and at 8 steps 0.3438 is chance |
| The emotion stream duplicates the visual stream | F1, F4 | It is the best single stream on both partitions and appears in the best combination on each |
| Lip-sync is a redundant audio duplicate and should be dropped | F5, F4 | It carries the best DFDC combination |
| Freezing BatchNorm is a clean win | F2 | It is a trade: in-domain fit for cross-corpus transfer |
| Training on a more varied corpus would improve transfer | F10 | FF++ c23 has six manipulation families against FakeAVCeleb's, and transfer to DFDC got worse, from 0.7611 to 0.4920 |
| Camera motion makes the model call genuine clips fake | F6 | Measured with the opposite sign: motion lowers the fake score, and the failure on moving video is missed detections, not false alarms |
| A clip-level bootstrap would report intervals several times too narrow | none | Measured the other way: the identity-clustered interval on DFDC is narrower, 0.0776 against 0.1065, because identities there carry balanced class proportions. Clustering by identity is still right, because it matches the question, not because it is wider |
