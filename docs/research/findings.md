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

### F1. A stream can reach 0.9991 ROC-AUC with attention that never moved

Result: `B-stream-scores`. The emotion stream reads 0.9991 on the holdout
partition and 0.9978 in-domain, the best of the five on both.

`diagonal_mass` is the fraction of attention mass near the diagonal, recorded
every epoch and never optimised, which is what makes it evidence rather than a
target. Uniform attention scores 0.0592 at 50 query steps and 0.3438 at 8.
Measured: lip-sync 0.0592 at 50 steps, emotion 0.3439 at 8. Emotion's AUC rose
from 0.9847 to 0.9991 across nine epochs while its diagonal mass stayed between
0.3435 and 0.3457.

The accuracy is real and the stated mechanism is not. The stream is named for
audiovisual correspondence and is not measuring it.

This also corrects an earlier reading of the same number. Emotion's 0.3439 was
first called six times chance, against lip-sync's 0.0592 baseline. The two
streams use different query lengths, so that comparison was wrong: at 8 steps
0.3438 is chance, and emotion sits on it.

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
| in-domain | audio + emotion + visual-efficientnet, 0.9990 | emotion, 0.9978 | 0.9988 |
| DFDC | emotion + lip-sync, 0.6076 | emotion, 0.5991 | 0.5411 |

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

### F6. Generalization collapses in a specific, mechanistic way

Result: `A-fusion-in-domain` and `A-fusion-dfdc`. The Design A pipeline reads
0.9990 in-domain and 0.7500 on DFDC. On MNW, which is fake-only, it detects 25
of 85 manipulated clips, and the breakdown is not uniform: 0 of 10 on `vasa_1`,
0 of 4 on `raskai`, 1 of 6 on `diff2lip`, against 7 of 8 on the unnamed
in-the-wild clips.

`visual_view` samples 16 frames from across the whole clip, so genuine camera
motion produces the same frame-to-frame variation a flickering forgery does.
The failure is not uniform across generators, and the generator it misses
completely is the one whose artifacts are temporally smooth.

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

## Superseded findings

| Claim | Replaced by | Why it was wrong |
| --- | --- | --- |
| Emotion's diagonal mass is six times chance | F1 | Compared against lip-sync's baseline; the two streams use different query lengths, and at 8 steps 0.3438 is chance |
| The emotion stream duplicates the visual stream | F1, F4 | It is the best single stream on both partitions and appears in the best combination on each |
| Lip-sync is a redundant audio duplicate and should be dropped | F5, F4 | It carries the best DFDC combination |
| Freezing BatchNorm is a clean win | F2 | It is a trade: in-domain fit for cross-corpus transfer |
| A clip-level bootstrap would report intervals several times too narrow | none | Measured the other way: the identity-clustered interval on DFDC is narrower, 0.0776 against 0.1065, because identities there carry balanced class proportions. Clustering by identity is still right, because it matches the question, not because it is wider |
