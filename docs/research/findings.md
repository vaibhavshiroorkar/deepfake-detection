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

One, and it is the finding about the seeds themselves. `A-visual-seeds` is the
only row in the registry at three seeds, because the visual branch was the only
branch `run_program.ps1` trained three times. See F21.

Every other row is still `provisional (1 seed)`, and F21 is the measurement of
what that costs: 0.0733 of ROC-AUC on DFDC, which is wider than most of the
effects the other findings report.

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
budget is not held fixed above. It was quoted here from a scratch run whose
output was never written under `runs/`. It has now been run and recorded, and
**it does not reproduce the size of the effect**:

| Arm | Epochs | Validation | DFDC |
| --- | ---: | ---: | ---: |
| frozen BatchNorm | 10 | 0.9072 | 0.5508 |
| live BatchNorm | 10 | 0.5500 | 0.5686 |
| frozen BatchNorm | 3 | 0.8728 | 0.5466 |

With the epoch budget held at 10, BatchNorm moves DFDC by **+0.0177**, not the
+0.1276 the full-scale comparison showed. Holding BatchNorm frozen and varying
epochs instead moves it by 0.0042, so the epoch budget is not the explanation
either. Neither lever is worth what the full-scale numbers implied.

What survives intact is the direction and the in-domain cost: live BatchNorm
still collapses validation, 0.9072 to 0.5500, a fall of 0.3572 to near chance.
The trade is real and still too steep to take. Its cross-corpus half is roughly
seven times smaller than the uncontrolled comparison suggested, which is the
same lesson as F21 and F24: an effect measured without its control is usually
measuring something else as well.

These are 1,400-clip arms, so they sit below full-scale numbers in absolute
terms and should be read as the difference between arms rather than as
performance.

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

### F12. Capture conditions bind generalization, not manipulation type or training protocol

Result: `B-lavdf-transfer`. The lip-sync stream trained on LAV-DF's matched
windows, scored on three corpora without retraining:

| Scored on | ROC-AUC | Clips | Diagonal mass |
| --- | --- | --- | --- |
| LAV-DF, in-domain | 0.9969 | 1,161 | 0.0592 |
| FakeAVCeleb, never seen | 0.7407 | 1,648 | 0.0592 |
| DFDC, never seen | 0.5059 | 2,351 | 0.0592 |

The middle row looks like a breakthrough and is not. A stream that never saw
FakeAVCeleb reads 0.7407 on it, against 0.7587 for a stream trained on
FakeAVCeleb itself. Transfer is nearly free.

**LAV-DF and FakeAVCeleb are both built on VoxCeleb2.** They share their real
footage, so they share speakers, lighting, microphones, codecs and capture
pipeline. That row is not cross-capture transfer. It is transfer across
manipulation types inside one capture distribution, and it is free because the
hard part was never varied.

DFDC is the only audio-bearing corpus here not built on VoxCeleb2, and both
streams sit at chance on it: 0.5059 for the LAV-DF-trained one and 0.4890 for
the FakeAVCeleb-trained one.

So the binding constraint is the capture pipeline. It is not the manipulation
type, which transfers freely within a capture distribution. It is not the
training protocol either: matched-pair training, which is the most
shortcut-controlled setup in the project and which produced 0.9969 in-domain,
buys nothing at all once the capture conditions change.

This is the third independent line of evidence for the same conclusion. The
family holdout (F34) said an unseen manipulation family costs 0.12 on average
and five of six still score above 0.70, while an unseen corpus reads 0.49. Training on FaceForensics++ instead of
FakeAVCeleb moved DFDC from 0.7611 to 0.4920. And now a corpus change inside
VoxCeleb2 costs almost nothing while a corpus change outside it costs
everything.

Diagonal mass sat at 0.0592 against a chance of 0.0600 on all three partitions,
so whatever transfers between the two VoxCeleb2 corpora, it is not
correspondence.

### F13. The region of interest decides the result, and there is no single right one

Result: `C-ffpp-logo`. A frozen DINOv3 probe over full frames, under
source-grouped leave-one-family-out on FaceForensics++:

| Held-out family | ROC-AUC |
| --- | --- |
| NeuralTextures | 0.5635 [0.4583, 0.6698] |
| Deepfakes | 0.5553 [0.4505, 0.6703] |
| Face2Face | 0.5477 [0.4506, 0.6539] |
| FaceSwap | 0.5281 [0.4217, 0.6397] |
| DeepFakeDetection | 0.5204 [0.4367, 0.6023] |
| FaceShifter | 0.5169 [0.4104, 0.6200] |
| macro mean | 0.5387 |

Every interval contains 0.5. The probe is at chance.

The cause is almost certainly the input rather than the model. FF++ manipulates
the face alone, and a 1280x720 frame resized to 224 leaves the face a few dozen
pixels across, so the evidence is gone before the encoder sees it. On the same
corpus and the same held-out-family protocol, face crops through a fine-tuned
EfficientNet read 0.7077 to 0.9322 on five of six families (F34). Two variables
differ between those rows, the region and the model, so the attribution is not
clean, but the direction is. The sixth family, FaceSwap, is at chance under both
(0.5281 full frame, 0.5195 face crop), so for that family neither the region
nor the model rescues it.

The design consequence is that the full-frame choice is not universally right.
It is right for media generated whole, where the entire scene is synthetic, and
wrong for face manipulation, where the evidence is local. A system covering both
needs both paths, routed by what the input contains, rather than one ROI chosen
on principle.

This also corrects the way the full-frame choice was argued earlier in this
project. It was justified by the Veo 3 coverage measurement, where face-dependent
branches answered on 52.5 percent of clips. That argument is still right about
person-free generated video and says nothing about face manipulation, which is
what this measures.

A secondary result worth keeping: the pipeline returned 0.54 with wide intervals
on a corpus where a leak would have been easy, after two leaks were found and
closed. An honest near-chance number is evidence the harness works.

### F14. DF26 cannot support audiovisual work, and would have looked like it could

Result: `C-df26-audio-audit`. Audio-track presence in DF26, sampled 30 clips per
group:

| Group | With an audio track |
| --- | --- |
| real | 3 of 30 |
| Grok Imagine, Kling 3.0, Veo 3.1, Wan 2.6 | 30 of 30 each |
| HunyuanVideo 1.5, LTX 2.3, Wan 2.2 | 0 of 30 each |

Audio presence predicts the label almost perfectly. A single boolean, whether
the container carries an audio stream, separates the commercial generators from
everything else. Any audio branch scored on DF26 would report near 1.0 while
detecting a container property, and it would have arrived looking like exactly
the contribution the paper invites: its limitations section names audio, speech
quality and lip-sync as future work.

Only 27 of the 271 real clips carry audio at all, so even after discarding the
open-source generators there is no usable authentic-audio comparison set.

The corpus is visual-only in a stronger sense than its authors stated. They
describe not having *evaluated* audio; the data cannot support the evaluation.
The audiovisual track therefore needs a different corpus, and the candidates are
in [corpora](datasets.md): DeepSpeak v2 for webcam speech, AV-Deepfake1M for
localized audiovisual edits, and LAV-DF which is already on disk.

The pattern is worth naming because it recurs: three times today a corpus
promised a result and delivered a shortcut instead. Veo 3 lab clips against
talking-head reals, real clips on both sides of a generator split, and now audio
presence as a label proxy. Each was visible in one cheap check run before
training rather than after.

### F15. First leave-one-generator-out result on DF26, pending a frame-rate control

Results: `C-df26-logo`, and `C-df26-logo-controlled` for the control below.
A frozen DINOv3 probe over full frames, trained on six
generators and scored on the seventh, with source videos split so no source
appears on both sides and generation modes grouped under their model.

| Held-out generator | ROC-AUC |
| --- | --- |
| Wan 2.6 | 0.7621 [0.6808, 0.8322] |
| Veo 3.1 | 0.7516 [0.6781, 0.8202] |
| Grok Imagine | 0.7316 [0.6556, 0.8093] |
| Kling 3.0 | 0.7216 [0.6445, 0.7928] |
| LTX 2.3 distilled | 0.6325 [0.5646, 0.7024] |
| HunyuanVideo 1.5 | 0.6316 [0.5616, 0.7010] |
| Wan 2.2 | 0.6285 [0.5596, 0.6970] |
| macro mean | 0.6942 |

Every interval clears 0.5, and the macro sits level with the best published
FF++-trained detector on DF26 at 0.697.

**Control run, and it found a real confound.** Re-encoding every clip to 640x360
at 24 frames per second and re-running gives macro 0.6786 against 0.6942. The
loss is concentrated: Wan 2.6, the only generator at 30 fps, fell 0.094, and
HunyuanVideo fell 0.045, while the other five moved by less than 0.014. So frame
rate was leaking, through one generator, and the controlled table is the one to
report:

| Held-out generator | Controlled ROC-AUC |
| --- | --- |
| Veo 3.1 | 0.7622 [0.6890, 0.8255] |
| Grok Imagine | 0.7308 [0.6466, 0.8057] |
| Kling 3.0 | 0.7217 [0.6429, 0.7911] |
| Wan 2.6 | 0.6686 [0.5871, 0.7484] |
| Wan 2.2 | 0.6414 [0.5772, 0.7062] |
| LTX 2.3 distilled | 0.6386 [0.5721, 0.7031] |
| HunyuanVideo 1.5 | 0.5867 [0.5137, 0.6544] |
| macro mean | 0.6786 |

**What remains unexplained.** The four commercial generators score 0.72 to 0.76
and the three open-source ones 0.63, which inverts the paper's finding that
closed-source generators are markedly harder, 34.5 and 48.3 against 57.4 and
70.5. When a result inverts a published one the first suspect is a shortcut.

Container properties nearly identify the group. Real clips run at 25 or 29.97
frames per second and nothing else does; generated clips run at 24 except Wan
2.2 at 16 and Wan 2.6 at 30, with durations from 81 to 150 frames. The feature
vector includes a spread-across-windows term that is motion-sensitive, and frame
rate changes apparent motion directly, so frame rate is a plausible path into
the score.

The commercial-versus-open-source inversion survives the control, 0.72 against
0.62. The remaining explanation is the training set rather than a shortcut:
the paper's detectors transfer from face manipulation while this probe transfers
between generative models, and the four commercial generators were all produced
through one platform and may be more similar to each other than the open-source
three are. That is untested and worth testing before the inversion is quoted.

### F16. False alarms on genuine video range from 0 to 91 percent with capture conditions

Result: `A-false-alarms`. Genuine clips called fake at the fixed 0.50 threshold:

| Corpus | Genuine clips | Called fake |
| --- | --- | --- |
| FakeAVCeleb test, in-domain | 75 | 0.00 percent |
| DFDC, cross-corpus | 145 | 4.14 percent |
| Celeb-DF, cross-corpus | 155 | **90.97 percent** |

On Celeb-DF the visual branch calls 141 of 155 authentic videos manipulated. Its
genuine clips are YouTube celebrity interviews rather than FakeAVCeleb's
recordings, and that difference alone moves the false-alarm rate by ninety-one
points.

The cost of removing the false alarms is the other half:

| Corpus | Threshold for zero false alarms | Manipulated clips still caught |
| --- | --- | --- |
| FakeAVCeleb | 0.1311 | 99.8 percent |
| DFDC | 0.8777 | 12.2 percent |
| Celeb-DF | 0.9998 | 0.0 percent |

On Celeb-DF no threshold gives both zero false alarms and any detection. The two
score distributions overlap almost completely, so the ordering carries no usable
information there, which the 0.6682 ROC-AUC on that corpus already said in a way
that is easier to overlook.

Three consequences for anything deployed.

**A guarantee of never accusing a genuine video is not available from a
classifier.** Every threshold trades false alarms against detection, and on
out-of-distribution capture the trade has no acceptable point.

**The thresholds in this project optimise the wrong thing for that requirement.**
They are chosen by Youden's J, which weights sensitivity and specificity
equally. A deployment that cannot accuse a genuine video needs the threshold
chosen at a target false-alarm rate on a held-out set of real clips that match
the deployment's capture conditions.

**The honest output is three-valued, not two.** "Manipulated" when the evidence
clears a high-precision bar, "no evidence of manipulation" otherwise, and
"outside what I was trained on" when the distribution gate fires. Calling the
second of those "real" is the claim the measurements do not support.

### F17. A density model over authentic footage cuts false accusations by four fifths

Result: `A-authenticity-density`. F16 measured the problem: 90.97 percent of
genuine Celeb-DF clips are called fake, and no threshold fixes it because the
score distributions overlap. Two gates were tried.

**The motion gate intercepts 5.7 percent of those false alarms.** Celeb-DF's
genuine clips sit at motion 0.2964, inside the training range of 0.1017 to
0.4789. They are unfamiliar in appearance, not in motion, and one hand-picked
statistic cannot see that.

**A Gaussian fitted over frozen encoder features of authentic clips only, scored
by Mahalanobis distance, does much better.** No fake example is involved, so it
generalises to capture conditions nobody anticipated.

| Abstention on familiar genuine footage | Unfamiliar genuine clips intercepted |
| --- | --- |
| 5.3 percent | 64.4 percent |
| 8.7 percent | 79.4 percent |
| 21.3 percent | 90.6 percent |

At the middle setting the false-accusation rate on unfamiliar real video falls
from about 91 percent to roughly 19 percent, and to about 9 percent if
abstaining on a fifth of familiar clips is acceptable.

Two things this cost to get right, both my own errors caught by a check rather
than by reasoning.

The first threshold was set on the clips the model was fitted on, and held-out
clips from the *same* corpus then flagged at 100 percent. A Gaussian over 768
dimensions fitted on 250 samples reports optimistically small in-sample
distances. The fix is PCA to 48 components and a threshold set on a held-out
half of the reference.

The second run still showed 26.7 percent on held-out familiar clips against a 5
percent target, which looked like residual miscalibration and was not. The
reference and held-out sets were the first 250 and next 150 entries of a
directory listing ordered by demographic folder, so they were different subsets
of the corpus rather than an unbiased split. Shuffling first brings the measured
rate to 5.3 percent against the 5 percent target.

**What it does not do.** It does not reach zero, and nothing does: 9 percent of
unfamiliar genuine clips still get accused at the most aggressive setting. It
also buys the improvement with abstentions, which are only useful if the system
is allowed to answer "outside what I was trained on" rather than being forced to
choose.

### F18. The logistic head ships, the MLP collapses cross-corpus, and neither calibration nor the quality features buys any ranking

Result: `A-fusion-model-ablation`. FUS-01 had sat `planned` since the roadmap was
written even though `train fusion --model mlp` existed the whole time. Five
heads over the same out-of-fold features, so only the head varies.

| Head | in-domain | DFDC | DFDC ECE |
| --- | --- | --- | --- |
| logistic, shipped | 0.9990 [0.9974, 1.0000] | 0.7500 [0.7092, 0.7926] | 0.6366 |
| small MLP | 0.9992 [0.9979, 1.0000] | 0.5992 [0.5515, 0.6508] | 0.7872 |
| raw logits, no calibration | 0.9995 [0.9986, 1.0000] | 0.7536 [0.7139, 0.7943] | 0.7421 |
| no quality features | 0.9989 [0.9973, 1.0000] | 0.7512 [0.7101, 0.7939] | 0.6347 |
| unweighted mean, no head | 0.9911 [0.9876, 0.9944] | 0.7561 [0.7173, 0.7974] | 0.0554 |

Paired against the logistic head over the same clips, clustered on speaker:

- The MLP is indistinguishable in-domain, +0.0003 [-0.0010, +0.0016], and loses
  -0.1508 [-0.2052, -0.0995] on DFDC. It is the only arm that separates
  downward anywhere. Eight hidden units were enough to fit the in-domain
  structure and carry none of it across. The scope control says prefer the
  cheaper model when results are indistinguishable; here the cheaper model also
  wins outright, so the choice is not close.
- Dropping calibration changes no ranking, +0.0005 [+0.0000, +0.0014]
  in-domain and +0.0037 [-0.0027, +0.0101] on DFDC, neither separated. Platt
  scaling per branch is monotone, so it cannot reorder one branch; what it can
  do is change how the head weighs them, and measurably it does not. It earns
  its place on calibration error alone, 0.6366 against 0.7421 on DFDC.
- The quality features buy +0.0012 [+0.0004, +0.0020] on DFDC and nothing
  in-domain. The interval clears zero and the effect is smaller than the fourth
  decimal the rest of this project reports. Three features for that.

The floor arm is the one worth keeping. An unweighted mean of the calibrated
branch probabilities, with nothing learned at all, is not separated from the
fitted head on DFDC, +0.0062 [-0.0017, +0.0139], and its calibration error there
is **0.0554 against the fitted head's 0.6366**, better by a factor of eleven.
The head's learned weights are the part that fails to cross corpora, not the
branches underneath it. In-domain the fitted head is genuinely better,
-0.0078 [-0.0107, -0.0050] for the mean, which is the same in-domain against
cross-corpus disagreement this project has found everywhere else.

### F19. Fusion's in-domain gain is fifteen times larger than it looks, because the branch number is measured on a different task

Result: `A-branch-ablation`. The branch evaluation reports score each branch
against its own cue. For the visual branch a `RealVideo-FakeAudio` clip is
genuine, because the video really is unmanipulated, and on that task the branch
reads 0.9988 in-domain. Fusion answers the clip-level question, where the same
clip is fake. The identical visual logits, relabelled, read 0.9742.

Both numbers are correct and they are not comparable. Putting fusion's 0.9990
beside the branch's 0.9988 makes fusion look worth +0.0002. Measured on one
task, paired over the same clips and clustered on speaker, all three branches
beat the strongest single branch by **+0.0159 [+0.0129, +0.0190]** in-domain.

Cross-corpus the sign flips and the interval does not clear zero,
-0.0085 [-0.0196, +0.0027], which agrees with the -0.0083 already recorded
under `A-operating-point` from a different route.

The whole of the in-domain gain is the 75 `RealVideo-FakeAudio` clips, which a
visual branch cannot detect even in principle because there is nothing visual to
detect. That is the cue-label ablation the research design asks for, and the
answer is that fusion's measured benefit here is exactly the cue coverage it was
built for, not a general accuracy improvement.

### F20. No method or subgroup gap in-domain, at a ceiling that could not show one, and an AUC that hides a threshold collapse

Result: `A-generalization-slices`. The shipped head on the in-domain test
partition, each manipulation method against the same 75 genuine clips. Five of
six methods read 1.0000 and miss nothing; `faceswap` reads 0.9671 [0.9191,
1.0000] and misses 3 of 50. Across race and gender no slice separates from any
other, and not one genuine clip is called fake in any slice.

That is not evidence of fairness. Every slice sits at or near 1.0, where a real
gap could not become visible, and the denominators are 14 to 16 genuine clips
per race group. The finding is that this partition cannot answer the question,
which is why the false-alarm work in F16 had to go to other corpora to find the
0 to 91 percent range.

The holdout arm found something the AUC column hides. Refitting the head with
one method dropped from the out-of-fold set leaves five of six methods
unchanged to four decimal places. Dropping `rtvc`, the audio-only manipulation,
costs 0.0437 of AUC, 1.0000 to 0.9563 [0.9225, 0.9845] -- and **misses 73 of
its 75 clips at the frozen 0.5 threshold**, a miss rate of 97.3 percent.

The ranking barely moves while detection collapses almost completely. The head
still scores those clips above the genuine ones and no longer scores them above
0.5. Reporting this slice on AUC alone would have called it a 4-point effect.
It is the difference between catching 75 of 75 and catching 2.

This bounds one narrow question and must not be read wider: the branch
checkpoints were trained on every method, so the held-out method is still seen
by everything upstream of the head. It says the head's audio weighting is
method-specific. It does not say anything about an unseen manipulation, which
needs the branches retrained per fold.

### F21. Seed moves the cross-corpus number by more than any effect this project has measured

Result: `A-visual-seeds`. The visual branch was trained on seeds 17, 29 and 43
by `run_program.ps1` on 2026-09-08. Only seed 17 was ever evaluated, because
stage 5 of that runner names the seed-17 checkpoint directly, so two finished
checkpoints sat unscored while every result row carried `provisional (1 seed)`
and the handoff costed the gap at eight hours of GPU.

Scored now on the two partitions the headline rests on, everything else held
fixed:

| Seed | in-domain | DFDC |
| --- | --- | --- |
| 17 | 0.9988 | **0.7611** |
| 29 | 0.9991 | 0.6963 |
| 43 | 0.9993 | 0.6878 |

In-domain the seeds agree to three decimal places, which is the ceiling this
project already knows that partition has. Cross-corpus they do not. The spread
is 0.0733 from nothing but the seed, and the mean is 0.7151.

Two things follow and both are uncomfortable.

The spread is four and a half times the in-domain fusion gain of +0.0159, and
nine times the cross-corpus fusion difference of -0.0085 that F19 reports as not
separated from zero. A single-seed cross-corpus number in this project is not
precise enough to support a comparison at the fourth decimal place, and several
numbers here are quoted at exactly that precision.

**Seed 17 is the best of the three on DFDC**, and seed 17 is the one every
cross-corpus result in this project reports. That is not misconduct: the seed
was fixed by the protocol before any of this was measured, and the other two
were trained in the same run and simply never scored. But 0.7611 has been
carried through this repository as the Design A cross-corpus baseline, and the
three-seed mean is 0.7151. Everything measured against that baseline was
measured against the most favourable draw.

It does not overturn the findings that rest on paired comparisons. Those hold
the seed fixed and resample clips, so seed variance is outside their interval by
construction rather than hidden inside it. What it does is set a floor on what
an unpaired cross-corpus comparison between two differently seeded runs can
claim, and that floor is wider than several such comparisons made here.

**The audio branch now has its three seeds too**, and they say nothing, which is
the expected result on this partition: 1.0000, 1.0000 and 0.9986 in-domain. The
audio branch has no cross-corpus evaluation at any seed, because the program
never built one, so the question the visual seeds answered cannot be asked of
audio yet.

**What the seeds do not yet settle.** All three branches now have checkpoints at
seeds 17, 29 and 43, but a checkpoint with no metrics moves no result row:

| Branch | Seeds trained | Seeds scored |
| --- | --- | --- |
| visual | 17, 29, 43 | 17, 29, 43, in-domain and DFDC |
| audio | 17, 29, 43 | 17, 29, 43, in-domain only |
| sync | 17, 29, 43 | none: `evaluate branch` takes visual or audio only |

The fusion rows stay `provisional (1 seed)` because fusion reads a feature store
exported from the seed-17 final checkpoints. Moving those rows needs the
out-of-fold and test feature stores re-exported per seed and the head refitted
on each, which is the remaining work rather than more training.

### F22. The system is wrong confidently, so abstention cannot rescue it cross-corpus

Result: `A-error-analysis`. Read off the frozen evaluation predictions at the
frozen 0.5 threshold. Confidence here is how sure the model was of whatever
answer it gave, so the two columns are directly comparable.

| Evaluation | Errors | Sure when right | Sure when wrong | Errors past 0.9 |
| --- | ---: | ---: | ---: | ---: |
| fusion, in-domain | 3 of 1,648 | 0.9990 | 0.9949 | 3 |
| fusion, DFDC | 1,590 of 2,351 | 0.8299 | **0.8742** | 951 |
| visual, DFDC | 1,942 of 2,410 | 0.8517 | **0.9244** | 1,482 |
| visual, Celeb-DF-v2 | 142 of 462 | 0.9736 | 0.9483 | 118 |
| visual, MNW | 61 of 85 | 0.8841 | 0.9468 | 53 |

On both cross-corpus sets the model is **more** confident when it is wrong than
when it is right. On DFDC 951 of 1,590 fusion errors sit past 0.9, and for the
visual branch 1,482 of 1,942.

That closes a route this project had left open. F9 found that abstaining on
partial coverage buys no accuracy, and the reply was that abstention is still
right because a verdict with no evidence is worse than a refusal. This is the
stronger version of the problem: abstaining on *low confidence* cannot work
either, because confidence does not separate the right answers from the wrong
ones once the corpus changes. There is no threshold on this score that drops
errors faster than it drops correct answers. The density model in F17 works for
exactly this reason, since it scores inputs rather than trusting the detector's
own certainty.

In-domain the picture is the opposite and equally unhelpful: 3 errors out of
1,648, all three on `faceswap`, all three confident. Three errors cannot support
any claim about confidence.

### F23. Fusion repairs 314 of the visual branch's cross-corpus errors and breaks 5, and 1,585 are beyond any head

Result: `A-error-analysis`. The same clips scored by fusion and by its own
visual branch, compared only where both scored the clip and both used the same
label convention.

| Comparison | Compared | Both wrong | Fusion repaired | Fusion broke |
| --- | ---: | ---: | ---: | ---: |
| DFDC | 2,351 | 1,585 | 314 | 5 |
| in-domain | 1,573 | 3 | 0 | 0 |

The DFDC trade is strongly asymmetric: 314 repairs against 5 breaks. Fusion is
doing real work there, which sits oddly beside the headline that fusion is not
separated from the visual branch on DFDC AUC. Both are true. Fusion fixes a
useful number of threshold decisions without improving the ranking, and ROC-AUC
only sees the ranking.

The number that bounds the project is 1,585. Those are clips where the fused
score and the visual branch are both wrong, so no fusion head over these
branches can recover them: the information is not in the inputs. Against 2,351
compared clips, 67 percent of DFDC is beyond anything the head can do, and work
on the head is work on the remaining third.

Two exclusions matter and both are printed by the script rather than assumed.
The in-domain row drops exactly 75 clips because the branch report labels
`RealVideo-FakeAudio` genuine for the visual branch and fusion labels it fake:
the F19 trap, caught automatically this time. On DFDC 59 clips are dropped
because fusion demands every branch and scores fewer clips than the visual
branch alone. Counting those as repairs would have inflated the figure to 357.

### F24. Mild recompression destroys the operating point while ROC-AUC barely moves

Result: `A-corruption-visual`. The frozen visual checkpoint on the 1,648-clip
in-domain test partition, corruption applied to the cached crop so the face
track is held fixed.

| Condition | ROC-AUC | Delta | Genuine called fake | Fakes missed |
| --- | ---: | ---: | ---: | ---: |
| clean | 0.9988 | | 0 of 150 | 3 of 1,498 |
| JPEG 75 | 0.9934 | -0.0054 | **76 of 150** | 1 |
| JPEG 50 | 0.9187 | -0.0801 | **150 of 150** | 0 |
| JPEG 30 | 0.8728 | -0.1259 | **150 of 150** | 0 |
| resize 0.50 | 0.4316 | -0.5671 | 150 of 150 | 0 |
| resize 0.25 | 0.6496 | -0.3492 | 150 of 150 | 0 |

The JPEG 75 row is the one that matters. ROC-AUC falls by 0.0054, which any
table would read as robustness, while the false-alarm rate goes from 0 to 51
percent. At JPEG 50 the model calls **every** genuine clip fake and misses
nothing, which is the behaviour of a detector that has stopped discriminating
and started saying yes. Its AUC there is still 0.9187.

This is the third time in this project that a headline metric has hidden a
threshold collapse, after the `rtvc` head holdout in F20 and the confidence
result in F22. Ranking survives degradation; the decision does not. A corruption
table reporting AUC alone would have concluded this model is robust to
recompression, and a deployment built on that conclusion would accuse every
genuine video a messaging app had recompressed.

The resize rows are not monotone: 0.50 reads 0.4316, below chance and therefore
ranking inverted, while the heavier 0.25 reads 0.6496. That is recorded as
measured rather than explained. Both are far outside anything usable.

**Scope, which is narrower than it looks.** The corruption is applied to the
224x224 crop with the face track held fixed, so this isolates the classifier
from the detector. Corrupting the source and re-running the pipeline would also
break face detection and produce abstentions this cannot see, and JPEG artifacts
at 224 are not the artifacts of compressing a 720p source and then cropping.
Every number above is an optimistic bound, not an estimate.

### F25. Audio noise fails in the opposite direction, and the silence shortcut is absent

Result: `A-corruption-audio`. Same partition, the frozen audio checkpoint.

| Condition | ROC-AUC | Delta | Genuine called fake | Fakes missed |
| --- | ---: | ---: | ---: | ---: |
| clean | 1.0000 | | 0 of 789 | 0 of 859 |
| noise 20 dB SNR | 0.9773 | -0.0227 | 0 | **857 of 859** |
| noise 10 dB SNR | 0.9189 | -0.0811 | 0 | **859 of 859** |
| noise 5 dB SNR | 0.8542 | -0.1458 | 0 | 859 of 859 |
| 0.5 s silence shift | 1.0000 | +0.0000 | 5 of 789 | 0 |

The same pattern as F24 with the sign reversed. At 20 dB SNR, which is light
background noise, ROC-AUC falls 0.0227 and the branch misses 857 of 859 fakes.
Where the visual branch degrades into calling everything fake, the audio branch
degrades into calling everything genuine. A fused system inherits both, and the
two do not cancel: they are different corruptions of different inputs, and a
real clip recompressed and a fake clip recorded noisily fail in ways that both
push toward the wrong answer.

The silence row is a negative result worth keeping. Shifting the waveform by
half a second changes nothing, 1.0000 against 1.0000 with 5 false alarms out of
789. The threat model names the silence shortcut, where audio labels leak
through leading silence, and the mitigation is the content-start offset in the
preprocessor. On this checkpoint that mitigation holds and the shortcut is not
present.

### F26. The sync branch is at or below chance on all three seeds, and this is its first measured number

Result: `A-sync-seeds`. The synchronization branch had no evaluation path:
`ddf evaluate branch` takes `visual` or `audio` only, because sync is not a
binary classifier. It predicts which of seven audio offsets aligns a mouth crop
with its waveform, plus a mismatch class. Three checkpoints were trained and
none had ever produced a metric.

`sync_anomaly_logit`, which the fusion feature export already uses, turns the
offset distribution into one scalar: the log-sum-exp of every non-aligned class
minus the aligned class. Scoring that directly measures exactly what fusion
consumes rather than inventing a new quantity.

| Seed | ROC-AUC | 95% interval |
| --- | ---: | --- |
| 17 | 0.4364 | [0.3888, 0.4832] |
| 29 | 0.4570 | [0.4136, 0.4999] |
| 43 | 0.4931 | [0.4341, 0.5524] |

Every seed sits below 0.5, and two of the three have intervals that exclude it.
The branch is not merely uninformative on this task: it ranks manipulated clips
as slightly *more* aligned than genuine ones, consistently, across independent
training runs. Seed spread is 0.0567, comparable to the visual branch's
cross-corpus 0.0733 in F21.

**The label is the caveat and it is a real one.** This scores against
`clip_fake`, matching the label the fusion store writes on its sync rows. That
is the wrong question for a `FakeVideo-FakeAudio` clip that was re-synchronised
after manipulation: such a clip is fake and may be genuinely well aligned, so
the branch is being marked wrong for a correct judgement. These numbers are a
floor, not a measurement of alignment detection, and F19 is the standing warning
about comparing a number like this against a differently labelled one.

What the result does establish is that the scalar fusion actually reads from
this branch carries no usable signal at any seed, which is consistent with the
branch ablation, where sync contributed nothing to any combination, and with F1,
where neither cross-modal stream measured correspondence either.

The scoring script validates itself against the feature store: on seed 17 it
reproduces the recorded logits to 8.5e-6 of ROC-AUC, the residual being float32
convolution nondeterminism rather than a logic difference.

### F27. The fused cross-corpus headline moves 0.1057 with the seed, and 0.7500 is the best of three

Result: `A-fusion-seed-spread`. The three final branch checkpoints at seeds 17,
29 and 43 export their own test and DFDC feature stores; the fusion head fitted
on the seed-17 out-of-fold store scores all three unchanged.

| Partition | Seed 17 | Seed 29 | Seed 43 | Spread |
| --- | ---: | ---: | ---: | ---: |
| in-domain | 0.9990 | 0.9990 | 0.9990 | **0.0001** |
| DFDC | **0.7500** | 0.6814 | 0.6442 | **0.1057** |

In-domain the seeds are identical to four decimal places. Cross-corpus they
range over 0.1057, and **0.7500, the number this repository reports everywhere
as its cross-corpus headline, is the best of the three.** The mean is 0.6919.

The spread is twelve times the -0.0085 paired difference that F19 reports
between fusion and its visual branch on this corpus, and six times the +0.0159
in-domain fusion gain. Any unpaired comparison at this corpus between two
differently seeded systems is measuring the seed first and the change second.

It also compounds: F21 found 0.0733 for the visual branch alone, and fusing
three branches whose seeds all move produces 0.1057 rather than averaging the
noise away. More branches did not buy stability here.

What survives unharmed is every paired comparison in this project. Those hold
the seed fixed and resample clips, so seed variance sits outside their intervals
by construction. F18's MLP collapse, F19's fusion gain, F23's repair counts and
the corruption results in F24 and F25 are all within-seed and are unaffected.
What does not survive is quoting 0.7500 as *the* cross-corpus number: the honest
form is the mean and the range, or the seed stated beside it.

**Scope.** Only the branch checkpoints vary. The fold models that produced the
out-of-fold features exist at seed 17 alone, so the head's weights are constant
across all three arms. A full three-seed replication needs three folds by three
branches by two further seeds, eighteen training runs that are not on disk, and
would likely widen this rather than narrow it.

### F28. The audio challenger flag would have built the wrong model, and the comparison it enables cannot be decided in-domain

Result: `A-audio-wavlm`. AUD-01 asks whether WavLM or AASIST beats Wav2Vec2 Base
under the same protocol. `train audio --audio-model` looked like it already
expressed the WavLM half.

It did not. `build_wav2vec2_audio_branch` called
`Wav2Vec2Model.from_pretrained(model_name)`, which accepts any checkpoint and
drops whatever does not fit the Wav2Vec2 body. Loaded that way
`microsoft/wavlm-base` has **94,371,712** parameters against `WavLMModel`'s
**94,381,936**: the gated relative position embeddings that distinguish WavLM
are simply absent. The flag would have produced a "WavLM result" measuring a
truncated Wav2Vec2, and nothing in the run would have said so. Transformers
prints a load report, which is the only signal, and a training log buries it.

The builder now goes through `AutoModel`, which instantiates the architecture
each checkpoint was trained as. `facebook/wav2vec2-base` resolves to the same
`Wav2Vec2Model` as before, so no shipped checkpoint changes.

With the comparison correctly expressible, it is still undecidable here:

| Encoder | ROC-AUC | Balanced accuracy | ECE | Validation loss |
| --- | ---: | ---: | ---: | ---: |
| Wav2Vec2 Base | 1.0000 | 1.0000 | 0.0002 | 0.0002 |
| WavLM Base | 1.0000 | 0.9981 | 0.0010 | 0.0014 |

Both saturate. The audio branch reaches ROC-AUC 1.0000 on this partition, so the
partition cannot separate two encoders, and the scope control about preferring
the cheaper model when results are indistinguishable resolves it by default
rather than by evidence. Wav2Vec2 stays.

This is the same ceiling problem as F20: the in-domain partition cannot answer
the question asked of it. The comparison that would discriminate is
cross-corpus, and **the audio branch has no cross-corpus evaluation at any
seed**, because the program never built one. AUD-01 is therefore recorded as
undecided rather than settled, and AASIST remains unimplemented.

### F29. VIS-01 decided, and not on accuracy: ConvNeXt diverged at one seed in three

Result: `A-visual-convnext`. ConvNeXt-Tiny against the EfficientNet-B0 baseline
at seeds 17, 29 and 43, same split, same cache, same budget of 8 epochs with one
frozen, same early-stopping rule.

| Seed | EffNet in-domain | ConvNeXt in-domain | EffNet DFDC | ConvNeXt DFDC |
| --- | ---: | ---: | ---: | ---: |
| 17 | 0.9988 | 0.9992 | 0.7611 | 0.7272 |
| 29 | 0.9991 | 0.9986 | 0.6963 | 0.6776 |
| 43 | 0.9993 | **0.9032** | 0.6878 | 0.6301 |

ConvNeXt is worse on DFDC at every seed, mean 0.6783 against 0.7151. Paired over
the same clips and clustered on speaker:

- seed 17: -0.0340 [-0.0761, +0.0066], not separated
- seed 29: -0.0187 [-0.0642, +0.0256], not separated
- seed 43: -0.0577 [-0.0973, -0.0131], separated

**The one seed that separates is the one where ConvNeXt failed to train**, and
that is the finding rather than a caveat to it. Its validation loss across the
three epochs it ran was 0.218, 0.6473, 0.7172: it diverged as soon as the
backbone unfroze, stopped on patience, and the selection rule kept epoch 1, the
frozen-backbone epoch. The shipped seed-43 ConvNeXt is therefore a
never-fine-tuned model, which is why its in-domain reads 0.9032 against its
siblings' 0.999.

EfficientNet did not do this. Its validation curves at all three seeds fall and
stay down, best epochs 3, 5 and 2 of 5, 7 and 4.

So VIS-01 resolves, and the deciding evidence is stability rather than accuracy.
On the two seeds where ConvNeXt trained successfully it is worse but not
separated, which is the undecidable comparison this finding previously recorded.
Across three seeds it also **failed outright once under an identical budget**,
and a backbone that diverges in one run of three is not a candidate to ship on
the strength of a single favourable one. EfficientNet-B0 stays.

Two things this makes visible that one seed could not. ConvNeXt's DFDC spread is
0.0971 against EfficientNet's 0.0733, inflated by the failed run. And the
failure mode is F3 again: checkpoint selection quietly kept a frozen-backbone
epoch because everything after it was worse, and nothing in the resulting
metrics announces that the model was never fine-tuned.

### F30. The audio branch is at chance cross-corpus, and the label that made this look unmeasurable was a placeholder

Result: `A-audio-crosscorpus-gap`. This finding replaces an earlier version of
itself that was wrong, and the way it was wrong is worth keeping.

The audio branch had no cross-corpus number. Counting `audio_fake` over the DFDC
manifest gives 0 of 2,410, which was read here as "DFDC contains no manipulated
audio, so no audio generalization test is possible". That conclusion was drawn
from the manifest without reading the ingestion module, which says the opposite
in its own docstring:

> A DFDC "FAKE" may have manipulated video, manipulated audio, or both, and the
> file does not say which. [...] Every forgery is written as
> `FakeVideo-RealAudio` [...] and the cue-specific `audio_fake` label it implies
> is not trustworthy here.

So `audio_fake = 0` is a placeholder meaning **cue unknown**, not a measurement
that the audio is genuine. DFDC was filmed for the challenge and its forgeries do
include swapped audio; the metadata simply does not mark which clips. The
counting was correct and the inference from it was not.

What is genuinely impossible is a **cue-specific** audio evaluation on DFDC:
scoring the audio branch against an audio-only label needs labels the corpus does
not publish. What is possible, and was available the whole time in the feature
store, is the **clip-level** evaluation:

| Partition | Audio branch ROC-AUC | Label |
| --- | --- | --- |
| FakeAVCeleb test | 0.7739 [0.7558, 0.7934] | clip-level |
| DFDC | **0.5054 [0.4747, 0.5371]** | clip-level |

The audio branch is at chance cross-corpus. Its interval contains 0.5 and is
centred on it.

That is a stronger result than the gap it replaces, and it lands on the same
conclusion by measurement instead of by absence. AUD-01 still cannot be decided
between Wav2Vec2 and WavLM, but now for a reported reason: the incumbent is at
chance on the only independent corpus available, so there is no performance there
for a challenger to improve on. It also completes the branch picture, because all
three branches now have a cross-corpus number and two of the three are at or
below chance:

| Branch | In-domain | Cross-corpus |
| --- | --- | --- |
| visual | 0.9988 | 0.7151 mean over three seeds |
| audio | 0.7739 | **0.5054**, contains chance |
| sync | 0.4364 to 0.4931 over three seeds | not scored |

A fused system built on one branch that transfers and two that do not is the
honest description of what this project built, and F23's finding that 1,585 of
2,351 DFDC errors are beyond any head follows naturally from it.

**On the mistake.** The manifest count was right, the ingestion docstring was
right, and reading only the first produced a confident claim in the opposite
direction. The project keeps that docstring precisely because the label is a
trap, and the trap still worked. It is recorded here rather than quietly fixed,
which is the same rule the superseded table below follows.

### F31. An exit criterion of this project has never been measured, and two required ablations never run

Result: `A-research-design-audit`. Audited against
[research design](../research-design.md) rather than against an external
standard, because falling short of your own stated bar is the harder finding.

**The unmeasured exit criterion.** The exit criteria say: *"The sync offset
confidence interval must exclude chance."* No such interval exists. The sync
branch's training history records `train_loss` and `validation_loss` and nothing
else: no offset accuracy, no per-class breakdown, no interval. The criterion is
not failed, it is **unmeasured**, and it is the only quantitative pass/fail
condition the design states for a branch.

What the recorded loss can and cannot say. Validation loss is 2.1843 at the best
epoch. Chance cross-entropy for the 8-way offset task is ln(8) = 2.079, so the
number looks close to chance, but it is **not comparable**: the loss is
`offset_CE + 0.1 x contrastive`, and only the total was logged. With a batch of 4
the contrastive term contributes roughly 0.14, which puts offset cross-entropy
near 2.05, marginally under chance. That is suggestive and it is not a
measurement, and it cannot be recovered from the artifacts, because the
decomposition was never written. The two terms are available in `SyncLoss`; only
`total` reached the history file.

This sits beside F26, where the anomaly logit the fusion head reads is below
chance at all three seeds. Both point the same way, and neither is the
measurement the design asked for.

**The two required ablations never run.** Of the eight the design lists:

| Required ablation | Status |
| --- | --- |
| Each branch alone | done, `A-branch-ablation` |
| Visual plus audio | done, `A-branch-ablation` |
| Visual plus sync | done, `A-branch-ablation` |
| All three branches | done, `A-branch-ablation` |
| Quality-aware abstention against silent fallback | done, `B-fusion-ablations`, F9 |
| Late fusion against the small MLP | done, `A-fusion-model-ablation`, F18 |
| Authentic-only sync learning against global fake-label tuning | **never run** |
| Leading-silence removal enabled and disabled | **never run** |

Both unrun ablations are expressible: `train sync --label-mode
authentic-offset|global-fake` and `cache build --keep-leading-silence` both
exist, and a config for the global-fake arm was even written
(`runs/full-20260904/configs/full-sync-temporal-global-fake-seed17.yaml`) and
never used. The leading-silence arm needs a second cache, which is the expensive
one; the sync label mode needs one training run, which at measured cost is twelve
minutes.

The design's own sentence after that list, "The CLI exposes these through fusion
branch selection, the fusion model option, the sync label mode, and the cache
leading-silence option", describes capability and was read here for years as
coverage. Exposing an ablation is not running it.

### F32. The sync label-mode ablation, and why the rejected arm looks better

Result: `A-sync-label-mode`. One of the two required ablations F31 recorded as
never run. The design chose authentic-offset learning over global fake-label
tuning, reasoning that "the global fake label does not mean every fake has broken
lip sync". Both arms now exist, same corpus, same budget, same seed.

| Arm | Objective | Chance | Best validation loss |
| --- | --- | ---: | ---: |
| authentic-offset, shipped | 8-way offset plus mismatch | ln(8) = 2.079 | **2.1842** |
| global-fake, rejected | binary fake or real | ln(2) = 0.693 | **0.0558** |

**These are not comparable as numbers**, and the temptation to read the second
column as "the rejected arm is twelve times better" is exactly the error to
avoid. They are different tasks with different chance levels. Each column is only
interpretable against its own chance value, and read that way:

- the shipped arm sits **at or slightly above** its chance loss, so it did not
  learn its objective
- the rejected arm sits **far below** its chance loss, so it learned its objective
  easily

That reframes the design decision rather than reversing it. The reasoning for
rejecting the global fake label was sound and is now better supported: a binary
head over these clips reaches 0.0558 because whole-clip manipulation is separable
by generator artifacts, which is precisely the shortcut F1 and F1a document for
the cross-modal streams. An arm that learns quickly on a label that does not mean
what the branch is named for is not evidence of synchronisation.

What the ablation establishes is narrower and more useful than a winner: the
branch's chosen objective is not learnable at this budget on this data, and the
alternative objective is learnable but measures something else. Combined with
F26, where the scalar fusion reads from this branch is below chance at all three
seeds, the sync branch is not viable as built and the label mode is not what makes
it so.

No ROC-AUC is reported for the global-fake arm. `sync_anomaly_logit` assumes
offset semantics, taking the log-sum-exp of the non-aligned classes against the
aligned one, and that reduction is meaningless for a head trained on a binary
target. Scoring the two arms on one axis needs a reduction that is fair to both,
and inventing one after seeing the losses is how a comparison becomes a
rationalisation.

### F33. Perception Encoder beats DINOv3 on unseen generators, by half what was claimed

Result: `A-encoder-comparison`, with `C-df26-logo-pe` as the new headline. Three
frozen encoders, one probe, one split: the frame-rate-controlled DF26 corpus
under leave-one-generator-out. The differences are paired and resampled by
source video, because every arm scores the same real clips.

| Encoder | Macro ROC-AUC | Against DINOv3 |
| --- | ---: | --- |
| Perception Encoder, PE-Core-B/16 | **0.7527** | +0.0741 [+0.0121, +0.1317] |
| CLIP ViT-B/16 | 0.7454 | +0.0668 [+0.0049, +0.1289] |
| DINOv3 ViT-S/16 | 0.6786 | reference |

Perception Encoder leads DINOv3 on all seven held-out generators. The largest
gains are Wan 2.6 at +0.148 and HunyuanVideo at +0.090, the two generators the
frame-rate control hit hardest. PE against CLIP is not separated, -0.0073
[-0.0603, +0.0463], so PE ships on its point estimate and because it was the
encoder the recommendation already named.

| Held-out generator | PE ROC-AUC |
| --- | --- |
| Veo 3.1 | 0.8496 [0.7886, 0.9015] |
| Wan 2.6 | 0.8164 [0.7472, 0.8785] |
| Grok Imagine | 0.7905 [0.7158, 0.8552] |
| Kling 3.0 | 0.7821 [0.7073, 0.8447] |
| Wan 2.2 | 0.6996 [0.6340, 0.7603] |
| HunyuanVideo 1.5 | 0.6764 [0.6071, 0.7401] |
| LTX 2.3 distilled | 0.6542 [0.5866, 0.7166] |
| macro mean | 0.7527 |

Two earlier statements were wrong, in opposite directions.

- The unverified claim, 72.7 for PE against 57.0 for DINOv3, had the right
  order and about twice the gap. Its DINOv3 figure was eleven points below this
  pipeline's own, which is why it was withdrawn.
- The mechanism argument for keeping DINOv3 was that a vision-language
  contrastive objective pulls features toward semantics and away from the
  texture detail forensics reads. Both vision-language encoders beat the
  self-supervised one here. On full frames of generated video, the semantic
  features carry more of the signal than that argument allowed.

The commercial generators still outscore the open-source ones, now 0.81 against
0.68. Under PE that gap widened, so the explanation the handoff records, that
the four commercial clips came through one platform and resemble each other,
remains untested and matters more.

One seed and one split. The probe is linear and deterministic given the split,
so the remaining variance is which source videos land in the test set.

### F34. An unseen manipulation family costs 0.12, and one family is at chance

Result: `C-ffpp-family-holdout`. The visual stream, face crops through a
fine-tuned EfficientNet, retrained six times on FaceForensics++ c23 with one
family held out each time. Each arm is scored on its held-out family's test
clips and the shared real test clips. The seen column is the all-family model
on the same clips.

| Held-out family | Seen | Unseen | Drop |
| --- | ---: | ---: | ---: |
| FaceSwap | 0.8110 | 0.5195 [0.4718, 0.5663] | 0.2915 |
| FaceShifter | 0.8531 | 0.7077 [0.6680, 0.7497] | 0.1454 |
| Deepfakes | 0.8912 | 0.7771 [0.7365, 0.8216] | 0.1140 |
| Face2Face | 0.8600 | 0.7839 [0.7505, 0.8237] | 0.0761 |
| NeuralTextures | 0.8486 | 0.7863 [0.7534, 0.8231] | 0.0623 |
| DeepFakeDetection | 0.9817 | 0.9322 [0.8816, 0.9739] | 0.0495 |
| macro mean | 0.8743 | 0.7511 | 0.1231 |

Four drops are separated: the unseen interval sits entirely below the seen one.
NeuralTextures and DeepFakeDetection overlap. The comparison is unpaired because
`score_by_family.py` keeps no per-clip scores, so two overlapping intervals do
not show the drop is zero, only that this test cannot see it.

FaceSwap is the outlier and the only family at chance. It is also the only
graphics-based identity swap in the set: Deepfakes and FaceShifter swap
identity with learned models, and Face2Face and NeuralTextures keep the
identity and change the expression. A detector that has seen learned swaps does
not recognise a graphics-based one. That explanation fits the table but was not
tested separately.

DeepFakeDetection barely moves, and the likely reason is capture rather than
manipulation. Its fakes come from paid actors filmed for the dataset, while
every real test clip is a FaceForensics++ YouTube original. The model can
separate the two on footage alone, whichever fakes it trained on.

This closes the Phase 5 leave-one-family-out item at its feasible scope: the
visual stream on FaceForensics++. FaceForensics++ has no audio, so the
three-branch version of this test cannot exist on it. The earlier three-arm
figure, 0.1119, moved to 0.1231 with the three missing arms, and FaceSwap is
the reason.

One seed per arm. F21 measured 0.0733 of seed spread for this architecture
cross-corpus. That is larger than two of the six drops, though this test is
closer to in-domain, where the seed spread was not measured.

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
