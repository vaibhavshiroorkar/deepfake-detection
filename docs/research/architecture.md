# Synthetic-media detection: the system to build now, and the one after it

Two designs, for two different jobs, and the first one is the priority.

**The demo detector** has to survive people generating fakes in front of it with
whatever tool they have and testing it live. It is narrow, it is trained on the
distribution it will meet, and it says so. That is section 0 and it is what to
build.

**The open-world design** is the research system: generator-agnostic, no
assumption of cross-modal inconsistency, built to survive unseen future
generators. It is everything from section 1 onward. It is the right long-term
shape and it is not what wins a live demo next month.

Every number below resolves to a row in
[result traceability](result-traceability.md) and is reproducible from `runs/`.

## 0. The demo detector

### What it must do

Someone generates a clip with Veo, Sora, Higgsfield or a phone app, uploads it,
and the system answers. Real comparison clips are phone videos shot by the
audience. The measured state of the current system against that job is bad:
0 of 10 on the one modern generator family measured, 0.4920 on the only
consumer-capture corpus available, and a learned prior that reads smoothness as
authenticity, which is backwards for generated video.

### Three layers, in order

**Provenance first.** Gemini and Veo embed SynthID; several pipelines carry
C2PA Content Credentials. When present this is exact, instant and needs no
training, and it is the most reliable thing in the system. It is defeated by
re-encoding and screen recording, so it is a first pass and never the answer.

**A specialist trained on the demo distribution.** Frozen DINOv3 or CLIP on full
frames, three windows of 32 contiguous frames, with a linear probe on top and a
cheap SRM residual branch beside it. A frozen encoder with a light probe is the
configuration that generalizes best to unseen generators, and it trains in
hours. Add the WavLM plus AASIST head when audio is present, since these tools
generate speech too.

**An abstention gate.** When the input is outside what the specialist was
trained on, it says so instead of guessing. In a live demo "I do not recognise
this, I will not guess" is defensible. A confident miss is not.

### Why a specialist rather than the general system

The binding constraint is capture conditions, not manipulation type. Three
independent measurements say so: an unseen manipulation family costs 0.11 and
still works while an unseen corpus reads 0.49; training on FaceForensics++
instead of FakeAVCeleb moved DFDC from 0.7611 to 0.4920; and a lip-sync stream
moved from 0.9969 to 0.7407 across two corpora that share VoxCeleb2 footage,
then to 0.5059 on the one that does not. Matching the training distribution to
the demo distribution is therefore the only intervention that reliably works,
and it is the one that fits the timeline.

### The data, which is the whole project now

- 200 to 500 clips per tool, from the tools the audience will actually use.
- The same number of real clips, recorded the way the audience records: phones,
  webcams, varied lighting, varied framing.
- **Both sides through identical post-processing.** Same container, codec,
  resolution, frame rate. `scripts/normalize_media.py` does this and it is not
  optional: the current system already learned post-processing instead of
  manipulation, detecting re-encoded in-the-wild clips at 86 percent and raw
  generator output at 23 percent, intervals that do not overlap.
- One tool held out entirely, never used in any decision. That number is the
  honest answer to "what about a tool you have not seen", and it should be known
  before anyone asks.

### The only metric that counts

Leave-one-generator-out, and nothing else is a headline.

Every reported number comes from a model trained on every generator except the
one it is scored on. There is no in-domain column, because detecting a generator
you trained on is not the problem and a strong in-domain number has repeatedly
been the sign of a shortcut rather than of skill. This project has three
measurements of that: 0.9990 in-domain against 0.4920 cross-corpus, a 0.9969
matched-pair stream that reads 0.5059 on a corpus with different capture
conditions, and a 0.9997 on Veo 3 that was pure content confound.

Consequences, all of which follow mechanically:

- With seven DF26 generators, the deliverable is seven rows plus a macro mean,
  each with an interval, not one aggregate.
- Model selection uses the held-out generator, never validation accuracy.
- Any change that buys in-domain accuracy and costs cross-generator transfer is
  refused. The BatchNorm measurement here priced exactly that trade at 0.5430
  in-domain for 0.1276 cross-corpus, and under this protocol the answer is no
  longer a judgement call.
- A frozen encoder with a light probe is preferred over fine-tuning, because
  fine-tuning is what converts a general representation into a generator-specific
  one.
- One generator is held back from every development decision, not just from the
  final training run, so the last row is honest rather than tuned against.

Realistic expectation, from the closest published run of this protocol: macro
mean around 0.82 across unseen generators, with per-generator values spanning
roughly 0.66 to 0.97. A single number near 0.99 under this protocol would be
evidence of a leak, not of success.

### What to claim on stage

Name the tools it covers. Per-generator collapse is the norm: a 2026 evaluation
found one detector at 69.8 percent on one generator and 15.4 percent on another.
A demo that says "these five tools, and anything else gets flagged as
unrecognised" survives contact. A demo claiming to catch everything is broken by
the first person with an unusual tool.

## 1. The open-world design

### Context

## Context

The brief changed the problem. Detecting a face swap in a real recording and
detecting a fully generated Veo or Sora clip are not the same task, and the
second one breaks the assumption the first was built on. A jointly generated
clip has consistent lips, consistent affect, consistent room tone and consistent
motion, because one model produced all of it. Any architecture whose Level 1
answer depends on cross-modal disagreement will call that clip authentic.

So the central axis moves. **Level 1 must ask whether the signal carries the
statistics of a physical capture chain and a real acoustic recording. Cross-modal
consistency drops to Level 2, where it still separates a face swap from a fully
generated scene.** Everything below follows from that.

This repository has measured several of the proposed components on the
face-manipulation half of the problem. Those measurements are the reason some of
the recommendations below contradict the proposal. Each carries an
identity-clustered 95 percent interval, each is registered in
`docs/research/result-traceability.md`, and each is single-seed and therefore
`provisional`.

### Measured evidence from this repository

| Design assumption | Measurement here |
| --- | --- |
| Cross-modal inconsistency is the core signal | Both cross-modal streams sit at chance `diagonal_mass` in every epoch of three runs. On LAV-DF, whose matched windows come from the same file so correspondence is the only cue left, the lip-sync stream reached 0.9969 with attention still at chance. The mechanism was never operating, even where the corpus forbade every alternative. |
| DINOv3 as primary visual backbone | Frozen DINOv3 ViT-S read 0.9199 [0.8882, 0.9446] in-domain against fine-tuned EfficientNet-B0 at 0.9719 [0.9692, 0.9741], and the two are indistinguishable cross-corpus. A frontier representation is not automatically the better forensic feature. |
| A separate emotion branch adds information | It is the best single stream here, 0.5951 [0.5558, 0.6351] on DFDC, and also the most redundant: +0.681 correlation and 41.7 percent shared errors with the visual branch, which reads the same crop. |
| Richer fusion beats simple fusion | Calibrated late fusion read 0.5687 [0.5279, 0.6052] cross-corpus against a cross-attention head's 0.5411 [0.4984, 0.5828]. In-domain they tie. |
| More branches help | The five-stream fusion scored 0.5411, below three of its own inputs. The best subset had two. |
| More training data fixes generalization | Training on FF++ instead of FakeAVCeleb moved DFDC from 0.7611 to 0.4920 [0.4478, 0.5350], an interval spanning chance. |
| The temporal prior is sound | It is inverted. Motion correlates -0.316 with the fake score among manipulated clips, so the model treats smoothness as authenticity. Generated video is smooth. |
| Confidence is usable off-corpus | Expected calibration error rises from 0.0022 to 0.6366 cross-corpus. |
| Compression is a nuisance, not a feature | On MNW the detector finds re-encoded in-the-wild clips at 86 percent [49, 97] and raw generator output at 23 percent [15, 34], non-overlapping. It is keying on post-processing. |

That last row is the one to take most seriously for open-world work, and it
matches the literature: CNN detectors that generalize across generators collapse
under JPEG recompression.

## Critique of the proposed stack

### Branches that earn their place

- **Generator-agnostic synthetic-media forensic branch.** The most important
  branch in the new design and the only one that answers Level 1 when no face
  and no speech are present. Promote it from an addition to the backbone of the
  system.
- **Camera and physical acquisition consistency.** Right idea, with a caveat
  large enough to change its design. See the laundering tension below.
- **Speech forensics with an anti-spoofing head.** Still the least redundant
  audio signal. Measured overlap with the visual branch here is +0.049 with 37.4
  percent shared errors.
- **Temporal branch.** Necessary, and currently the biggest hole, since the
  existing system has an inverted temporal prior.
- **Provenance.** Cheap and exact when present. Correctly scoped as supplementary.

### Redundant

| Pair | Verdict |
| --- | --- |
| emotion2vec beside WavLM | Redundant. emotion2vec is a speech SSL model; its affect embedding is a projection of what WavLM already encodes. One trunk, two heads. |
| EfficientNet emotion beside a face-crop visual branch | Measured +0.681 correlation, 41.7 percent shared errors. Two models on one crop. |
| AV-HuBERT beside a separate speech encoder | AV-HuBERT contains an audio tower. The waveform is encoded twice. |
| DINOv3 beside V-JEPA 2 for appearance | Overlapping on appearance. V-JEPA 2 earns its place only for motion. |
| A forensic encoder beside both | Less redundant than it looks, and this is the important asymmetry: DINOv3 and V-JEPA 2 are trained to be invariant to the low-level detail a forensic encoder exists to read. They are complementary by construction. |

### Missing

- **A model of authentic capture, not just a classifier of known fakes.** A
  binary discriminator trained on today's generators answers "is this like the
  generators I saw". Open-world needs "is this unlike a real camera". These are
  different learning problems and the second one does not need fake examples.
- **A laundering estimator.** Several forensic traces survive only in unlaundered
  media. The system must know which regime it is in before it trusts a branch.
- **Level 2 requires generator-family labels** in training data, which none of
  the public corpora fully provide. Plan the label schema before collecting.
- **No-face and no-speech paths.** Constraints 11 and 12 mean the face pipeline
  cannot be on the critical path at all.

### Remove or replace

| Component | Verdict |
| --- | --- |
| SyncNet | Remove. Offset classification is the wrong objective, and a jointly generated clip is perfectly synced. |
| emotion2vec as a branch | Replace with a head on the speech trunk. |
| EfficientNet emotion as a branch | Replace with an auxiliary head on the shared face trunk, feeding Level 2 only. |
| Wav2Vec2 | Replace with WavLM-Large or XLS-R plus an AASIST-style graph head, using a learned weighted sum of early-to-mid layers rather than the final layer, which carries the soft artifacts better. |
| DINOv3 as primary | Demote to a frozen second view. Keep it: frozen frontier encoders with light probes remain the strongest generalizers to unseen generators. Do not fine-tune it, which destroys that property. |
| V-JEPA 2 as a baseline component | Treat as a research bet. A JEPA objective predicts representations and is trained to discard pixel detail, which is where forensic evidence lives. V-JEPA 2.1 specifically targets dense features and is the variant to try. No established literature applies it to this task. |

### Shortcut and leakage risks, ranked by measured likelihood

1. **Post-processing and compression.** Measured here, and the dominant risk. Any
   corpus where fakes and reals went through different encoders teaches the
   encoder, not the manipulation.
2. **Semantic content.** Generated clips are prompted, so their subject matter
   differs systematically from real corpora. A detector that learns "video of an
   astronaut riding a horse" is not a detector. Content-matched pairs are the fix.
3. **Watermarks and generator borders.** A de-watermarked Sora benchmark exists
   precisely because detectors learn the mark. Strip or verify before training.
4. **Capture pipeline as dataset identity.** FF++-trained reads 0.8253 on
   FakeAVCeleb and 0.4920 on DFDC.
5. **Frozen normalisation statistics.** Measured at 0.1276 of cross-corpus AUC.
6. **Identity.** Tested clean here, moved the headline by 0.004. Keep as a check.

## The laundering tension, stated plainly

The camera-pipeline branch is the most theoretically sound idea in the brief and
the most fragile to exactly the transformations the brief requires robustness to.

| Trace | Survives resize | Survives re-encode | Survives screen record |
| --- | --- | --- | --- |
| CFA and demosaicing periodicity | no | weak | no |
| PRNU sensor noise | no | weak | no |
| Lens vignetting, chromatic aberration | partly | yes | partly |
| Depth-of-field consistent with scene depth | yes | yes | yes |
| Motion blur consistent with exposure and motion | yes | yes | mostly |
| Rolling shutter geometry | partly | yes | no |

Design consequence: split it. The **sensor-level** half is a high-precision,
low-recall branch that the reliability layer masks out unless the laundering
estimator says the media is plausibly original. The **optical and physical** half
survives laundering and belongs in the always-on path. Do not build one branch
that mixes them, because its reliability is then unestimable.

## Which forensic traces disappear as generators improve

| Tier | Traces | Outlook |
| --- | --- | --- |
| Disappearing now | Visible watermarks, GAN checkerboard fingerprints, fixed-upsampler frequency peaks, temporal flicker, hand and text failures | These are defects. Every release fixes some. Do not build a system whose Level 1 rests here. |
| Durable for a while | Diffusion decoder periodicity, patch-level statistics, absence of sensor noise, over-smooth high-frequency residuals | Tied to architecture families rather than to bugs, so they move with the family, not the version. Useful with family-level retraining. |
| Most durable | Physical consistency: depth of field against scene depth, motion blur against motion magnitude, lighting and shadow coherence, object permanence, inter-object physics | A generator must model the world to get these right. They are also the most expensive to measure and the hardest to learn from limited data. |
| Structurally durable | The *absence* of a capture chain | Hardest for a generator to fake deliberately, because it requires simulating an entire imaging pipeline it never had. This is the strongest basis for open-world detection. |

The strategic implication: model the real class. A generator can remove any
artifact it knows about, but it cannot easily manufacture the full joint
statistics of optics, sensor, ISP and codec that a genuine capture carries.

## Three architectures

Compute is relative to architecture B at 1.0, forward FLOPs per 10-second clip.

### A. Research maximum, roughly 5x

| Branch | Backbone | Features | Dim | Frozen? |
| --- | --- | --- | --- | --- |
| Forensic residual | SRM high-pass plus learned residual CNN, full frame, no resize | final pooled plus patch-statistic head | 512 | trained |
| Frozen semantic probe | DINOv3 ViT-B/16 and a CLIP-family ViT, full frame | blocks 6 and 11 concatenated | 1536 | frozen, probe only |
| Temporal | V-JEPA 2.1 ViT-L | intermediate blocks | 1024 | frozen |
| Physical consistency | depth, optical flow and blur estimators into a small transformer | derived scalars plus 256-d embedding | 256 | trained heads on frozen estimators |
| Sensor forensics | CFA periodicity, PRNU correlation, ISP statistics | handcrafted plus small CNN | 128 | trained, reliability-masked |
| Face artifact | ConvNeXt-V2-B on face crops | penultimate | 1024 | fine-tuned, live norm stats |
| Affect, Level 2 only | AU intensities plus valence and arousal head on the face trunk | - | 128 | trained head |
| AV correspondence, Level 2 only | AV-HuBERT-Large mouth ROI plus biomechanical constraint features | layer 9 | 1024 | frozen, then top 4 blocks |
| Speech forensics | WavLM-Large, learned weighted sum of layers 1 to 8 | AASIST graph head | 768 | frozen, then top 4 blocks |
| Authenticity density | normalizing flow or energy model over the frozen-probe and residual features, trained on real media only | log-likelihood | 1 | trained on real only |

Temporal modelling: 4-layer transformer over per-window embeddings from three
windows of 32 contiguous frames. Fusion: reliability-weighted late fusion over
calibrated per-branch scores, with a gated cross-attention challenger reported
only if it wins a paired cross-corpus bootstrap. Heads: Level 1 binary, Level 2
family softmax over the eight classes, an OOD score from the density model, and
one calibrated score per branch.

### B. Balanced, 1.0x, recommended default

| Branch | Backbone | Features | Dim |
| --- | --- | --- | --- |
| Forensic residual | SRM plus 6-block CNN, full frame at native resolution | pooled | 256 |
| Frozen probe | DINOv3 ViT-S or CLIP ViT-B, full frame | penultimate block | 768 |
| Temporal | 4-layer transformer over per-frame probe and residual embeddings | - | 512 |
| Physical consistency | monocular depth plus flow, blur-vs-motion and depth-vs-aperture residuals | 12 scalars plus 128-d | 128 |
| Face artifact, when a face is present | EfficientNetV2-S or ConvNeXt-T | penultimate | 512 |
| Affect head, Level 2 | on the face trunk | - | 128 |
| Speech | WavLM-Base+, layers 1 to 6 | AASIST-lite | 512 |
| Correspondence, Level 2 | small mouth-audio net, self-supervised offset objective on real clips only | - | 256 |
| Authenticity density | Gaussian mixture or flow over frozen-probe features, real media only | 1 | |

Fusion: reliability-aware calibrated late fusion. Everything face-related and
audio-related is optional and masked when absent, so a silent, person-free
generated clip still gets a Level 1 answer from the residual, probe, temporal
and physical branches alone.

### C. Production, 0.2x

Frozen CLIP or DINOv3-S probe plus the SRM residual CNN on full frames, a GRU
over 32 contiguous frames, WavLM-Base+ with an AASIST-lite head when audio
exists, the authenticity density model for the OOD score, provenance check
first, reliability-aware late fusion, per-media-kind thresholds. Face and
correspondence branches omitted entirely; they buy Level 2 detail, not Level 1
accuracy.

## Branch-specific answers

### Synthetic video, the new core

Current practice for unseen generators is a frozen frontier encoder with a light
probe, which still beats detectors trained end to end, and 2026 work continues to
confirm it. Its known bottleneck is real: a probe over frozen features can only
push a sample toward "real" or "generated" and cannot learn an artifact the
encoder does not already represent. So pair it with a trained residual branch
that can, and accept that the residual branch is the part that ages.

Generator fingerprints versus generator-agnostic representations is not a choice,
it is a split by purpose. Fingerprints are excellent for Level 2 attribution and
worthless for Level 1 open-world. Build them as a separate classifier over known
families, not as the primary evidence.

Two published data points worth designing against. On a fourteen-generator
image-to-video benchmark, image-level detectors transferred to video averaged
higher AUC than the best dedicated video detector, and Sora remained the hardest
at roughly 0.73. And a 2026 evaluation found EfficientNet-B7 detecting Sora-2 at
69.8 percent while collapsing to 15.4 percent on Kling v2. Per-generator collapse
is the norm, so a single aggregate number will mislead you. Report per generator
with intervals, as this project now does for MNW.

### Lip-sync, including the jointly generated case

| Option | Verdict |
| --- | --- |
| SyncNet | Remove. |
| AV-HuBERT features | Strong, well supported, and its visual tower sees only the lip region, so it loses evidence outside the mouth and underperforms on face swaps. |
| LipFD-style specialists | Strong in-domain, language-dependent, degrading to roughly 72 percent on Chinese content. |
| Biomechanical constraint violation | Best for unseen attacks, language-agnostic by construction, reported 0.843 AUC zero-shot on a seven-language set with an unseen generator. |

For jointly generated media, all four are expected to fail, and that is not a
defect to engineer around. A perfectly synchronized clip is evidence *against*
the "lip-sync manipulation" class and says nothing about Level 1. Use the branch
as a Level 2 discriminator and let the reliability layer down-weight it when the
correspondence score is high and the forensic branches disagree. That
configuration, high sync quality with anomalous capture statistics, is itself the
signature of joint generation and deserves its own Level 2 class.

### Audio

WavLM-Large is the best general choice, XLS-R where languages vary. HuBERT adds
nothing over WavLM here. Use early-to-mid layers. Pair with an AASIST-style graph
head, the configuration behind the strongest ASVspoof 5 results.

emotion2vec as a second encoder is not justified; attach an affect head to the
WavLM trunk. Fully generated audio differs from voice cloning in one way that
matters: it includes room tone, sound effects and music, so the anti-spoofing
head should be trained on full mixtures rather than clean speech alone, and a
"no speech present" path must still produce an audio Level 1 score.

### Video encoders and ROIs

DINOv3 gives semantic appearance, V-JEPA 2 gives motion, and both suppress the
pixel detail forensics needs, which is exactly why the residual branch is not
optional. DINOv3 should run on **full frames**, not face crops. Face and mouth
crops feed the Level 2 branches only. A face-only pipeline cannot satisfy
constraint 11.

Sampling: three windows of 32 contiguous frames, scored independently and
aggregated, never 16 frames spread across a clip. That spread sampling is what
produced the inverted motion prior measured here. Aggregate by mean of logits
plus max, and keep per-window scores as crude localization.

### Fusion

| Mechanism | Assessment |
| --- | --- |
| Concatenation | Cannot express absence; highest-variance branch dominates. |
| Calibrated late fusion | Measured best cross-corpus here and trivially handles missing branches. |
| Cross-attention or multimodal transformer | Lost to late fusion on this data. Keep as challenger. |
| Mixture of experts | Promising for per-generator routing, needs family labels, and routing is itself a shortcut risk. |
| Reliability-aware fusion | The right frame for this problem, and the piece the earlier design lacked. |

Recommended: each branch emits a calibrated score and a reliability estimate.
Reliability is computed, not learned end to end: face branch reliability from
face coverage and track stability, audio from speech presence and SNR, sensor
forensics from the laundering estimator, correspondence from mouth visibility.
Fuse as a reliability-weighted linear model over calibrated scores, mask absent
branches after projection so they contribute exactly zero including bias, and cap
any single branch's weight so one cannot dominate. Sample presence and
reliability patterns during fusion training at deployment rates.

## Emotion branch verdict

EfficientNet-B2/B3 with attention and temporal aggregation is a reasonable
engineering choice, and it is answering a question that no longer sits on the
critical path. Measured redundancy with a visual branch on the same crops is
+0.681 with 41.7 percent shared errors. A POSTER-family model would improve FER
accuracy, and there is no evidence that FER accuracy is what the forensic signal
needs.

Represent affect as **AU intensities plus valence and arousal**, not categorical
labels. AUs are geometric and generator failures show up in specific ones, which
also yields a human-readable reason.

Against jointly generated content the branch is close to useless for Level 1, by
construction, and this project's measurement is the stronger warning: the stream
built to exploit facial-vocal inconsistency is the best stream here and provably
does not use the mechanism. Implement the inconsistency as a recorded quantity,
the correlation between facial and vocal affect trajectories, keep it out of the
loss, and let it earn Level 2 weight if it ever separates anything.

## Training and evaluation recipe

1. **Gates first, untrained.** Provenance, then the laundering estimator, then
   the OOD score from the authenticity density model.
2. **Authenticity density on real media only.** No fake examples. This is the
   open-world component and it must never see a generator.
3. **Per-branch training, backbones frozen.** Heads only. Select on ROC-AUC, not
   validation loss: two corpora here show them diverging, and on one the
   loss-selected checkpoint scored below chance.
4. **Partial unfreeze** of the trained branches only. Never unfreeze the frozen
   probe: fine-tuning destroys the property it was chosen for. Prefer LayerNorm
   backbones; where BatchNorm is unavoidable keep statistics live.
5. **Calibrate every branch** on held-out data, then fit reliability-aware fusion
   on out-of-fold scores with presence and reliability sampling.
6. **Level 2 head last**, conditioned on Level 1, trained only on the subset with
   family labels.

Augmentation, and this is the highest-value intervention given the measured
compression shortcut: sweep H.264, H.265 and AV1 at several CRFs, resize, crop,
frame-rate convert, color grade, sharpen, denoise, add noise, simulate a
social-media re-upload chain and a screen recording. Apply identically to real
and synthetic. Then verify: train a classifier to predict the augmentation from
the model's features, and require it to fail. Forensic-oriented augmentation that
deliberately suppresses semantic content is worth adopting, since semantic
shortcut learning is the second-ranked risk here.

Hard negatives: real clips pushed through the exact pipeline the synthetic ones
went through, and synthetic clips with matched prompts to real content so the
subject matter cannot separate the classes.

Leakage prevention: group splits on both identities in a swap; hold out whole
generator families, not individual checkpoints; never let two versions of one
generator straddle the split; strip or verify watermarks before training.

Balancing: class weights rather than discarding real clips. Report per family.

## Evaluation

Every row with an identity-clustered or generator-clustered interval, and per
generator rather than aggregate:

- within dataset, and identity-strict within dataset
- cross dataset, at least two targets
- leave one generator out, and leave one generator family out
- a held-back "future-like" generator never used in any development decision
- videos with no people, and videos with no audio
- jointly generated audiovisual media as its own partition
- per codec and per CRF curves, plus the full laundering chain
- each modality dropped, coverage reported beside accuracy
- adversarial post-processing: crop, rescale, re-encode, frame drop, screen record
- calibration per partition: Brier and ECE
- mechanism checks recorded and never optimised: attention diagonal mass,
  affect-trajectory correlation, motion against score, augmentation predictability

## Final architecture

```text
input
 ├── provenance: C2PA, Content Credentials, known watermarks      (exact when present)
 ├── laundering estimator: codec, resample, screen-record traces   (gates sensor branch)
 └── authenticity density model over frozen features, real-only    (OOD / open-world score)

ALWAYS ON, no face or speech required
 ├── forensic residual CNN, full frame, native resolution      256
 ├── frozen probe: DINOv3 / CLIP ViT, full frame               768   (never fine-tuned)
 ├── physical consistency: depth, flow, blur, lighting         128
 ├── sensor forensics: CFA, PRNU, ISP                          128   (masked if laundered)
 └── temporal transformer over 3 windows x 32 contiguous frames

WHEN A FACE IS PRESENT                    WHEN AUDIO IS PRESENT
 ├── face artifact trunk           512     ├── WavLM layers 1-6 + AASIST     512
 ├── affect head: AU + valence/arousal 128 └── affect head                   128
 └── AV correspondence (mouth ROI) 256

              ▼ per-branch calibrated score + reliability estimate
     reliability-weighted late fusion, absent branches masked to exactly zero
              ▼
 Level 1: authentic vs synthetic        Level 2: family, conditioned on Level 1
 OOD score · per-branch evidence · per-window localization · calibrated confidence
```

## Roadmap

**Phase 1, minimum viable strong baseline.** Frozen probe plus forensic residual
on full frames, contiguous-window sampling, temporal transformer, WavLM with
AASIST when audio is present, calibrated late fusion, provenance and OOD gates,
and the evaluation harness with leave-one-generator-family-out from day one.
Reuse from this repository: `training/ranking.py` for AUC-based selection,
`evaluation/bootstrap.py` for clustered intervals, `fusion/deep.py` for presence
masking, `inference/distribution_gate.py` for the OOD pattern, `views/` for
content-addressed caching under a `preprocessing_config_hash`, and
`scripts/update_result_registry.py` so every number stays traceable. No face
branch in Phase 1: it is not on the Level 1 critical path.

**Phase 2, highest-value upgrades.** The authenticity density model trained on
real media only, which is the actual open-world mechanism. Physical-consistency
branch. Full laundering-chain augmentation with the augmentation-predictability
check. Reliability estimation per branch. Level 2 family head.

**Phase 3, research extensions.** Sensor forensics behind the laundering gate.
V-JEPA 2.1 dense features as a frozen temporal view. Biomechanical lip-sync
constraints. Mixture-of-experts routing over families. Each admitted only on a
paired bootstrap against Phase 2 on a held-out generator family.

## Verification

- Every branch ablatable alone, with a registry row and an interval.
- No improvement claimed without a paired bootstrap on a held-out generator family.
- A person-free, silent generated clip produces a Level 1 answer and a reason.
- A jointly generated clip with perfect lip sync is not called authentic.
- An augmentation classifier trained on the model's features fails.
- Mechanism metrics recorded per epoch and absent from every loss.
