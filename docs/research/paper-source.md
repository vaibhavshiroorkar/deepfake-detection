# Paper source book

Everything the paper needs, in one place. Write the paper from this file.

Two kinds of content live here and they are kept apart on purpose. The prose is
written by hand and holds the judgement: what a result means, why a decision was
taken, what is still unknown. The tables are generated from the artifacts under
`runs/` and hold the numbers. Do not type a number into the prose that is not in
a generated table above it.

Refresh the tables with:

```powershell
uv run python scripts/update_paper_source.py
```

Run it after any scoring or training run. It reads the artifacts, so a table
that says "not generated yet" means that experiment has not been run, and it
names the command that would run it. That is the honest state, not a gap to fill
in by hand.

Companion documents: [architecture](architecture.md) is the design for the next
system, built against these measurements, [corpora](datasets.md) surveys the
datasets this project could use next, [findings](findings.md) states what the measurements support, [result traceability](result-traceability.md) resolves every number to
a file hash and an MLflow run, [the draft](paper.md) is the paper itself, and
[obstacles](../obstacles.md) is the full failure log.

## 1. The claim

The research question, fixed before the experiments:

> Under source-disjoint and shortcut-controlled evaluation, does cue-specific
> fusion of visual artifacts, audio spoofing, and mouth-audio alignment
> generalize better than a strong visual baseline?

The answer on this data is no. The exit criteria anticipated that and required
the negative result to be reported with its confidence interval, which is what
section 6 does.

The contribution is therefore not the architecture and not the accuracy. It is
the set of validity checks that caught three things an accuracy table cannot
see:

1. Two streams named for audiovisual correspondence never measured it, and one
   of them is the most accurate model in the project.
2. A regulariser that improved validation destroyed cross-corpus transfer.
3. A checkpoint selection rule preferred a model scoring below chance.

Each was caught by a quantity recorded beside the training objective and never
optimised against it. That practice is what the paper argues transfers.

## 2. What was built

Two systems, scored through one evaluation path.

**Design A**, the baseline pipeline. Three cue-specific branches: EfficientNet-B0
with a GRU over 16 sampled frames for visual artifacts, Wav2Vec2 Base with
attentive pooling for audio spoofing, and a framewise ResNet-18 mouth encoder
with a separate Wav2Vec2 for mouth-audio alignment. Fusion is regularized
logistic regression over Platt-calibrated branch logits plus three quality
features, fitted only on out-of-fold rows from source-grouped cross-fitting.

**Design B**, the streams. Visual streams take a backbone and a temporal model as
configuration. Audiovisual streams are cross-attention over temporal tokens.
`StreamFusion` projects each stream's embedding to a common width and fuses in
feature space, masking absent streams after the projection so an absent stream
contributes exactly zero including its bias.

Both are scored by the same code. That mattered once already: Design A's visual
branch reads 0.7611 on DFDC and Design B's reads 0.5067 on the same clips, and
the gap was first suspected to be an evaluation artifact. Scoring the Design A
checkpoint through the Design B path returned 0.7611, matching to four decimals,
which established the regression as real before anything was concluded from it.

### What each stream actually is

Read from the history file each trainer wrote, so this is the model that exists
rather than the model that was intended.

<!-- BEGIN GENERATED TRAINING -->
| Stream | Model | Optimizer | Schedule | Best epoch | Validation AUC | Diagonal mass |
|---|---|---|---|---:|---:|---:|
| `stream-emotion` | attention_heads=4, audio_model=facebook/wav2vec2-base, common_dim=256, video_backbone=tf_efficientnet_b0.ns_jft_in1k | adamw, head 0.0001, encoder 5e-06 | 9 of 10 epochs, freeze 2, patience 3 | 6 | 0.9991 | 0.3439 |
| `stream-lipsync` | attention_heads=4, audio_model=facebook/wav2vec2-base, common_dim=256, video_backbone=tf_efficientnet_b0.ns_jft_in1k | adamw, head 0.0001, encoder 5e-06 | 4 of 10 epochs, freeze 2, patience 3 | 1 | 0.7593 | 0.0592 |
| `stream-lipsync-lowlr` | attention_heads=4, audio_model=facebook/wav2vec2-base, common_dim=256, video_backbone=tf_efficientnet_b0.ns_jft_in1k | adamw, head 0.0001, encoder 1e-06 | 6 of 10 epochs, freeze 3, patience 5 | 1 | 0.7593 | 0.0592 |
| `visual-dinov3` | backbone=vit_small_patch16_dinov3.lvd1689m, common_dim=256, frozen_backbone=True, global_pool=token, temporal=lstm, temporal_hidden=256 | adamw, head 0.001, encoder 5e-06 | 8 of 10 epochs, freeze 10, patience 3 | 5 | 0.9559 | not applicable |
| `visual-efficientnet` | backbone=tf_efficientnet_b0.ns_jft_in1k, common_dim=256, frozen_backbone=False, global_pool=avg, temporal=lstm, temporal_hidden=256 | adamw, head 0.0001, encoder 5e-06 | 10 of 10 epochs, freeze 2, patience 3 | 10 | 0.9997 | not applicable |
<!-- END GENERATED TRAINING -->

## 3. Data and protocol

FakeAVCeleb is the development set, split 70/15/15 by source identity with a
frozen split hash. DFDC is the cross-corpus test set and the only audio-bearing
corpus here not built on VoxCeleb2. Celeb-DF-v2 and the evaluation-only MNW
benchmark are external checks. FaceForensics++ is absent, so no number in this
project is comparable to a published table.

Metrics are ROC-AUC with 95 percent bootstrap confidence intervals clustered on
source identity. Clustering is on the speaker, not the clip, because the
question is what happens on a different set of speakers. On DFDC the
identity-clustered interval came out narrower than a clip-level one, 0.0776
against 0.1065, since identities there carry balanced class proportions.
Narrower is not the reason to choose it.

Abstention rather than imputation: a clip with no stable face track produces no
visual view, and is reported and counted rather than filled in with a full-frame
crop or dropped from an inner join. Coverage is reported beside every metric.

The view cache is content-addressed under a `preprocessing_config_hash` covering
both the view settings and the code version, so a server whose preprocessing
differs from the cache a model was trained on is refused rather than silently
served.

## 4. Design A results

<!-- BEGIN GENERATED DESIGNA -->
| Report | Dataset | Rows | ROC-AUC | Balanced accuracy |
|---|---|---:|---:|---:|
| `audio-in-domain-test-metrics.json` | FakeAVCeleb | 1,648 | 1.0000 | 1.0000 |
| `fusion-dfdc-metrics.json` | DFDC | 2,351 | 0.7500 | 0.6203 |
| `fusion-test-metrics.json` | FakeAVCeleb | 1,648 | 0.9990 | 0.9990 |
| `visual-celebdf-metrics.json` | Celeb-DF-v2 | 462 | 0.6682 | 0.5435 |
| `visual-dfdc-metrics.json` | DFDC | 2,410 | 0.7611 | 0.5677 |
| `visual-in-domain-test-metrics.json` | FakeAVCeleb | 1,648 | 0.9988 | 0.9990 |
| `visual-mnw-metrics.json` | MNW | 85 | 0.0000 | 0.1429 |
<!-- END GENERATED DESIGNA -->

## 5. Design B results

### 5.1 Per-stream accuracy

<!-- BEGIN GENERATED STREAMS -->
| Stream | holdout | in-domain | DFDC |
|---|---|---|---|
| `final-audio-seed17` | 0.7919 [0.7717, 0.8137] | 0.7739 [0.7564, 0.7940] | 0.5070 [0.4771, 0.5359] |
| `stream-emotion` | 0.9991 [0.9981, 0.9999] | 0.9978 [0.9956, 0.9996] | 0.5951 [0.5558, 0.6351] |
| `stream-lipsync` | 0.7593 [0.7360, 0.7819] | 0.7587 [0.7345, 0.7825] | 0.4890 [0.4569, 0.5217] |
| `visual-dinov3` | 0.9216 [0.8912, 0.9457] | 0.9199 [0.8882, 0.9446] | 0.5147 [0.4795, 0.5516] |
| `visual-efficientnet` | 0.9733 [0.9712, 0.9750] | 0.9719 [0.9692, 0.9741] | 0.5067 [0.4710, 0.5458] |

Labels are `clip_fake`, which the fusion store carries. The visual stream optimises `video_fake`, so its in-domain figure differs there; see the BatchNorm table.
<!-- END GENERATED STREAMS -->

### 5.2 Does fusing streams beat the best single stream

Every subset of the available streams, each with its own head fitted on the same
holdout rows and scored on the same two partitions, so the only thing varying is
which streams are in the input.

<!-- BEGIN GENERATED ABLATION -->
| Streams | in-domain | dfdc |
|---|---|---|
| final-audio-seed17 | 0.7780 [0.7603, 0.7961] | 0.4933 [0.4652, 0.5227] |
| stream-emotion | 0.9978 [0.9952, 0.9996] | 0.5991 [0.5593, 0.6361] |
| stream-lipsync | 0.8249 [0.7995, 0.8480] | 0.5543 [0.5189, 0.5885] |
| visual-dinov3 | 0.9205 [0.8943, 0.9438] | 0.5112 [0.4754, 0.5454] |
| visual-efficientnet | 0.9753 [0.9722, 0.9779] | 0.5027 [0.4663, 0.5402] |
| final-audio-seed17 + stream-emotion | 0.9979 [0.9957, 0.9996] | 0.5999 [0.5611, 0.6384] |
| final-audio-seed17 + stream-lipsync | 0.8473 [0.8279, 0.8679] | 0.5676 [0.5324, 0.6034] |
| final-audio-seed17 + visual-dinov3 | 0.9662 [0.9501, 0.9801] | 0.5137 [0.4779, 0.5469] |
| final-audio-seed17 + visual-efficientnet | 0.9979 [0.9957, 0.9995] | 0.4915 [0.4545, 0.5335] |
| stream-emotion + stream-lipsync | 0.9979 [0.9959, 0.9996] | 0.6076 [0.5666, 0.6488] |
| stream-emotion + visual-dinov3 | 0.9979 [0.9958, 0.9995] | 0.5372 [0.4964, 0.5749] |
| stream-emotion + visual-efficientnet | 0.9988 [0.9974, 0.9999] | 0.5452 [0.5064, 0.5858] |
| stream-lipsync + visual-dinov3 | 0.9493 [0.9304, 0.9655] | 0.5182 [0.4833, 0.5527] |
| stream-lipsync + visual-efficientnet | 0.9952 [0.9912, 0.9983] | 0.5139 [0.4768, 0.5537] |
| visual-dinov3 + visual-efficientnet | 0.9769 [0.9744, 0.9793] | 0.5180 [0.4805, 0.5549] |
| final-audio-seed17 + stream-emotion + stream-lipsync | 0.9980 [0.9960, 0.9997] | 0.6003 [0.5585, 0.6409] |
| final-audio-seed17 + stream-emotion + visual-dinov3 | 0.9981 [0.9960, 0.9998] | 0.6017 [0.5611, 0.6408] |
| final-audio-seed17 + stream-emotion + visual-efficientnet | 0.9990 [0.9976, 1.0000] | 0.5450 [0.5066, 0.5866] |
| final-audio-seed17 + stream-lipsync + visual-dinov3 | 0.9625 [0.9463, 0.9762] | 0.5144 [0.4796, 0.5491] |
| final-audio-seed17 + stream-lipsync + visual-efficientnet | 0.9979 [0.9959, 0.9995] | 0.4984 [0.4610, 0.5386] |
| final-audio-seed17 + visual-dinov3 + visual-efficientnet | 0.9981 [0.9963, 0.9995] | 0.5138 [0.4749, 0.5550] |
| stream-emotion + stream-lipsync + visual-dinov3 | 0.9981 [0.9961, 0.9997] | 0.5924 [0.5491, 0.6348] |
| stream-emotion + stream-lipsync + visual-efficientnet | 0.9987 [0.9969, 0.9999] | 0.5416 [0.5032, 0.5826] |
| stream-emotion + visual-dinov3 + visual-efficientnet | 0.9988 [0.9975, 0.9999] | 0.5376 [0.4982, 0.5783] |
| stream-lipsync + visual-dinov3 + visual-efficientnet | 0.9963 [0.9937, 0.9987] | 0.5204 [0.4799, 0.5578] |
| final-audio-seed17 + stream-emotion + stream-lipsync + visual-dinov3 | 0.9981 [0.9962, 0.9996] | 0.5407 [0.4980, 0.5812] |
| final-audio-seed17 + stream-emotion + stream-lipsync + visual-efficientnet | 0.9988 [0.9972, 1.0000] | 0.5493 [0.5090, 0.5905] |
| final-audio-seed17 + stream-emotion + visual-dinov3 + visual-efficientnet | 0.9989 [0.9974, 1.0000] | 0.5415 [0.4995, 0.5827] |
| final-audio-seed17 + stream-lipsync + visual-dinov3 + visual-efficientnet | 0.9977 [0.9953, 0.9995] | 0.5250 [0.4844, 0.5647] |
| stream-emotion + stream-lipsync + visual-dinov3 + visual-efficientnet | 0.9987 [0.9970, 1.0000] | 0.5451 [0.5042, 0.5855] |
| final-audio-seed17 + stream-emotion + stream-lipsync + visual-dinov3 + visual-efficientnet | 0.9988 [0.9971, 1.0000] | 0.5411 [0.4984, 0.5828] |

- in-domain: best is final-audio-seed17 + stream-emotion + visual-efficientnet at 0.9990; best single is stream-emotion at 0.9978; all 5 streams reach 0.9988.
- dfdc: best is stream-emotion + stream-lipsync at 0.6076; best single is stream-emotion at 0.5991; all 5 streams reach 0.5411.
<!-- END GENERATED ABLATION -->

### 5.3 Redundancy between streams

Logit correlation and the fraction of clips both streams get wrong. The second
is the one that matters for fusion: two streams with similar accuracy that fail
on different clips are worth combining, and two that fail together are not.

<!-- BEGIN GENERATED CORRELATION -->
**dfdc**

| Pair | Logit correlation | Shared errors | Clips |
|---|---:|---:|---:|
| `stream-emotion` and `visual-efficientnet` | +0.455 | 56.4% | 2,410 |
| `visual-dinov3` and `visual-efficientnet` | +0.219 | 32.8% | 2,410 |
| `stream-lipsync` and `visual-dinov3` | +0.181 | 47.4% | 2,351 |
| `stream-lipsync` and `visual-efficientnet` | -0.160 | 26.2% | 2,351 |
| `final-audio-seed17` and `visual-dinov3` | +0.136 | 40.7% | 2,410 |
| `stream-emotion` and `stream-lipsync` | +0.131 | 32.8% | 2,351 |
| `final-audio-seed17` and `stream-lipsync` | +0.109 | 37.7% | 2,351 |
| `final-audio-seed17` and `stream-emotion` | +0.061 | 34.5% | 2,410 |
| `stream-emotion` and `visual-dinov3` | -0.055 | 31.0% | 2,410 |
| `final-audio-seed17` and `visual-efficientnet` | -0.011 | 31.8% | 2,410 |

**holdout**

| Pair | Logit correlation | Shared errors | Clips |
|---|---:|---:|---:|
| `final-audio-seed17` and `stream-lipsync` | +0.817 | 81.0% | 1,644 |
| `stream-emotion` and `visual-efficientnet` | +0.699 | 45.6% | 1,644 |
| `visual-dinov3` and `visual-efficientnet` | +0.637 | 44.5% | 1,644 |
| `stream-emotion` and `visual-dinov3` | +0.472 | 39.9% | 1,644 |
| `final-audio-seed17` and `stream-emotion` | +0.353 | 51.8% | 1,644 |
| `stream-emotion` and `stream-lipsync` | +0.292 | 50.1% | 1,644 |
| `final-audio-seed17` and `visual-dinov3` | +0.090 | 35.8% | 1,644 |
| `stream-lipsync` and `visual-dinov3` | +0.043 | 34.4% | 1,644 |
| `final-audio-seed17` and `visual-efficientnet` | +0.040 | 38.0% | 1,644 |
| `stream-lipsync` and `visual-efficientnet` | +0.008 | 37.2% | 1,644 |

**in-domain**

| Pair | Logit correlation | Shared errors | Clips |
|---|---:|---:|---:|
| `final-audio-seed17` and `stream-lipsync` | +0.820 | 82.5% | 1,648 |
| `stream-emotion` and `visual-efficientnet` | +0.681 | 41.7% | 1,648 |
| `visual-dinov3` and `visual-efficientnet` | +0.617 | 41.6% | 1,648 |
| `stream-emotion` and `visual-dinov3` | +0.475 | 38.6% | 1,648 |
| `final-audio-seed17` and `stream-emotion` | +0.352 | 52.1% | 1,648 |
| `stream-emotion` and `stream-lipsync` | +0.297 | 49.2% | 1,648 |
| `stream-lipsync` and `visual-dinov3` | +0.132 | 38.2% | 1,648 |
| `final-audio-seed17` and `visual-dinov3` | +0.111 | 36.5% | 1,648 |
| `final-audio-seed17` and `visual-efficientnet` | +0.049 | 37.4% | 1,648 |
| `stream-lipsync` and `visual-efficientnet` | +0.046 | 37.7% | 1,648 |
<!-- END GENERATED CORRELATION -->

### 5.4 Late fusion against deep fusion, and abstention against fallback

Two ablations the research design required. Both read the same feature stores,
so both are minutes of CPU.

<!-- BEGIN GENERATED FUSIONABLATIONS -->
| Partition | Head | ROC-AUC | Coverage | Clips scored |
|---|---|---|---:|---:|
| dfdc | late fusion, complete clips only | 0.5687 [0.5279, 0.6052] | 97.6% | 2,351 |
| dfdc | deep fusion, abstains on partial coverage | 0.5418 [0.5002, 0.5843] | 97.6% | 2,351 |
| dfdc | deep fusion, answers from what ran | 0.5411 [0.4984, 0.5828] | 100.0% | 2,410 |
| in-domain | late fusion, complete clips only | 0.9989 [0.9974, 1.0000] | 100.0% | 1,648 |
| in-domain | deep fusion, abstains on partial coverage | 0.9988 [0.9971, 1.0000] | 100.0% | 1,648 |
| in-domain | deep fusion, answers from what ran | 0.9988 [0.9971, 1.0000] | 100.0% | 1,648 |
<!-- END GENERATED FUSIONABLATIONS -->

### 5.5 BatchNorm

Both visual streams trained twice on the full training partition, one flag
apart. Labels here are `video_fake`, the objective the visual stream optimises,
which is why the in-domain column differs from section 5.1, where the fusion
store's `clip_fake` is used. A FakeAVCeleb clip with a real video track and
spoofed audio is `clip_fake` and not `video_fake`. The two labels coincide on
DFDC, where every manipulated clip has a manipulated video track, and the DFDC
columns agree to four decimals across both paths.

DINOv3 is the control: a frozen ViT has no running statistics, so the flag must
change nothing there.

<!-- BEGIN GENERATED BATCHNORM -->
| Arm | Stream | Validation | In-domain (`video_fake`) | DFDC |
|---|---|---:|---:|---:|
| frozen | `visual-efficientnet` | 0.9997 | 0.9987 | 0.5067 |
| frozen | `visual-dinov3` | 0.9559 | 0.9546 | 0.5147 |
| live | `visual-efficientnet` | 0.4909 | 0.4557 | 0.6343 |
| live | `visual-dinov3` | 0.9559 | 0.9546 | 0.5147 |

The controlled sweep that holds the epoch budget fixed has not run. `python scripts/score_batchnorm_arms.py` measures the arms; `python scripts/batchnorm_sweep.py --run-dir runs/design-b-20260910` adds the epoch control.
<!-- END GENERATED BATCHNORM -->

### 5.6 Per-media-kind thresholds

The served head takes a video, an image or a sound file, and an image reaches it
through one stream where a video reaches it through five. The fused probability
therefore sits on a different scale per kind, and one threshold cannot serve all
three.

<!-- BEGIN GENERATED THRESHOLDS -->
| Media kind | Threshold | Rows it was chosen on |
|---|---:|---:|
| audio | 0.2619 | 378 |
| image | 0.7631 | 378 |
| video | 0.5000 | 378 |

Head fitted on 1,266 holdout rows over 5 streams, best epoch 4, presence patterns {'audio': 0.15, 'image': 0.15, 'video': 0.7}.
<!-- END GENERATED THRESHOLDS -->

### 5.7 How generalization fails

MNW is fake-only, so the aggregate is a detection count rather than a ranking
metric. The aggregate hides the shape, which is in the per-generator split.

<!-- BEGIN GENERATED MNW -->
| Generator | Detected | Clips | Rate | 95% interval |
|---|---:|---:|---:|---|
| `mnw-raskai` | 0 | 4 | 0% | [0%, 49%] |
| `mnw-vasa_1` | 0 | 10 | 0% | [0%, 28%] |
| `mnw-diff2lip` | 1 | 6 | 17% | [3%, 56%] |
| `mnw-wav2lip_gfpgan` | 1 | 6 | 17% | [3%, 56%] |
| `mnw-sadtalker_video` | 1 | 5 | 20% | [4%, 62%] |
| `mnw-video_retalking` | 2 | 7 | 29% | [8%, 64%] |
| `mnw-wav2lip` | 2 | 7 | 29% | [8%, 64%] |
| `mnw-echo_mimic` | 3 | 10 | 30% | [11%, 60%] |
| `mnw-heygen_v1` | 2 | 6 | 33% | [10%, 70%] |
| `mnw-musetalk` | 2 | 6 | 33% | [10%, 70%] |
| `mnw-wav2liphq_esrgan` | 2 | 6 | 33% | [10%, 70%] |
| `mnw-sadtalker_video_v2` | 2 | 4 | 50% | [15%, 85%] |
| `mnw-wild-likely-manipulated` | 6 | 7 | 86% | [49%, 97%] |
| **all** | **24** | **84** | **29%** | [20%, 39%] |

- named generators: 18 of 77, 23% [15%, 34%]
- in the wild: 6 of 7, 86% [49%, 97%]
- the 1 genuine clip(s) in the set: 1 called fake

Almost fake-only, so no ranking metric exists and the column is a detection count at the fixed threshold. Intervals are Wilson, because the per-generator counts are 4 to 10 clips and a normal interval on 0 of 10 would read as certainty. The honest reading of a single generator's row is therefore weak; the pattern across rows is what carries.
<!-- END GENERATED MNW -->

Motion was measured against the score rather than assumed, and the result went
the other way. See section 5.8: motion suppresses the fake score, so the failure
on moving video is missed detections rather than false alarms. The generators
missed completely are the ones whose artifacts are temporally smooth, and the
clips the detector does find are the in-the-wild ones, which carry compression
and editing artifacts as well.

### 5.8 Motion, and what the model does with it

<!-- BEGIN GENERATED MOTION -->
| Corpus | Clips | Mean motion | Top quartile | Correlation, manipulated | Correlation, authentic | False alarms, calm | False alarms, moving |
|---|---:|---:|---:|---:|---:|---:|---:|
| FakeAVCeleb | 1,648 | 0.2429 | 0.3038 | -0.202 | -0.010 | 0.0% | 0.0% |
| DFDC | 2,410 | 0.2710 | 0.3627 | -0.316 | -0.035 | 0.0% | 2.7% |

Motion is the mean absolute difference between consecutive frames of the cached view, which is the tensor the model is handed. A negative correlation means more motion pushes the score towards `real`.
<!-- END GENERATED MOTION -->

### 5.9 Calibration and the operating point

<!-- BEGIN GENERATED OPERATING -->
| Partition | ROC-AUC | Precision | Recall | FPR | FPR at 95% TPR | EER | Brier | Calibration error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FakeAVCeleb, in-domain | 0.9990 | 1.0000 | 0.9981 | 0.0000 | 0.0000 | 0.0010 | 0.0018 | 0.0022 |
| DFDC, cross-corpus | 0.7500 | 0.9904 | 0.2820 | 0.0414 | 0.8690 | 0.2958 | 0.5466 | 0.6366 |

Fusion against the visual baseline, paired source bootstrap:

| Partition | Difference in ROC-AUC | 95% interval | Separated |
|---|---:|---|---|
| FakeAVCeleb, in-domain | 0.0247 | [0.0220, 0.0280] | yes |
| DFDC, cross-corpus | -0.0083 | [-0.0185, 0.0024] | no, the interval includes zero |

The paired bootstrap is the protocol's comparison and the one the research question turns on: it resamples identities and takes the difference within each resample, so the two systems are never compared across different draws.
<!-- END GENERATED OPERATING -->

### 5.10 Trained on FaceForensics++, scored on corpora it never saw

The protocol behind nearly every published cross-dataset table, and the first
number here a reader can compare with someone else's. Every other model in this
project was trained on FakeAVCeleb, which almost nobody uses as a training
corpus, so its cross-corpus figures could only be compared against themselves.

FF++ c23 carries no audio, so this is the visual stream alone.

<!-- BEGIN GENERATED ZEROSHOT -->
| Partition | ROC-AUC | Clips | Identities |
|---|---|---:|---:|
| FaceForensics++ test, in-domain | 0.8723 [0.8425, 0.9008] | 954 | 145 |
| Celeb-DF-v2, zero-shot | 0.6348 [0.5766, 0.6920] | 462 | 106 |
| DFDC, zero-shot | 0.4920 [0.4478, 0.5350] | 2,410 | 150 |
| FakeAVCeleb test, zero-shot | 0.8253 [0.7921, 0.8638] | 1,648 | 75 |

Trained on FaceForensics++ c23 and scored without re-caching, since every corpus here shares one preprocessing hash. FakeAVCeleb is included to turn the usual comparison around: it is the corpus every other model in this project was trained on.
<!-- END GENERATED ZEROSHOT -->

## 6. The findings

Stated in full, with their supersessions, in [findings](findings.md). In short:

- **F1** Neither cross-modal stream measures correspondence. Both sit on chance
  `diagonal_mass` for every epoch of every run, and one of them is the most
  accurate model in the project at 0.9991.
- **F1a** A lip-sync stream trained on LAV-DF's matched windows, where
  correspondence is the only cue the corpus leaves available, reads 0.9969 with
  its attention still at chance. The mechanism is absent even where the data
  forbids every alternative.
- **F1b** Lip-sync's collapse is head overfitting, not encoder instability. It
  breaks at epoch 2 while the encoders are still frozen, so the retrain at a
  lower encoder learning rate changed nothing.
- **F2** Live BatchNorm buys cross-corpus transfer and costs in-domain accuracy,
  and at full scale the cost is too steep to take.
- **F2b** The same visual checkpoint has two correct in-domain numbers, under
  `video_fake` and under `clip_fake`.
- **F3** Selecting on validation loss kept a checkpoint scoring below chance.
- **F4** Fusing a subset beats fusing everything, cross-corpus.
- **F5** Redundancy is measurable and was not where it was predicted.
- **F11** The identity-strict stress test the protocol requires changes nothing:
  0.8761 [0.8039, 0.9356] against 0.8723 [0.8425, 0.9008]. The ordinary split
  was not inflated by the face donor leaking across partitions.
- **F10** Training on FF++ c23, the corpus the literature treats as the
  cross-dataset standard, did not improve transfer: DFDC fell from 0.7611 to
  0.4920 and Celeb-DF was a tie. More manipulation families in training is not
  the lever.
- **F6** Motion hides manipulation rather than causing false alarms, which
  supersedes the mechanism asserted in earlier drafts.
- **F7** One decision threshold cannot serve three media kinds.
- **F8** The deep head does not beat the late head.
- **F9** Abstaining on partial coverage buys no accuracy.

## 7. Failures that changed a result

The full log is [obstacles](../obstacles.md), about 50 entries. These are the
ones that changed a number rather than costing time, and several are here
because the first diagnosis was wrong.

| Failure | What it cost | What it changed |
|---|---|---|
| Selecting a checkpoint on validation loss | Two published checkpoints withdrawn | Both trainers now select on ROC-AUC |
| Freezing BatchNorm recorded as a clean win | A wrong recommendation in three documents | It is a trade, measured at full scale |
| `diagonal_mass` compared against the wrong baseline | A claim that a stream was six times chance | It is at chance; baselines are per query length |
| A cache loader materialising every view | A run killed after two and a half hours | The loader takes the views it needs |
| A checkpoint picker walking the cache | Ten seconds per dashboard rerun | Two shallow globs |
| Windows commit charge read as free RAM | Several runs killed with 15 GB apparently free | Commit charge is the real ceiling on this platform |
| Calling the emotion stream a visual duplicate | Nearly dropped the best stream | It is best on both partitions |
| Calling lip-sync a redundant audio duplicate | Nearly dropped it | It carries the best cross-corpus combination |

## 8. Limitations

- Trained on FakeAVCeleb, not FaceForensics++. No number here is comparable to a
  published table.
- One seed per Design B stream where the protocol asks for three. Every Design B
  result is `provisional` for that reason alone.
- No in-the-wild training data.
- The cross-modal streams do not measure correspondence (F1), so the
  architecture's stated mechanism is unverified and the streams should be
  described by what they do rather than what they are named.
- Required evaluations not yet run: the identity-strict stress subset,
  leave-one-method-family-out, and the compression, noise and resolution stress
  tests.

## 9. What this system is not built for

Fully generated video is a different manipulation class and the measurements say
so. Every corpus here manipulates a real recording: a face region swapped, a
mouth re-rendered. The traces are blending seams, boundary warping and flicker
along the seam. Video from a generative model has no real source and no seam,
and the per-generator table in section 5.7 shows what that costs: 0 of 10 on
`vasa_1`, 0 of 4 on `raskai`, 1 of 6 on `diff2lip`, against 7 of 8 on the
unnamed in-the-wild clips.

One measured result makes it worse than a blind spot. Motion correlates
negatively with the fake score (section 5.8), so the model treats smoothness as
evidence of authenticity, and generated video is typically smooth. The learned
feature points the wrong way for this class.

What a system for that class would need, none of which is here:

- The whole frame, not a 224-pixel face crop. Generative implausibility shows in
  backgrounds, hands, text and reflections, which this pipeline discards.
- Frequency-domain features. Diffusion decoders and GAN upsamplers leave
  periodic artifacts; nothing here computes a spectrum.
- A leave-one-generator-out protocol, so the reported number is what happens on
  the next model rather than the one trained on. It is a required ablation in
  [the research design](../research-design.md) and has never been run.
- Provenance checks first: SynthID and C2PA are exact when present and prove
  nothing when absent.

Treating this as a refinement of the current question would be a mistake. It is
a second question, and the evidence for that is in the table.

The corpora that would answer it, with licences and the two traps to avoid, are
surveyed in [corpora for the generative-video problem](datasets.md).

## 10. Provenance

<!-- BEGIN GENERATED REGISTRY -->
16 registered results.

- 1 `pending`
- 15 `provisional (1 seed)`

The rows themselves, with each artifact's SHA-256 and MLflow run, are in [result traceability](result-traceability.md).
<!-- END GENERATED REGISTRY -->

Reproduction, in order:

```powershell
pwsh scripts/run_program.ps1            # Design A: cache, folds, branches, fusion
pwsh scripts/train_design_b.ps1         # Design B streams

# FaceForensics++, the cross-dataset corpus. Five shards, about four hours.
uv run python scripts/build_ffpp_manifest.py
uv run python -m deepfake_detection.cli split build --manifest data/manifests/faceforensics.csv --output-dir runs/ffpp-20260913/split --dataset FaceForensics++ --seed 17
# then one `cache build --shard i/5` per shard, `cache merge`, and
# `manifest usable --branch visual` per partition, followed by
# `train visual-stream --backbone efficientnet`.
uv run python scripts/score_ffpp_zeroshot.py
uv run python scripts/score_streams.py --run-dir runs/design-b-20260910
uv run python scripts/run_deep_ablation.py --run-dir runs/design-b-20260910
uv run python scripts/stream_correlation.py --run-dir runs/design-b-20260910
uv run python scripts/run_fusion_ablations.py --run-dir runs/design-b-20260910
uv run python scripts/score_batchnorm_arms.py
uv run python scripts/fit_gate_fusion.py --run-dir runs/design-b-20260910
uv run python scripts/backfill_mlflow.py --run-dir runs/design-b-20260910
uv run python scripts/update_result_registry.py
uv run python scripts/update_paper_source.py
```
