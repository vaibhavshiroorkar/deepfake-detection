# Audio-video synchronization branch

**Current status:** The ResNet-18 plus Wav2Vec2 temporal alignment branch,
eight-class objective, contrastive term, and anomaly logit are implemented.
SyncNet-style and AV-HuBERT comparisons are planned.

## Learning goals

After this chapter, you should be able to:

1. Build all seven offset classes and the cross-identity mismatch class.
2. Trace mouth frames and audio samples to time-aligned token sequences.
3. Derive the offset, contrastive, and anomaly equations.
4. Explain why temporal tokens must remain separate until alignment is scored.

## Required background

Read [deep learning foundations](02-deep-learning-foundations.md),
[audio-video foundations](03-audio-video-foundations.md), the
[preprocessing pipeline](05-preprocessing-pipeline.md), and the first two model
chapters. You need cosine similarity, softmax cross-entropy, Transformers,
sampling, and shared timestamps.

## Cue and hypothesis

The cue is correspondence between visible mouth motion and speech audio. A
synthetic or edited clip may remain plausible in each modality while their
timing or identity correspondence breaks. The branch hypothesis is that
learning alignment from authentic speech pairs produces an anomaly signal for
misaligned or mismatched content.

This hypothesis does not say every deepfake is out of sync. A well-generated
fake can be aligned. Natural dubbing, voice-over, latency, poor tracking, and
non-speaking faces can also look anomalous.

## Correspondence task

The current primary dataset, `CachedSyncDataset`, trains only from authentic
clips. Each authentic mouth sequence produces eight variants:

1. Seven same-clip audio crops at fixed temporal offsets.
2. One audio crop from a different source identity.

The aligned pair has zero offset. Shifted same-clip pairs keep speaker and
content context but move audio earlier or later. The mismatch pair uses another
authentic identity and represents broken cross-modal identity correspondence.

The CLI also exposes `global-fake` as an ablation. That dataset maps authentic
clips to the zero-offset class and all fake clips to the mismatch class. It is
implemented, but it is not the primary research target. A global fake label
does not prove a clip has a synchronization error.

## Offset classes

The current class mapping is exact:

| Class index | Meaning |
|---:|---|
| 0 | -320 ms |
| 1 | -160 ms |
| 2 | -80 ms |
| 3 | 0 ms, aligned |
| 4 | +80 ms |
| 5 | +160 ms |
| 6 | +320 ms |
| 7 | cross-identity mismatch |

`OFFSET_MILLISECONDS` stores the first seven values. Positive offset means
`crop_audio_context()` starts later in the decoded audio context. For context
length `S_c`, requested output length `S_o`, sample rate `R`, and offset `d` in
milliseconds, the crop is:

```text
sample_offset = round(d R / 1000)
center_start = floor((S_c - S_o) / 2)
start = center_start + sample_offset
output = context[:, start:start + S_o]
```

The method rejects a crop that would cross the context edge. The default
preprocessor caches 2.64 seconds of real audio context around a two-second
window. That supplies 0.32 seconds on both sides, so class-specific zero
padding cannot reveal the offset.

## Architecture

The current inputs are:

- `mouth_video`: `[B,T,C,H,W]`, by default `[B,50,3,112,112]`.
- `waveform`: `[B,S]`, by default `[B,32000]` at 16 kHz.

The path is:

```text
mouth [B,T,C,H,W] -> ResNet-18 per frame -> [B,T,D_v]
                    -> projection -> video Transformer -> V [B,T,P]

audio [B,S] -> Wav2Vec2 Base -> [B,U,D_a] -> projection [B,U,P]
              -> nearest timestamp selection to T -> audio Transformer -> A [B,T,P]

V,A -> per-time cosine similarities [B,T]
    -> mean(V), mean(A), mean(similarity), max(similarity) [B,2P+2]
    -> offset head -> [B,8]
```

The builder uses ResNet-18 with its final classifier replaced by identity.
Its feature width is `D_v = 512`. Wav2Vec2 Base supplies its configured audio
width, normally `D_a = 768`. Both project to `P = 256` by default. Each
modality then has two Transformer encoder layers, four attention heads, and a
feed-forward width of `4P`.

For video token `v_(b,t)` and selected audio token `a_(b,t)`, the aligned
similarity is cosine similarity:

```text
s_(b,t) = (v_(b,t)^T a_(b,t)) / (||v_(b,t)||_2 ||a_(b,t)||_2)
```

The offset head receives `[mean_t(v), mean_t(a), mean_t(s), max_t(s)]`.
Temporal tokens must remain unpooled before `s_(b,t)` is computed. If each
stream became one vector first, the model could compare clip identity or topic,
but it could not ask whether the matching evidence occurs at the same time.

### Worked shape example

Take `B = 2`, `T = 50`, `D_v = 512`, `U = 99`, `D_a = 768`, and `P = 256`.
The 100 mouth frames are reshaped to `[100,3,112,112]`. ResNet returns
`[100,512]`, which becomes `[2,50,512]` and then `V = [2,50,256]`.
Wav2Vec2 returns `[2,99,768]`. Projection gives `[2,99,256]`. Nearest
timestamp selection and the audio Transformer give `A = [2,50,256]`. Cosine products give
`[2,50]`. Pooling gives `[2,514]` because `2P + 2 = 514`. The head returns
eight logits per clip, `[2,8]`.

## Losses

Let `o_(i,c)` be offset logit `c` for sample `i`, and let `y_i` be its class.
The offset loss is multiclass cross-entropy:

```text
L_offset = -(1/N) sum_i log(
    exp(o_(i,y_i)) / sum_(c=0)^7 exp(o_(i,c))
)
```

The current contrastive term first mean-pools temporal tokens for every
non-mismatch sample, then L2-normalizes them. Let `v_i` and `a_j` be those
normalized clip vectors. With temperature `tau > 0`:

```text
q_(i,j) = v_i^T a_j / tau
L_v2a = -(1/M) sum_i log(exp(q_(i,i)) / sum_j exp(q_(i,j)))
L_a2v = -(1/M) sum_i log(exp(q_(i,i)) / sum_j exp(q_(j,i)))
L_contrast = 0.5 (L_v2a + L_a2v)
L_total = lambda_offset L_offset + lambda_contrast L_contrast
```

`M` is the count of non-mismatch samples in the batch. The term is zero when
fewer than two such samples exist. Current defaults are `tau = 0.07`,
`lambda_offset = 1`, and `lambda_contrast = 0.1`. Shifted same-clip examples
count as corresponding for this clip-level contrastive term. Offset
cross-entropy, not the contrastive term, distinguishes their timing classes.

`sync_anomaly_logit()` converts eight class logits to one fusion input. If
aligned class index is 3, then:

```text
anomaly_logit = log(sum_(c != 3) exp(o_c)) - o_3
```

This is the log odds of all anomalous classes against the aligned class under
the same softmax denominator. The raw anomaly logit enters calibration and
fusion.

## Negative pairs

The current primary negatives have two roles:

- Shifted same-clip pairs teach offsets without changing identity or recording.
- Cross-identity pairs teach obvious correspondence failure without using a
  fake generator label.

`CachedSyncDataset` filters to authentic records. Its mismatch search selects
the next available authentic record with a different `source`. It requires at
least two authentic source identities. This rule prevents a same-source audio
track from becoming the mismatch class.

The design reduces generator shortcuts, but it does not make the task perfect.
A different source can differ in voice, channel, language, or background. The
model may solve mismatch without learning detailed lip motion.

## Candidate comparison

| Status | Candidate | Research question |
|---|---|---|
| Current | ResNet-18, Wav2Vec2, temporal Transformers, and eight-class head | This is the implemented baseline. |
| Planned | Published SyncNet-style baseline | Does a direct audio-mouth embedding objective give stronger offset and mismatch evidence at lower cost? |
| Planned if compute allows | AV-HuBERT adaptation | Do joint audio-visual speech features improve alignment and method holdout evidence? |

Neither planned candidate has project results. Keep offsets, authentic pairs,
splits, seeds, budgets, and evaluation fixed. Compare offset accuracy, mismatch
detection, temporal localization, calibration, runtime, and memory under
[model selection](../model-selection.md).

### Design trade-offs

- Authentic-only construction avoids assuming every fake is misaligned, but it
  may not cover subtle artifacts produced by modern generators.
- Offset classification is easy to inspect, but seven discrete offsets do not
  model every real timing error.
- Nearest timestamp selection has deterministic CUDA backward and preserves
  local audio tokens, but it discards tokens between selected positions.
- Separate Transformers model within-modality time before comparison, but add
  cost and can overfit small data.
- Clip-level contrastive means encourage correspondence, but do not directly
  localize the matching moment.

## Current limitations

- Transformer and Wav2Vec2 calls receive no valid-length padding masks.
- The offset head reduces its similarity sequence to mean and maximum values.
- Audio tokens use nearest timestamp selection instead of alignment by
  exact encoder receptive-field timestamps.
- Cross-identity mismatches may expose speaker or recording shortcuts.
- The primary contrastive term uses clip means after per-time tokens are made.
- SyncNet-style and AV-HuBERT comparisons remain planned and unmeasured.

### Failure cases

- Missing mouth or audio views make the branch unavailable. Full fusion
  abstains.
- Too little real audio context makes `crop_audio_context()` reject an offset.
- One authentic source cannot construct a cross-identity mismatch.
- A non-speaking face, dub, voice-over, or off-screen speaker can appear
  anomalous without manipulation.
- Wrong face tracking compares one person's mouth with another person's audio.
- A batch with fewer than two non-mismatch samples has no contrastive gradient.

### Supporting tests

[`test_branches.py`](../../tests/test_branches.py) checks temporal token shapes,
eight offset logits, and per-time similarities.
[`test_sync_objective.py`](../../tests/test_sync_objective.py) fixes the seven
offset values, checks real-context cropping, and shows lower contrastive loss
for matching fixture pairs.
[`test_sync_dataset.py`](../../tests/test_sync_dataset.py) covers authentic
offset variants and cross-source mismatches.
[`test_training_recipes.py`](../../tests/test_training_recipes.py) checks that
correct offset logits reduce training loss, shifted pairs contribute to the
contrastive term, and a one-epoch sync smoke run updates the aligned-class
offset-head bias.
[`test_feature_export.py`](../../tests/test_feature_export.py) checks that the
producer writes a row named `sync` beside visual and audio rows. It also checks
the global clip label and selected provenance fields. It does not assert the
sync anomaly logit, embedding, or checkpoint hash.

## Project code path

1. [`ViewConfig` and `make_sync_window()`](../../src/deepfake_detection/views/timeline.py)
   define 50 mouth frames, 32,000 audio samples, and shared timestamps.
2. [`CachedSyncDataset`](../../src/deepfake_detection/data/datasets.py) creates
   authentic offset and cross-identity examples.
3. [`OFFSET_MILLISECONDS` and `crop_audio_context()`](../../src/deepfake_detection/branches/sync_objective.py)
   define exact classes and real-context crops.
4. [`SynchronizationBranch` and `build_sync_branch()`](../../src/deepfake_detection/branches/sync.py)
   implement the encoders, temporal tokens, similarity, and offset head.
5. [`contrastive_alignment_loss()` and `sync_anomaly_logit()`](../../src/deepfake_detection/branches/sync_objective.py)
   implement clip-level contrast and the fusion score.
6. [`sync_training_loss()`](../../src/deepfake_detection/training/losses.py)
   combines offset and contrastive losses.
7. [`fit_sync_branch()`](../../src/deepfake_detection/training/sync.py) applies
   staged tuning, early stopping, and best-state restore.
8. [`save_checkpoint()`](../../src/deepfake_detection/training/checkpoints.py)
   records checkpoint and provenance metadata.
9. [`export_features()`](../../src/deepfake_detection/fusion/export.py) writes
   the anomaly logit, pooled embedding, quality fields, and checkpoint hash.

## Exercises

1. Convert every signed offset to samples at 16 kHz.
2. For a 42,240-sample context and 32,000-sample output, calculate the centered
   start and the start for +320 ms.
3. Trace a `[4,50,3,112,112]` video and `[4,32000]` waveform through the worked
   architecture.
4. Explain why a cross-identity pair is not equivalent to a shifted same-clip
   pair.
5. Design a negative-pair audit for language, channel, and background shortcuts.
6. Explain what information is lost if each modality is pooled before cosine
   similarity.

## Viva questions

1. How many current classes exist?
   Expected answer: eight, seven signed offsets including zero and one
   cross-identity mismatch class.
2. What is class 3?
   Expected answer: the zero-millisecond aligned class.
3. Why cache wider audio context?
   Expected answer: every shifted crop uses real samples, so padding edges
   cannot reveal the offset class.
4. Why keep temporal tokens?
   Expected answer: per-time audio and mouth evidence must be compared before
   any clip summary can express alignment.
5. What enters fusion?
   Expected answer: log-sum-exp over all non-aligned class logits minus the
   aligned logit.
6. Does the primary sync dataset use fake clips?
   Expected answer: no. It builds offsets and mismatches from authentic clips.
7. Has AV-HuBERT improved this project?
   Expected answer: unknown. That comparison is planned and unimplemented.

## Sources

- [SyncNet project and paper](https://robots.ox.ac.uk/~vgg/software/lipsync/)
- [Wav2Vec2 paper](https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html)
- [ResNet paper](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
- [Attention Is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need)
- [AV-HuBERT paper](https://arxiv.org/abs/2201.02184)
- [PyTorch TransformerEncoder documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html)
