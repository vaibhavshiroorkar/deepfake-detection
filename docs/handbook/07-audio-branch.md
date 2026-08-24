# Audio spoof branch

**Current status:** The Wav2Vec2 Base plus learned attention pooling path is
implemented. Valid-length attention masks and candidate comparisons are not.

## Learning goals

After this chapter, you should be able to:

1. Trace a waveform through Wav2Vec2, projection, attention, and classification.
2. Calculate attention weights and a pooled embedding by hand.
3. State the exact audio target and checkpoint contract.
4. Explain why missing padding masks can leak clip duration.

## Required background

Read [deep learning foundations](02-deep-learning-foundations.md),
[audio-video foundations](03-audio-video-foundations.md), and the
[preprocessing pipeline](05-preprocessing-pipeline.md). You need sampling,
waveforms, softmax, binary logits, transfer learning, and cached view shapes.

## Cue and hypothesis

The cue is evidence that an audio track was synthesized or manipulated.
Possible evidence includes vocoder residue, phase or spectral inconsistencies,
and unnatural temporal detail. The branch hypothesis is that speech features
pretrained from raw waveforms can support the cue-specific `audio_fake` task.

This remains a hypothesis until source-disjoint and method-holdout evaluation.
Recording devices, codecs, silence, and duration can become shortcuts.

## Waveform representation

The current input `waveform` has shape `[B, S]`. `B` is batch size and `S` is
the sample count. The default four-second, 16 kHz view has `S = 64,000`. A
sample is a normalized amplitude, not a spectrogram pixel.

The preprocessor centers valid audio and divides it by its standard deviation.
Short clips are padded with zeros after valid samples. The model rejects input
that does not have exactly two axes.

Wav2Vec2 first uses temporal convolutions to turn samples into latent frames.
Its Transformer adds context across those frames. Let the encoder output be
`E` with shape `[B, U, D]`, where `U` is its token count and `D` is its hidden
width. `U` is determined by the encoder's convolution strides, not assumed by
the branch.

## Architecture

The current path is:

```text
[B,S] -> Wav2Vec2 Base -> [B,U,D] -> linear projection -> [B,U,P]
      -> learned attention over U -> [B,P] -> linear layer -> [B]
```

`build_wav2vec2_audio_branch()` loads `facebook/wav2vec2-base` by default.
`D` comes from `encoder.config.hidden_size`. The current projection width is
`P = 256`. `_audio_tokens()` accepts an encoder result with
`last_hidden_state`, or a tensor directly, and requires `[B,U,D]`.

The ImageNet analogy is transfer learning for speech. Wav2Vec2 pretraining
learns contextual speech representations from unlabeled waveforms. The binary
training recipe freezes the encoder first, then makes it trainable after the
configured freeze period.

## Attention pooling

For projected token `z_(b,t)` in `R^P`, the learned scalar score is:

```text
e_(b,t) = w_a^T z_(b,t) + b_a
alpha_(b,t) = exp(e_(b,t)) / sum_(k=1)^U exp(e_(b,k))
h_b = sum_(t=1)^U alpha_(b,t) z_(b,t)
l_b = w_c^T h_b + b_c
```

`w_a` and `b_a` are attention parameters. `alpha_(b,t)` is a nonnegative
weight and the weights sum to 1 over time. `h_b` is the `[P]` clip embedding.
`w_c`, `b_c`, and `l_b` are the classifier weight, bias, and raw clip logit.

### Three-token worked example

Take one clip with `P = 2` and three projected tokens:

```text
z_1 = (1, 0)
z_2 = (0, 1)
z_3 = (1, 1)
```

Suppose the attention scores are `(0, log(2), 0)`. Their exponentials are
`(1, 2, 1)`, so the weights are `(0.25, 0.50, 0.25)`. The pooled vector is:

```text
h = 0.25(1,0) + 0.50(0,1) + 0.25(1,1) = (0.50, 0.75)
```

If `w_c = (2, -1)` and `b_c = 0`, then `l = 2(0.50) - 0.75 = 0.25`.
Its uncalibrated probability is `sigmoid(0.25)`, about `0.562`. These are
chosen numbers for learning, not project measurements.

## Training target

The current dataset selects `record.audio_fake`. It does not use `clip_fake`.
For clip `i`, `y_i` is 1 for manipulated audio and 0 for authentic audio.
With branch logit `l_i`, training minimizes:

```text
L_i = -[w_pos y_i log(sigmoid(l_i))
        + (1 - y_i) log(1 - sigmoid(l_i))]
```

`w_pos` is the positive-class weight and defaults to 1. Training uses
inverse-frequency sampling, staged encoder unfreezing, validation-loss early
stopping, and best-state restore. The saved `RunMetadata` contains seven
fields: run ID, branch, Git commit, split hash, preprocessing hash, config
hash, and seed. The selected epoch is stored separately as the top-level
checkpoint `epoch` value. Exported feature rows also carry the checkpoint hash.

## Padding and masks

**Current limitation:** padded batches lack valid-length attention masks.
The cache stores a fixed `[64,000]` waveform, but it does not store the number
of valid samples. `AudioSpoofBranch.forward()` calls `self.encoder(waveform)`
without an `attention_mask`. Its learned pooling softmax also has no mask.

For a valid-token mask `m_(b,t)` in `{0,1}`, masked attention should assign:

```text
alpha_(b,t) = m_(b,t) exp(e_(b,t))
              / sum_k m_(b,k) exp(e_(b,k))
```

That behavior is planned, not implemented. Without it, padded tokens can
receive attention. The number and position of zero-padded samples correlate
with original duration. A model can learn duration instead of spoof evidence.
Zero padding is not automatically harmless because convolutions, positional
context, biases, and attention can turn it into nonzero token features.

## Candidate comparison

| Status | Candidate | Research question |
|---|---|---|
| Current | Wav2Vec2 Base plus learned pooling | This is the implemented baseline. |
| Planned | WavLM with a matched classifier | Does denoising-oriented speech pretraining improve method holdout and corruption results? |
| Planned | AASIST under the same waveform information and budget | Does a spoof-specific spectro-temporal graph model improve cue-specific validation evidence? |

The project has no WavLM or AASIST comparison result. The planned comparison
must keep splits, seeds, view information, tuning budget, and evaluation code
fixed. Selection uses cue-specific source-grouped evidence, EER, calibration,
runtime, and memory as defined in [model selection](../model-selection.md).

### Design trade-offs

- Raw-waveform transfer avoids a hand-picked spectrogram, but the encoder is
  large and speech-pretrained rather than spoof-pretrained.
- Learned pooling can focus on brief evidence, but it can also focus on padding
  or silence without a valid-token mask.
- One clip embedding is easy to export and fuse, but it hides when an artifact
  occurred.
- Fine-tuning can adapt the encoder, but small datasets can damage general
  speech features and overfit codecs.

## Current limitations

- Valid sample lengths and attention masks are absent.
- The branch returns only clip-level evidence, with no temporal localization.
- The baseline assumes the waveform normalization suits the pretrained model.
- It can learn microphones, codecs, loudness, silence, or duration shortcuts.
- WavLM and AASIST are planned only. No project measurements exist.
- The raw branch logit is not a calibrated probability.

### Failure cases

- Missing audio makes the branch unavailable and full fusion abstain.
- Long padding can dominate attention and reveal clip duration.
- Clipping can destroy spoof evidence. `audio_clipped` reaches fusion as a
  quality feature, but it does not mask tokens.
- Music, noise, or non-speech audio may lie outside Wav2Vec2's useful domain.
- A mismatched preprocessing or split hash invalidates the checkpoint-feature
  pairing.

### Supporting tests

[`test_branches.py`](../../tests/test_branches.py) uses a small fixture encoder
that returns five temporal tokens. It checks output logit and embedding shapes
and preserves `token_count = 5`. It does not inspect attention weights or
masking.
[`test_training_recipes.py`](../../tests/test_training_recipes.py) checks that a
one-epoch binary smoke run updates a tiny branch and performs one accumulated
optimizer step. It sets `freeze_epochs = 0`, so it does not test staged
freezing.
[`test_feature_export.py`](../../tests/test_feature_export.py) checks three
named branch rows, the global clip label, selected provenance fields, and
missing-cache availability. It does not assert exported logits, embeddings,
checkpoint hashes, or the sync anomaly value.
[`test_inference.py`](../../tests/test_inference.py) checks indeterminate output
when audio plus both sync views are absent. It asserts no fused probability
and a `missing_audio` blocker; it does not isolate the audio branch from sync.

## Project code path

1. [`ViewConfig`](../../src/deepfake_detection/views/timeline.py) defines the
   default four-second, 16 kHz waveform view.
2. [`CachedBranchDataset`](../../src/deepfake_detection/data/datasets.py) loads
   `audio_view` and selects `audio_fake`.
3. [`AudioSpoofBranch` and `build_wav2vec2_audio_branch()`](../../src/deepfake_detection/branches/audio.py)
   implement encoder tokens, projection, attention pooling, and classification.
4. [`fit_binary_branch()`](../../src/deepfake_detection/training/binary.py)
   implements weighted binary loss, freezing, early stopping, and best-state
   restore.
5. [`save_checkpoint()`](../../src/deepfake_detection/training/checkpoints.py)
   records the branch artifact and training provenance.
6. [`export_features()`](../../src/deepfake_detection/fusion/export.py) writes
   the audio logit, embedding, quality fields, and checkpoint hash.
7. [`LateFusion`](../../src/deepfake_detection/fusion/late.py) calibrates the
   raw audio logit before combining it with other evidence.

## Exercises

1. For `[8,64000]` input, an encoder output width `D = 768`, `U = 199`, and
   `P = 256`, write every later tensor shape.
2. Recalculate the worked example for scores `(log(3), 0, 0)`.
3. Explain why `clip_fake` is wrong for a fake-video, real-audio clip.
4. Write pseudocode for masked attention using a `[B,U]` Boolean mask.
5. Design a test that proves padding length cannot change a valid clip score.
6. State the evidence needed before replacing Wav2Vec2 with WavLM.

## Viva questions

1. What does Wav2Vec2 return to this branch?
   Expected answer: contextual temporal tokens with shape `[B,U,D]`.
2. Why project tokens before pooling?
   Expected answer: projection gives the attention and classifier a controlled
   branch width `P`, currently 256.
3. What does the attention softmax do?
   Expected answer: it converts one learned score per token into nonnegative
   weights that sum to 1 over time.
4. What is the current masking gap?
   Expected answer: valid lengths are not stored or passed, so encoder and
   pooling attention can use padded positions.
5. What enters fusion?
   Expected answer: the raw audio clip logit, plus separately stored quality
   values. The branch embedding is stored but not used by current fusion.
6. Is AASIST the selected model?
   Expected answer: no. It is a planned controlled candidate with no project
   result.

## Sources

- [Wav2Vec2 paper](https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html)
- [Transformers Wav2Vec2 documentation](https://huggingface.co/docs/transformers/model_doc/wav2vec2)
- [WavLM paper](https://arxiv.org/abs/2110.13900)
- [AASIST paper](https://arxiv.org/abs/2110.01200)
- [AASIST official implementation](https://github.com/clovaai/aasist)
- [PyTorch softmax documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)
