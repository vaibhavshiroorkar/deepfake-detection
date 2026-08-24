# Visual artifact branch

**Current status:** The EfficientNet-B0 plus GRU path in this chapter is
implemented. Candidate comparisons are planned research and have no project
results yet.

## Learning goals

After this chapter, you should be able to:

1. State which visual cue and label this branch learns.
2. Trace every tensor axis from a face clip to one clip logit.
3. Explain why ImageNet transfer learning and temporal aggregation are useful.
4. Separate the current baseline from planned model comparisons.

## Required background

Read [deep learning foundations](02-deep-learning-foundations.md),
[audio-video foundations](03-audio-video-foundations.md), and the
[preprocessing pipeline](05-preprocessing-pipeline.md). You need tensors,
binary logits, transfer learning, video frames, face crops, and cache hashes.

## Cue and hypothesis

The cue is visual evidence inside a tracked face. Examples include blending
boundaries, inconsistent texture, and frame-to-frame instability. The branch
hypothesis is narrow: a model trained on the cue-specific `video_fake` target
can learn visual manipulation evidence that transfers across source identities
and manipulation methods.

This is a research hypothesis, not a result. Compression, crop quality, and
dataset style can mimic manipulation evidence. Source-disjoint evaluation and
stress tests must test whether the branch learned the intended cue.

## Input shape

The current input is `frames` with shape `[B, T, C, H, W]`:

- `B` is the batch size.
- `T` is the number of sampled face frames.
- `C` is the channel count.
- `H` and `W` are crop height and width.

The default cached view has `T = 16`, `C = 3`, and `H = W = 224`. Values are
floating-point RGB channels normalized for the image backbone. The forward
method rejects any tensor that does not have five axes.

## Architecture

The current path is:

```text
[B,T,C,H,W] -> EfficientNet-B0 per frame -> [B,T,D]
              -> GRU with hidden size K -> [B,T,K]
              -> last time step -> [B,K] -> linear layer -> [B]
```

`build_efficientnet_b0()` asks `timm` for an ImageNet-pretrained
EfficientNet-B0. It sets `num_classes=0` and `global_pool="avg"`, so the
backbone returns features instead of ImageNet class scores. `D` is read from
`backbone.num_features`. The current GRU hidden size is `K = 256` by default.

ImageNet transfer learning supplies filters that already respond to edges,
textures, and object parts. The project first freezes the backbone, then makes
it trainable after `freeze_epochs`. This reduces early damage to pretrained
features while the new temporal and classification layers learn.

For one GRU step, let `x_t` be the frame feature in `R^D` and `h_(t-1)` the
previous state in `R^K`. A first-principles form is:

```text
r_t = sigmoid(W_r x_t + U_r h_(t-1) + b_r)
z_t = sigmoid(W_z x_t + U_z h_(t-1) + b_z)
n_t = tanh(W_n x_t + U_n (r_t * h_(t-1)) + b_n)
h_t = (1 - z_t) * n_t + z_t * h_(t-1)
```

Here `r_t` is the reset gate, `z_t` is the update gate, `n_t` is the candidate
state, `*` is elementwise multiplication, and every `W`, `U`, and `b` is a
learned parameter. The gates decide which past evidence to keep.

## Forward pass

`VisualArtifactBranch.forward()` performs these current operations:

1. Read `[B, T, C, H, W]` and reshape to `[B*T, C, H, W]`.
2. Run every frame through the shared backbone.
3. `_flatten_features()` converts spatial `[B*T, D, h, w]` features to
   `[B*T, D]` by averaging `h` and `w`. It also supports token features by
   averaging their token axis. The configured EfficientNet already returns
   `[B*T, D]` because global average pooling is enabled.
4. Reshape features to `[B, T, D]`.
5. Run the GRU to obtain `encoded` with shape `[B, T, K]`.
6. Select `encoded[:, -1]` as the clip embedding `[B, K]`.
7. Apply a linear classifier and squeeze the last axis to get logits `[B]`.

The returned `BranchOutput` holds the logits, the `[B, K]` embedding, and
`token_count = T`. A logit is an unbounded score. Its independent branch
probability is `sigmoid(logit)`, but fusion consumes the raw logit and performs
its own calibration.

### Worked example

Let `B = 2`, `T = 3`, `C = 3`, `H = W = 224`, `D = 4`, and `K = 2`.
Reshaping gives six images with shape `[6,3,224,224]`. The backbone returns six
four-value vectors `[6,4]`. Reshaping gives the GRU sequence `[2,3,4]`. The GRU
returns `[2,3,2]`. Selecting the final state gives `[2,2]`. The classifier
returns two logits `[2]`, one for each clip.

If one clip logit is `1.10`, its uncalibrated probability is
`sigmoid(1.10) = 1 / (1 + exp(-1.10))`, about `0.750`. This number is a worked
calculation, not a measured project score.

## Training target

The current dataset selects `record.video_fake`, not the global clip label.
For clip `i`, let `y_i` equal 1 when its video is manipulated and 0 otherwise.
Let `l_i` be the branch logit. Training uses weighted binary cross-entropy:

```text
L_i = -[w_pos y_i log(sigmoid(l_i))
        + (1 - y_i) log(1 - sigmoid(l_i))]
```

`w_pos` is the configured positive-class weight. It defaults to 1. The CLI
also uses inverse-frequency sampling on the training set. Validation loss
selects the best epoch. `save_checkpoint()` stores the model and optimizer
state with run metadata. That metadata includes the Git commit, split hash,
preprocessing hash, config hash, seed, run ID, branch name, and best epoch.
The feature store later records the resulting checkpoint hash for every row.

## Candidate comparison

| Status | Candidate | Research question |
|---|---|---|
| Current | EfficientNet-B0 plus GRU | This is the implemented baseline. |
| Planned | ConvNeXt-Tiny plus the same temporal head | Does a newer convolutional backbone improve source-grouped validation evidence at acceptable cost? |
| Optional and unimplemented | Frozen DINOv2 features plus a small temporal head | Do general self-supervised features help if compute allows a fair comparison? |

No candidate has a project result. Selection must follow the fixed protocol in
[model selection](../model-selection.md). Keep views, splits, seeds, training
budget, and evaluation code fixed. Report macro and worst-method performance,
calibration, runtime, and memory. Retain the cheaper model when the planned
confidence intervals overlap materially.

### Design trade-offs

- Per-frame transfer learning is simple and compute-aware, but it does not
  learn motion directly from adjacent pixels.
- A GRU uses ordered context at low cost, but the last-state summary can weaken
  evidence from early frames.
- Global spatial pooling makes one feature vector per frame, but it removes the
  exact location of a small artifact.
- Sixteen uniform frames cover a clip, but a short manipulation can fall
  between them.
- Staged unfreezing protects pretrained features early, but the fixed schedule
  may not be best for every dataset.

## Current limitations

- The branch has no frame-validity mask. The preprocessor supplies a fixed
  frame count or marks the visual view unavailable.
- The GRU uses only its final output. There is no learned temporal attention.
- EfficientNet sees face crops only. It cannot use body or scene evidence.
- The model can learn compression, detector, identity, or background shortcuts.
- There are no current ConvNeXt-Tiny or DINOv2 project measurements.
- A branch logit is not treated as calibrated until the fusion stage.

### Failure cases

- No stable face track makes the visual branch unavailable. Full fusion then
  abstains.
- A wrong tracked identity produces a valid tensor for the wrong person.
- Strong compression can hide real artifacts or create fake-looking texture.
- Uniform samples can miss a brief manipulated interval.
- A changed preprocessing or split hash makes old checkpoints and exported
  features incompatible with the intended run.

### Supporting tests

[`test_branches.py`](../../tests/test_branches.py) checks that the visual branch
preserves batch size and returns one logit and one embedding per clip.
[`test_training_recipes.py`](../../tests/test_training_recipes.py) covers the
binary training recipe and staged backbone control.
[`test_feature_export.py`](../../tests/test_feature_export.py) checks export of
branch logits, embeddings, availability, and provenance.
[`test_inference.py`](../../tests/test_inference.py) checks that missing evidence
causes an indeterminate result instead of a fused score.

## Project code path

1. [`ViewConfig`](../../src/deepfake_detection/views/timeline.py) defines the
   default 16 by 3 by 224 by 224 cached view.
2. [`CachedBranchDataset`](../../src/deepfake_detection/data/datasets.py) loads
   `visual_view` and selects `video_fake`.
3. [`VisualArtifactBranch` and `build_efficientnet_b0()`](../../src/deepfake_detection/branches/visual.py)
   implement the frame backbone, GRU, and clip classifier.
4. [`fit_binary_branch()`](../../src/deepfake_detection/training/binary.py)
   applies weighted binary loss, freezing, early stopping, and best-state
   restore.
5. [`save_checkpoint()`](../../src/deepfake_detection/training/checkpoints.py)
   writes the checkpoint and provenance metadata.
6. [`export_features()`](../../src/deepfake_detection/fusion/export.py) writes
   the visual logit, embedding, quality fields, and checkpoint hash.
7. [`LateFusion`](../../src/deepfake_detection/fusion/late.py) calibrates the
   raw visual logit before combining it with other evidence.

## Exercises

1. Trace `[4,16,3,224,224]` through a backbone with `D = 1280` and a GRU with
   `K = 256`. Write every intermediate shape.
2. Explain why using `clip_fake` would give the visual branch a wrong target
   for a real-video, fake-audio clip.
3. Replace final-state selection on paper with mean pooling. State one benefit
   and one loss.
4. Design a fair EfficientNet-B0 versus ConvNeXt-Tiny comparison without using
   final-test results.
5. List three shortcuts that source-disjoint splitting does not remove.

## Viva questions

1. What does this branch predict?
   Expected answer: a raw logit for the cue-specific `video_fake` target.
2. Why flatten batch and time before EfficientNet?
   Expected answer: the image backbone accepts four-axis image batches, so all
   frames share one backbone call before the time axis is restored.
3. Why use a GRU?
   Expected answer: it aggregates ordered frame features while keeping the
   baseline smaller than a full video transformer.
4. What enters fusion?
   Expected answer: the raw visual clip logit. Fusion calibrates it and adds
   audio, sync, and quality features.
5. Is ConvNeXt-Tiny better here?
   Expected answer: unknown. Its comparison is planned and must use controlled
   source-grouped validation evidence.
6. What does the checkpoint hash protect?
   Expected answer: it identifies the exact saved branch artifact attached to
   each feature row. Split and preprocessing hashes protect other provenance.

## Sources

- [EfficientNet paper](https://proceedings.mlr.press/v97/tan19a.html)
- [GRU paper](https://arxiv.org/abs/1406.1078)
- [ConvNeXt paper](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html)
- [DINOv2 paper](https://arxiv.org/abs/2304.07193)
- [timm model creation documentation](https://huggingface.co/docs/timm/reference/models)
- [PyTorch GRU documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)
