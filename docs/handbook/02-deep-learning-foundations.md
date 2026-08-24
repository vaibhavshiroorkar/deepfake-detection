# Deep learning foundations

## Learning goals

After this chapter, you should be able to trace branch tensor shapes, compute a
two-sample binary loss, explain gradient accumulation, and distinguish frozen
from trainable parameters.

## Required background

You should know Python arrays and basic algebra. Calculus helps, but the worked
gradient defines every step.

## Tensors and shapes

A tensor is an array with a shape and data type. The first dimension in this
project is usually the batch size `B`.

The visual and sync branches accept video shaped
`[batch, time, channels, height, width]`. With current defaults, the visual
input is `[B, 16, 3, 224, 224]`. The sync mouth input is
`[B, 50, 3, 112, 112]`. Each color frame has three channels.

The audio branches accept `[batch, samples]`. A four-second audio view at
16,000 Hz is `[B, 64000]`. The two-second sync waveform is `[B, 32000]`.

For a batch of two clips, the visual tensor is `[2, 16, 3, 224, 224]`.
`VisualArtifactBranch` reshapes it to `[32, 3, 224, 224]` for the frame
backbone. It then restores `[2, 16, features]` before the GRU.

`BranchOutput` contains:

- `logits`: one raw score per clip, shape `[B]`.
- `embedding`: one learned feature vector per clip, shape `[B, features]`.
- `token_count`: the number of temporal tokens used by the branch.

## Forward pass and gradients

A forward pass applies the model to inputs. A simple binary classifier is:

```text
z = w^T x + b
p = sigmoid(z) = 1 / (1 + exp(-z))
```

Here `x` is an embedding, `w` is a weight vector, `b` is a bias, `z` is a
logit, and `p` is a probability. A logit can be any real number.

Backpropagation applies the chain rule from the loss to each parameter. For one
linear weight `w_j`:

```text
dL/dw_j = dL/dz * dz/dw_j = dL/dz * x_j
```

PyTorch records the forward operations. Calling `backward()` fills parameter
gradients. The optimizer then updates the parameters.

## Binary classification loss

For label `y` in `{0, 1}`, binary cross entropy is:

```text
L(y, p) = -y log(p) - (1 - y) log(1 - p)
```

Computing the loss from logits is more stable than first computing `p`.
PyTorch's equivalent per-sample form is:

```text
L(y, z) = max(z, 0) - z*y + log(1 + exp(-abs(z)))
```

For an unweighted batch of size `B`, the mean gradient with respect to logit
`z_i` is:

```text
dL/dz_i = (sigmoid(z_i) - y_i) / B
```

`positive_weight` in `BinaryTrainingConfig` changes the contribution from
positive examples. It can address class imbalance in training. It does not
justify balancing validation or test data.

## Optimization and regularization

Gradient descent with learning rate `eta` updates a parameter `theta` by:

```text
theta_next = theta - eta * dL/dtheta
```

Weight decay also pulls weights toward zero. A simple coupled form adds
`lambda * theta` to the gradient:

```text
theta_next = theta - eta * (dL/dtheta + lambda * theta)
```

Optimizers such as AdamW use adaptive moments and decoupled weight decay. The
optimizer is supplied to `fit_binary_branch()` by the caller.

`run_accumulated_epoch()` divides each loss by its accumulation group size,
calls `backward()`, and steps after `accumulation_steps` batches. For five
batches and an accumulation value of two, the groups have sizes two, two, and
one. This produces three optimizer steps without dropping the final batch.

Early stopping monitors validation loss. `fit_binary_branch()` saves a state
only when the loss improves by more than `minimum_improvement`. It stops after
`early_stopping_patience` stale epochs and restores the best state.

Automatic mixed precision is planned, not current. The current loop does not
use autocast or gradient scaling.

## Transfer learning

Transfer learning starts from an encoder trained on a larger task. This project
uses an EfficientNet-B0 visual backbone and Wav2Vec2 Base audio encoder.

Freezing sets encoder parameters to `requires_grad = False`. The new temporal
and classification layers can learn without changing the encoder. Unfreezing
later allows the encoder to adapt. `BinaryTrainingConfig.freeze_epochs`
controls this boundary. A model used with a nonzero value must expose
`set_backbone_trainable()`.

This staged approach reduces early damage to pretrained features and lowers
initial memory use. It can also delay useful adaptation. Unfreezing every
parameter too early can overfit a small dataset.

## Worked example

Take a batch of two logits and labels:

```text
z = [2, -1]
y = [1, 0]
p = [0.8808, 0.2689]
```

The positive sample loss is `log(1 + exp(-2)) = 0.1269`. The negative sample
loss is `log(1 + exp(-1)) = 0.3133`. Their mean is:

```text
L = (0.1269 + 0.3133) / 2 = 0.2201
```

The mean gradients are:

```text
dL/dz = [(0.8808 - 1) / 2, (0.2689 - 0) / 2]
       = [-0.0596, 0.1345]
```

Gradient descent raises the first logit's support for class 1 and lowers the
second. The signs follow from the labels, not from a hard threshold.

### Project code path

[`branches/contracts.py`](../../src/deepfake_detection/branches/contracts.py)
defines `BranchOutput`. The visual and audio modules validate input ranks and
return that contract. [`training/engine.py`](../../src/deepfake_detection/training/engine.py)
implements gradient accumulation. [`training/binary.py`](../../src/deepfake_detection/training/binary.py)
defines `BinaryTrainingConfig` and `fit_binary_branch()`.

### Design trade-offs

- Logits give stable loss computation, but users need calibrated probabilities
  for confidence claims.
- Accumulation supports larger effective batches, but updates happen less often.
- Positive weighting changes training emphasis, but not the natural test mix.
- Early stopping limits overfitting, but noisy validation loss can stop early.
- Transfer learning saves compute, but pretrained shortcuts can transfer too.

### Failure cases

- A wrong tensor rank raises a clear branch error.
- Empty training batches or non-scalar losses stop the training loop.
- A nonzero freeze period fails for models without backbone control.
- Very large logits can overflow a naive sigmoid plus log implementation.
- Test-based early stopping leaks test information into model selection.

### Supporting tests

[`test_branches.py`](../../tests/test_branches.py) checks shapes and branch
contracts. [`test_training.py`](../../tests/test_training.py) checks gradient
accumulation. [`test_training_recipes.py`](../../tests/test_training_recipes.py)
checks freezing and early stopping. Run:

```powershell
uv run pytest tests\test_branches.py tests\test_training.py `
  tests\test_training_recipes.py -v
```

## Exercises

1. Recalculate the worked loss for `z = [0, 1]` and `y = [1, 0]`.
2. Write the visual shape after flattening when `B = 3`.
3. Find the optimizer step count for seven batches accumulated in groups of
   three.
4. Set up a two-batch fixture and observe which epoch `best_epoch` records.

## Viva questions

1. Why does the loss accept logits rather than probabilities?
2. What does `token_count` mean for visual and audio branches?
3. How is gradient accumulation different from adding losses without scaling?
4. Why freeze a pretrained backbone?
5. Why is automatic mixed precision described as planned?

## Sources

- [PyTorch tensor documentation](https://docs.pytorch.org/docs/stable/tensors.html)
- [PyTorch autograd documentation](https://docs.pytorch.org/docs/stable/autograd.html)
- [BCEWithLogitsLoss documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
- [AdamW documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
- [EfficientNet paper](https://proceedings.mlr.press/v97/tan19a.html)
- [Wav2Vec 2.0 paper](https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html)
