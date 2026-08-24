# Deep learning foundations

## Learning goals

After this chapter, you should be able to trace branch tensor shapes, compute a
two-sample binary loss, explain gradient accumulation, and distinguish frozen
from trainable parameters.

## Required background

You should know Python lists and basic arithmetic. No calculus is assumed. This
chapter defines the algebra, probability, and gradient notation before using
it.

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
Here `features` is the number of learned values that represent each frame.

`BranchOutput` contains:

- `logits`: one raw score per clip, shape `[B]`.
- `embedding`: one learned feature vector per clip, shape `[B, features]`.
- `token_count`: the number of temporal tokens used by the branch.

## Forward pass and gradients

A forward pass applies the model to inputs. Start with an embedding written as
a list `x = [x_1, x_2, ..., x_K]`. A subscript is a position: `x_1` is the
first value. `K` is the number of values and has no physical unit. The model
has one weight `w_j` for each `x_j` and one bias `b`. All are ordinary real
numbers.

The classifier multiplies matching positions, adds the products, then adds the
bias:

```text
z = w_1*x_1 + w_2*x_2 + ... + w_K*x_K + b
p = sigmoid(z) = 1 / (1 + exp(-z))
```

The multiplication and sum are sometimes shortened to `w^T x`. The superscript
`T` means transpose. It turns the weight list into the orientation needed for
the same multiply-and-add operation. No matrix rotation happens in the stored
model.

The output `z` is a logit. It can be any real number and has no physical unit.
`exp(a)` means the exponential `e` raised to power `a`, where
`e` is about 2.718. Positive powers grow quickly. A negative power is a
reciprocal, so `exp(-a) = 1 / exp(a)`.

The sigmoid maps any logit to a value `p` between 0 and 1. For the visual
branch, `p` refers to class `video_fake = 1`. For the audio branch, it refers
to `audio_fake = 1`. A sigmoid output is probability-like, but calibration
must be checked before interpreting it as reliable confidence.

A gradient tells us how a small change in one number changes the loss. The
notation `dL/dz` means the local slope of loss `L` as logit `z` changes:

```text
dL/dz is approximately change_in_L / change_in_z
```

The approximation becomes exact as the change becomes extremely small. For
one weight `w_j`, changing `w_j` by a small amount changes `z` by that amount
times `x_j`. Therefore `dz/dw_j = x_j`.

The chain rule handles a sequence of changes. Multiply the loss change per
logit change by the logit change per weight change:

```text
dL/dw_j = dL/dz * dz/dw_j = dL/dz * x_j
```

Here `j` names one position. The gradient `dL/dw_j` has units of loss per unit
of weight, although both are unitless in this classifier. PyTorch records the
forward operations. Calling `backward()` applies these local rules in reverse
and fills parameter gradients. The optimizer then updates the parameters.

## Binary classification loss

Let `y` be a binary label: `1` means the branch's positive class and `0` means
its negative class. Let `p` be the model value assigned to class 1. The chance
assigned to the observed label is `p` when `y = 1` and `1 - p` when `y = 0`.

The natural logarithm `log(a)` answers: "To what power must `e` be raised to
get positive number `a`?" Thus `log(1) = 0`. Probabilities near zero have large
negative logarithms. Negating the logarithm makes a confident wrong answer
costly.

The chance assigned to either label can be written in one line:

```text
q = p^y * (1 - p)^(1 - y)
```

The caret means "raised to a power." Any nonzero value to power zero is one.
If `y = 1`, the expression becomes `q = p`. If `y = 0`, it becomes
`q = 1 - p`. Define the loss as `L = -log(q)`. The rules
`log(a*b) = log(a) + log(b)` and `log(a^c) = c*log(a)` expand it. Here `a` and
`b` are positive numbers, and `c` is a power:

```text
L(y, p) = -y log(p) - (1 - y) log(1 - p)
```

This derives binary cross entropy from the probability assigned to the correct
label. Now substitute `p = 1 / (1 + exp(-z))`. For a positive label:

```text
L(1, z) = -log(p) = log(1 + exp(-z))
```

For a negative label, `1 - p = 1 / (1 + exp(z))`, so:

```text
L(0, z) = -log(1 - p) = log(1 + exp(z))
```

Both cases combine into the binary cross entropy with logits equation:

```text
L(y, z) = log(1 + exp(z)) - y*z
```

Putting a very large `z` into `exp(z)` can overflow a computer number. PyTorch
rearranges the same equation into this stable per-sample form:

```text
L(y, z) = max(z, 0) - z*y + log(1 + exp(-abs(z)))
```

`max(z, 0)` selects the larger of `z` and zero. `abs(z)` is the distance from
zero, so it is never negative. The rearrangement avoids exponentiating a large
positive number.

The gradient follows from two local slope rules. The slope of `exp(z)` is
`exp(z)`. The slope of `log(u)` is `1/u` times the slope of its inside value
`u`. Therefore:

```text
d/dz log(1 + exp(z))
  = exp(z) / (1 + exp(z))
  = sigmoid(z)

d/dz (-y*z) = -y

dL/dz = sigmoid(z) - y
```

The notation `d/dz` means "take the slope as `z` changes." The label `y` is
fixed during this calculation. For an unweighted batch of `B` samples, the
mean divides the sum of sample losses by `B`. Each logit's gradient is also
divided by `B`:

```text
dL/dz_i = (sigmoid(z_i) - y_i) / B
```

Here `i` names one sample. `B` is the number of samples. `z_i` and `y_i` are
that sample's logit and label. Every value in this equation is unitless.

`positive_weight` in `BinaryTrainingConfig` changes the contribution from
positive examples. It can address class imbalance in training. It does not
justify balancing validation or test data.

## Optimization and regularization

Gradient descent with learning rate `eta` updates a parameter `theta` by:

```text
theta_next = theta - eta * dL/dtheta
```

`theta` is any current model parameter. `theta_next` is its value after one
update. `eta` is a positive step size with units of parameter per gradient.
`dL/dtheta` is the local loss slope for that parameter.

Weight decay also pulls weights toward zero. A simple coupled form adds
`lambda * theta` to the gradient:

```text
theta_next = theta - eta * (dL/dtheta + lambda * theta)
```

`lambda` is a nonnegative regularization strength. A larger value applies a
stronger pull toward zero. Its practical unit is chosen so the added term is
compatible with the loss gradient.

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
   Expected answer: the stable logits form avoids overflow, underflow, and
   taking the logarithm of a rounded zero.
2. What does `token_count` mean for visual and audio branches?
   Expected answer: it records the number of temporal feature positions. For
   visual it is the frame count. For audio it is the encoder token count.
3. How is gradient accumulation different from adding losses without scaling?
   Expected answer: each loss is divided by its actual group size before
   backpropagation. This preserves a mean gradient, including the final short
   group, before one optimizer step.
4. Why freeze a pretrained backbone?
   Expected answer: new layers can learn first without immediately changing
   useful pretrained features. Later unfreezing permits controlled adaptation.
5. Why is automatic mixed precision described as planned?
   Expected answer: the current loop uses neither autocast nor gradient
   scaling. Documentation must not describe future behavior as implemented.

## Sources

- [PyTorch tensor documentation](https://docs.pytorch.org/docs/stable/tensors.html)
- [PyTorch autograd documentation](https://docs.pytorch.org/docs/stable/autograd.html)
- [BCEWithLogitsLoss documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
- [AdamW documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
- [EfficientNet paper](https://proceedings.mlr.press/v97/tan19a.html)
- [Wav2Vec 2.0 paper](https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html)
