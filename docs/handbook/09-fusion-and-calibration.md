# Fusion, calibration, and abstention

**Current status:** Per-branch sigmoid calibration, logistic late fusion,
quality features, strict missing-branch rejection, provenance validation,
balanced-accuracy threshold selection, and inference abstention are
implemented. Model comparison results are planned research.

## Learning goals

After this chapter, you should be able to:

1. Explain why fusion training requires source-grouped out-of-fold predictions.
2. Derive per-branch calibration and logistic late fusion.
3. Trace feature records through assembly, fitting, artifacts, and scoring.
4. Separate probability calibration, decision thresholds, and abstention.

## Required background

Read [data and leakage](04-data-and-leakage.md), all three branch chapters,
and [model selection](../model-selection.md). You need logits, sigmoid,
cross-validation, source-disjoint splits, hashes, and quality gates.

## Why late fusion

The three branches learn different targets and produce scores on different
scales. Late fusion keeps those branch contracts visible. It combines one raw
logit from each selected branch only after branch inference. This makes branch
ablations, calibration checks, missing evidence, and provenance easier to
audit than one end-to-end multimodal network.

The current primary fusion model is regularized logistic regression. Its input
has shape `[N,J+3]`, where `N` is the number of clips and `J` is the number of
selected branches. The first `J` columns are calibrated branch logits. The
last three are quality features.

## Out-of-fold predictions

A fusion row must predict a clip with a branch checkpoint that did not train on
that clip's source identity. Otherwise, the branch score carries training-set
knowledge. A fusion model can learn that overconfident pattern and appear
better during meta-training than it will on unseen identities.

Current cross-fitting groups by `source`. All folds stay inside the frozen
training partition. Validation and test sources never enter branch cross-fit
training. `ddf train fusion` requires every assembled row to have
`partition_role = "oof"`.

### Four-source worked example

Suppose the training partition has sources `A`, `B`, `C`, and `D`. A two-fold
teaching example is:

| Fold | Train branch checkpoints on | Export predictions for |
|---|---|---|
| 1 | A, B | C, D |
| 2 | C, D | A, B |

Concatenate the held-out predictions. Every source appears once as held out.
None of its fusion rows came from a checkpoint trained on that source. Fit
calibrators and fusion on the concatenated rows. For final validation or test
scoring, train branch checkpoints only on the allowed development data, then
apply the fitted protocol without using locked labels.

The actual CLI supports a configurable fold count and defaults to three. The
four-source table explains the rule. It does not describe a recorded run.

## Calibration

Each branch `j` produces a raw logit `l_(i,j)` for clip `i`. There are `N`
out-of-fold meta-training clips. The global fusion label is `y_i` in `{0,1}`,
where 1 means the clip is fake. Define the signed label `t_i = 2y_i - 1`, so
`t_i` is in `{-1,+1}`. The current calibrator maps a score with:

```text
p_(i,j) = sigmoid(a_j l_(i,j) + b_j)
sigmoid(x) = 1 / (1 + exp(-x))
c_(i,j) = log(p_(i,j) / (1 - p_(i,j)))
```

The score mapping does not explain how `a_j` and `b_j` are fitted. For branch
`j`, the current `liblinear` logistic regression minimizes this equivalent
L2-regularized negative log-likelihood objective:

```text
J_cal,j(a_j, b_j) = sum_(i=1)^N log(
    1 + exp(-t_i (a_j l_(i,j) + b_j))
) + (a_j^2 + b_j^2) / (2C)
```

`a_j` is the learned slope, `b_j` is the learned intercept, and `C > 0` is the
scikit-learn inverse regularization strength. `liblinear` represents the
intercept as a synthetic feature with default `intercept_scaling = 1`, so its
weight is also penalized. Multiplying the full expression by positive `C`
gives liblinear's equivalent form: `C` times the loss sum plus one half the
squared parameter norm. The current default is `C = 1.0`.

`p_(i,j)` is the fitted branch probability estimate. `c_(i,j)` converts it
back to calibrated log-odds for fusion. The implementation clips probability
to `[1e-6, 1 - 1e-6]` before the log-odds transform to avoid infinity. This is
the project's sigmoid, or Platt-style, calibration step.

Calibration and decision thresholds solve different problems. Calibration
maps a score to a probability estimate. A threshold maps the final probability
to a decision. The current `ddf threshold` command selects the candidate that
maximizes balanced accuracy on a supplied scored file. It tests zero, one, and
midpoints between distinct probabilities. Ties prefer the value nearest 0.5,
then the lower value. The predictor receives the chosen threshold and checks
that it lies in `[0,1]`. The research protocol requires a validation file for
selection. It must not use test labels.

## Fusion features

For `J` selected branches, current `FusionSample` holds:

```text
x_i = [c_(i,1), ..., c_(i,J), f_i, k_i, d_i]
```

`f_i` is face coverage in `[0,1]`. `k_i` is 1 when audio clipping was detected
and 0 otherwise. `d_i` is the audio-video duration difference in seconds. The
logistic fusion model is:

```text
eta_i = beta_0 + sum_(j=1)^J beta_j c_(i,j)
        + beta_f f_i + beta_k k_i + beta_d d_i
p_i = sigmoid(eta_i)
```

Let `Q = J + 3` be the fusion feature count. Let `beta` in `R^Q` contain the
weights for every calibrated branch logit and quality value. Let `beta_0` be
the intercept, so `eta_i = beta^T x_i + beta_0`. The current logistic fusion
fit minimizes:

```text
J_fusion(beta, beta_0) = sum_(i=1)^N log(
    1 + exp(-t_i (beta^T x_i + beta_0))
) + (||beta||_2^2 + beta_0^2) / (2C)
```

`||beta||_2^2` is the sum of squared feature weights. `N`, `t_i`, and `C` have
the definitions above. The current model again uses scikit-learn
`LogisticRegression`, `liblinear`, default `intercept_scaling = 1`, and
`C = 1.0`. The intercept is therefore penalized as a synthetic-feature
weight. Smaller `C` means stronger L2 regularization. After fitting, the model
maps `eta_i` to final probability `p_i`; fitting and score mapping are separate
steps.

### Worked score example

Assume calibrated branch probabilities are visual `0.80`, audio `0.60`, and
sync `0.70`. Their log-odds are about `1.3863`, `0.4055`, and `0.8473`.
Suppose quality is `(face_coverage, audio_clipped, duration_delta) =
(0.90, 0, 0.02)`. Choose teaching coefficients only:

```text
beta_0 = -1
branch weights = (1, 0.5, 0.75)
quality weights = (0.2, -0.3, -2)
```

Then `eta` is about:

```text
-1 + 1.3863 + 0.5(0.4055) + 0.75(0.8473)
   + 0.2(0.90) - 0.3(0) - 2(0.02) = 1.3645
```

The fused probability is `sigmoid(1.3645)`, about `0.796`. This is a worked
calculation. It is not a learned project coefficient or experiment result.

`FeatureRecord` is the stored branch-level row. It contains dataset, clip,
segment, branch, logit, embedding, availability, global clip label, quality
fields, source and subgroup metadata, partition role, checkpoint hash,
preprocessing hash, split hash, cache fingerprint, and run ID.

`FeatureStore.assemble()` groups records by dataset, clip, and segment. In
strict mode it requires every selected branch to be present and available. It
also rejects conflicting labels, preprocessing hashes, split hashes, run IDs,
source metadata, or partition roles. Checkpoint hashes remain branch-specific
inside the assembled row.

`LateFusion.fit()` validates complete branch logits, fits one calibrator per
branch, appends quality columns, and fits logistic regression or the optional
MLP. `FusionArtifact` stores the fitted model with its split and preprocessing
hashes. Scoring validates those hashes at load boundaries, builds the same
ordered features, calls `predict_proba()`, and uses column 1 as fake
probability.

## Missing evidence

The current primary rule is rejection, not imputation. `FeatureStore.assemble()`
uses `strict=True` by default. An absent record or `available=False` record
raises an error for fusion training. `LateFusion` also rejects a sample missing
any configured branch logit.

At video inference, preprocessing quality blockers or a missing visual, audio,
or sync branch produce:

```text
verdict = "indeterminate"
probability = None
```

Available branch logits and blocker names remain in the result for audit.
The system does not replace a missing score with zero, real, fake, or a mean.
This is abstention. Coverage and abstention rate must appear beside predictive
metrics, because a model can improve apparent accuracy by refusing hard clips.

## Ablations

| Status | Ablation | Purpose |
|---|---|---|
| Current interface | Any configured subset of visual, audio, and sync logits | Measure each branch and pair against all-branch fusion. |
| Current optional model | One hidden-layer MLP with 8 units | Test whether nonlinear feature interaction helps. |
| Planned evaluation | Quality features included versus removed | Test whether quality explains reliable score variation or becomes a shortcut. |
| Planned evaluation | Quality-aware abstention versus silent fallback | Measure performance together with coverage. Silent fallback is not the primary path. |
| Planned candidate | Isotonic calibration when sample size supports it | Compare validation Brier score and calibration error with Platt scaling. |

The interfaces exist for branch subsets and the MLP. No ablation result is
claimed. Logistic fusion stays primary unless the predeclared validation
protocol selects another candidate across the fixed seeds.

### Design trade-offs

- Logistic fusion is inspectable and data-efficient, but it models only linear
  interactions in calibrated log-odds and quality values.
- Per-branch calibration makes scales comparable, but adds fitted parameters
  and needs held-out predictions.
- Quality features can teach reliability, but can also encode dataset or method
  shortcuts.
- Strict complete-case fusion protects semantics, but lowers coverage.
- An MLP can model interactions, but has greater overfitting and stability risk.

## Current limitations

- Calibration and fusion fit on the same out-of-fold meta-training rows. The
  branch checkpoints remain out of sample by source, but calibration does not
  use a nested meta-fold.
- Logistic coefficients do not prove causal cue importance.
- Only three quality features enter the current model.
- Strict fusion cannot produce a partial-modality probability.
- The artifact stores split and preprocessing hashes, while branch checkpoint
  hashes stay in feature rows rather than `FusionArtifact`.
- The current threshold objective is balanced accuracy only. Other operating
  constraints require a planned protocol change before test evaluation.

### Failure cases

- In-sample branch predictions leak training knowledge into fusion.
- A missing branch rejects training assembly and causes inference abstention.
- Conflicting split, preprocessing, run, label, or source metadata rejects
  feature assembly.
- A changed branch order changes feature meaning. `branch_names` preserves the
  configured order in the artifact.
- Poor calibration can make a high raw score look more certain than evidence
  supports.
- A threshold tuned on test labels invalidates the locked evaluation.
- Reporting scores without coverage hides the cost of abstention.

### Supporting tests

[`test_crossfit.py`](../../tests/test_crossfit.py) checks that each source is
held out once and that train and holdout sources never overlap.
[`test_feature_store.py`](../../tests/test_feature_store.py) checks Parquet
round trips, duplicate keys, strict coverage, unavailable rows, and provenance
conflicts.
[`test_fusion.py`](../../tests/test_fusion.py) fits the three-branch logistic
path and checks that a negative fixture scores below 0.5 while a positive one
scores above 0.5. Separate cases check missing-branch rejection, ordered MLP
scores, and split or preprocessing hash rejection.
[`test_cli.py`](../../tests/test_cli.py) runs fusion training and scoring with
rows marked `oof`. It checks artifact type and hashes, ordered output
probabilities, populated visual probabilities, preserved sources, and a blank
fused probability for an incomplete appended clip. It has no non-OOF rejection
case.
[`test_metrics.py`](../../tests/test_metrics.py) checks one valid
balanced-accuracy threshold and rejection of single-class labels. It does not
test invalid probability values.
[`test_inference.py`](../../tests/test_inference.py) checks one complete fused
prediction. Its incomplete fixture removes audio plus both sync views, then
checks indeterminate output, no probability, and a `missing_audio` blocker. It
does not isolate missing sync.

Implementation guarantees beyond those isolated tests are explicit in code:
`_train_fusion()` rejects assembled partition roles other than `oof`;
`select_balanced_accuracy_threshold()` rejects nonfinite or out-of-range
probabilities; and `PredictionEngine.predict()` checks every configured branch
before fusion.

## Project code path

1. [`build_group_folds()`](../../src/deepfake_detection/training/crossfit.py)
   creates repeatable source-grouped train and holdout indices.
2. [`export_features()`](../../src/deepfake_detection/fusion/export.py) creates
   one `FeatureRecord` per clip and branch with logit, availability, quality,
   the global clip label, checkpoint hashes, and other provenance.
3. [`FeatureRecord` and `FeatureStore.assemble()`](../../src/deepfake_detection/fusion/store.py)
   persist and validate branch evidence.
4. [`FusionSample`, `LateFusion.fit()`, and `LateFusion.predict_proba()`](../../src/deepfake_detection/fusion/late.py)
   calibrate branch logits, append quality features, fit fusion, and score.
5. [`FusionArtifact`](../../src/deepfake_detection/fusion/late.py) binds the
   fitted model to split and preprocessing hashes.
6. [`_train_fusion()`](../../src/deepfake_detection/cli.py) requires assembled
   rows to have `partition_role = "oof"` and saves the artifact.
7. [`PredictionEngine.predict()`](../../src/deepfake_detection/inference/predictor.py)
   computes current branch logits, abstains on blockers, or applies fusion and
   the separately supplied threshold.

## Exercises

1. Build three source-grouped folds for six sources. Show that every source is
   held out once.
2. Convert probabilities `0.2`, `0.5`, and `0.9` to log-odds.
3. Recalculate the worked fusion example after setting `audio_clipped = 1`.
4. Explain why calibrating in-sample branch scores does not remove leakage.
5. Design a table that reports accuracy, coverage, and abstention together.
6. State which hashes belong to a feature row and which belong directly to a
   fusion artifact.

## Viva questions

1. Why require out-of-fold branch predictions?
   Expected answer: each meta-training score must come from a checkpoint that
   did not train on that clip's source, or fusion can learn training leakage.
2. What is Platt scaling here?
   Expected answer: one logistic regression per branch maps its raw logit to a
   calibrated probability, then fusion uses the calibrated log-odds.
3. What are the current quality features?
   Expected answer: face coverage, audio-clipped indicator, and audio-video
   duration difference.
4. What happens when sync evidence is missing?
   Expected answer: strict training assembly rejects the row, and inference
   returns indeterminate with no fused probability.
5. Is a 0.5 threshold part of calibration?
   Expected answer: no. Calibration estimates probability. Threshold selection
   is a separate validation decision.
6. Does the MLP currently beat logistic fusion?
   Expected answer: unknown. It is an implemented ablation with no claimed
   project result.
7. What does `FusionArtifact` validate?
   Expected answer: the split hash and preprocessing hash used by the fitted
   fusion model.

## Sources

- [Stacked generalization paper](https://doi.org/10.1016/s0893-6080%2805%2980023-1)
- [Platt calibration paper](https://www.cs.cornell.edu/courses/cs678/2007sp/platt.pdf)
- [Predicting good probabilities with supervised learning](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf)
- [scikit-learn probability calibration guide](https://scikit-learn.org/stable/modules/calibration.html)
- [scikit-learn LogisticRegression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [scikit-learn GroupKFold documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
