# PROTOCOL

Stage 2: the evaluation protocol and its acceptance test. Written 2026-10-24 (amended the same day by D-035: split `dev_x`, check A9) from D-011, D-012, D-021 and D-024. It is fixed before any model exists. A change after the acceptance test passes is a new decision in DECISIONS.md, never a silent edit. Definitions of the metrics are in MATH_NOTES.md.

## 1. Units, groups and roles

- Sample: one image, one video clip or one audio clip. Frame-level scores are aggregated to the clip (section 3).
- Group: the unit that may sit on one side of a split only. A group is the source video together with the person identity where metadata exists (D-012). Everything that shares a source or an identity shares a `group_id`. When identity metadata is missing the group falls back to the source video and the manifest records `identity_known = false`.
- Splits (the `split` field):
  - `train`: training data.
  - `val`: held out from the training dataset, group-disjoint from `train`; used for model selection, the threshold and temperature scaling.
  - `dev_x` (added by D-035): a development slice carved out of a cross-dataset source, group-disjoint from the final `test_x` of the same source. Used to rank candidates and choose hyperparameters; never the headline; not a final split. Chosen by the seeded rule of D-035.
  - `test_in`: in-dataset test, group-disjoint from `train` and `val`. Reported, never the headline.
  - `test_x`: a dataset never used in training. The headline generalisation result.
- `test_in` and `test_x` are final splits; `dev_x` is not (so the `--final` guard does not apply to it, and it may be evaluated repeatedly). A dataset family (for example Celeb-DF v2 and Celeb-DF++, or LAV-DF and FakeAVCeleb when identities overlap) counts as one source family: a family used in training is not a `test_x`.

## 2. Score file

One CSV per run, UTF-8, header required, one row per sample (clip level) or per frame when frame scores are stored. Columns:

| Column | Meaning |
|---|---|
| `run_id` | Unique run name |
| `iteration` | Iteration tag, for example `I-0` |
| `seed` | Integer training seed |
| `modality` | `visual`, `audio`, `image` or `fused` |
| `dataset` | Dataset key as in the manifest |
| `split` | `train`, `val`, `test_in` or `test_x` |
| `sample_id` | Manifest sample ID |
| `group_id` | Manifest group ID |
| `label` | 0 real, 1 fake |
| `score` | Probability of fake in [0, 1] |
| `logit` | Optional raw logit; if missing it is `ln(p / (1 - p))` with `p` clipped to [1e-7, 1 - 1e-7] |

## 3. Aggregation (Decided autonomously - needs user review, D-024)

Clip score = mean of the per-frame probabilities. Alternatives are iteration hypotheses, not part of the frozen protocol (R-025 shows the rule changes AUC).

## 4. Metrics

Headline: ROC-AUC with a bootstrap interval. Secondary: EER, balanced accuracy at the chosen threshold, log loss, ECE (15 equal-width bins, top-label version), Brier score; accuracy at a fixed 0.5 as a reference only; for standalone audio on ASVspoof 2019 the EER is also reported for comparison with the literature. Probabilities are clipped to [1e-7, 1 - 1e-7] before log loss. Ties in AUC count one half.

## 5. Threshold and calibration

- The threshold maximises balanced accuracy on `val` of the training dataset. Candidate thresholds are the midpoints between adjacent distinct validation scores; among equal maxima the one closest to 0.5 is taken. The chosen threshold is then fixed for every test set.
- Temperature scaling: `p = sigmoid(z / T)` with `T > 0` fitted on `val` by minimising the negative log likelihood. ECE and Brier are reported before and after. The threshold is chosen on the scores before scaling and is re-chosen on the scaled scores only if the protocol is changed by a new decision.

## 6. Uncertainty

- Seeds: at least 3 per candidate (comparison rule 2).
- Bootstrap: 2000 resamples, 95 percent percentile interval, base random seed 20261024.
- Resampling unit: the `group_id`, with replacement, keeping all samples of a drawn group. A resample that contains one class only is redrawn.
- Paired comparison: the same resampled groups for both candidates; the interval is that of the difference of the metric.
- With several seeds: for every resample the metric is computed per seed and averaged over the seeds; the interval is that of this seed-averaged value. The standard deviation over seeds of the point estimate is reported next to it.

## 7. Final-test guard and ledger (CLAUDE.md rule 3)

- Any evaluation that reads a `test_in` or `test_x` row is refused unless the run is flagged `--final` and carries `--iteration <tag>`.
- A ledger `ledger/final_ledger.jsonl` (tracked in git) records every final evaluation: date, iteration, dataset, split, run IDs and the score-file hash. A second final evaluation of the same iteration and dataset is refused unless `--repeat "<reason>"` is given; the reason is stored in the ledger and the repeat is listed under "Mistakes and corrections" in STATUS.md.

### Paired comparison on final splits (Decided autonomously - needs user review, 2026-10-24)

The `--compare` command of the harness compares two runs with the paired bootstrap. On `dev_x`, `val` and `train` it needs no flag. On `test_in` or `test_x` it needs `--final --iteration <tag>` and it refuses unless a regular final evaluation of that iteration and dataset is already in the ledger; it re-analyses logged scores, never gives a first look at a final split, and writes no ledger entry.

## 8. Acceptance test (the Stage 2 gate)

Stage 2 is done when all of the following pass; the results are logged in EXPERIMENTS.md.

| # | Check |
|---|---|
| A1 | AUC, log loss and Brier agree with scikit-learn to 1e-9 on at least 20 random score sets; balanced accuracy, EER, precision, recall and ECE equal hand-computed values on small examples |
| A2 | Temperature scaling recovers a known temperature to within 2 percent on synthetic logits |
| A3 | The cluster bootstrap interval covers the true AUC in at least 90 percent of 200 simulated data sets |
| A4 | The paired bootstrap of two identical runs gives a difference of exactly 0 with the interval [0, 0] |
| A5 | The guard refuses a final split without `--final`, refuses a second final evaluation of the same iteration without `--repeat`, and accepts a first one |
| A6 | The leakage check exits non-zero on a manufactured group overlap and zero on clean splits, including the check across source families |
| A7 | Manifests build for FF++ c23, ASVspoof 2019 LA and the ProGAN validation and test sets with the documented counts (FF++ 1,000 originals and 4,000 manipulated; ASVspoof 2019 LA 121,461; ProGAN 8,000 and 8,000) |
| A9 (added with D-035) | `dev_x` is accepted by the score file reader and is not treated as final; the leakage check fails when a group is in both `dev_x` and `test_x` and when content in `dev_x` also appears in `test_x` |
| A8 | An end-to-end run with a random scorer through the whole pipeline gives AUC about 0.5 with an interval containing 0.5 |

## 9. Not decided here

Model, backbone, fine-tuning, fusion, loss, frame sampling rate and face detector (the last is chosen by measurement in the data-preparation work, with the choice logged).
