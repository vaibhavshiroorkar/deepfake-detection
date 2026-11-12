# S2-A report: Stage 2 evaluation harness

Date: 2026-10-24. Scope: `src/dfd/eval/`, `tests/eval/`, this file. No other file was edited. No git state was changed. No model, training or detection code.

## 1. What was built

`src/dfd/eval/`
- `metrics.py`: roc_auc, eer, balanced_accuracy, precision_recall, log_loss, brier, ece (top-label), ece_class1, accuracy_at; all numpy, all accept optional sample weights; `make_evaluator(name, y, s, **kw)` returns a weight-to-value function with everything weight-independent precomputed (used by the bootstrap); `OneClassError` for one-class input.
- `threshold.py`: `choose_threshold_max_balanced_accuracy` (midpoints, ties to closest to 0.5, then the lower one; maxima compared in exact integer arithmetic).
- `calibration.py`: `logit` (clip 1e-7), `sigmoid`, `apply_temperature`, `nll`, `fit_temperature` (bisection on the derivative of a convex NLL in a = 1/T, bounded T in [0.01, 100], numpy only, `at_bound` flag).
- `bootstrap.py`: `ClusterResampler` (group draws as weight vectors, redraw of one-class resamples, counted), `bootstrap_many` (several metrics on shared resamples), `cluster_bootstrap`, `paired_bootstrap`; multi-seed rule of PROTOCOL section 6 (scores of shape `(n_seeds, n)`, `seed_sd` reported). Defaults B=2000, 95 percent, seed 20261024.
- `scorefile.py`: CSV reader and validator, `ScoreTable`, frame-to-clip aggregation (mean of probabilities), sha256, `write_scores` helper.
- `guard.py`: `check_guard` (final-test guard). `ledger.py`: read and append of `ledger/final_ledger.jsonl` (default path is the repo root one; tests only use tmp paths).
- `run.py`: `python -m dfd.eval.run`; report schema `dfd-eval-report/1` is documented in its module docstring.

`tests/eval/`: `conftest.py` (synthetic score-file builder), `test_eval_metrics.py` (A1), `test_eval_threshold_calibration.py` (threshold rule, A2), `test_eval_bootstrap.py` (A3, A4, paired and multi-seed rules, redraw rule, speed), `test_eval_guard.py` (A5, library and command line), `test_eval_scorefile.py`, `test_eval_pipeline.py` (A8). A6 and A7 are left to S2-B.

## 2. How to run

The package is not installed in the venv, so outside pytest put `src` on the path:

```
PYTHONPATH=src .venv/Scripts/python.exe -m dfd.eval.run --scores scores.csv --out report.json
PYTHONPATH=src .venv/Scripts/python.exe -m dfd.eval.run --scores scores.csv --final --iteration I-0 [--repeat "reason"] [--ledger path] [--out report.json] [--n-boot 2000] [--splits val,test_x]
```
Exit codes: 0 ok, 1 invalid input, 2 refused by the guard. A first final evaluation appends one ledger line per (dataset, split) to `ledger/final_ledger.jsonl` (written only after the evaluation succeeded). The text table goes to stdout, the JSON to `--out`.

Tests: `.venv/Scripts/python.exe -m pytest tests/eval -q`

## 3. Test results (actual run, 2026-10-24)

Command: `.venv/Scripts/python.exe -m pytest tests/eval -q`
Result: `80 passed in 12.20s` (A1, A2, A3, A4, A5, A8 and the supporting tests all pass; nothing skipped, nothing relaxed).

Measured values printed by the tests (one deterministic run, fixed seeds):
- A3: the 95 percent cluster-bootstrap interval (B=2000) covered the true AUC in 192 of 200 simulated data sets (0.96; an earlier run with B=1000 gave 0.955). Required: at least 0.90. The simulation: 80 groups of 6 samples, group effect sd 0.7, noise sd 1, shift 1.0; the true AUC is the exact expectation including same-group pairs.
- A8: random scorer, 3 seeds, 100 groups per split, 4 blocks: every interval contains 0.5 and every |AUC - 0.5| < 0.06. The seed (20261024) was fixed before running. As a one-off check outside the test suite (not a test), 30 other synthetic files (120 blocks) had the interval containing 0.5 in 117 of 120 blocks (97.5 percent), consistent with the nominal 95 percent, so A8 is not a lucky seed.

## 4. Bootstrap benchmark (item 4)

2000 resamples, 50,000 samples, 5,000 groups, single seed, one metric, this machine (Windows, CPU only):
- AUC: 1.0 to 2.0 s (1.04 s inside pytest, 1.96 s in a standalone script; both measured, run to run noise)
- EER: 2.1 s
- Brier: 0.5 s; ECE: 0.3 s
Full CLI run on a synthetic file with a 10,000-row val block and a 50,000-row test block (5,000 groups), 14 metrics per block with bootstrap: 9.0 s wall clock in total. Method: a resample is a weight vector; the metric is evaluated on weights with precomputed sort order and bins, so there is no sorting and no Python loop over samples inside a resample.

## 5. Ambiguities and choices (nothing silently resolved)

1. EER "interpolation rule". The brief points to an interpolation rule in MATH_NOTES section 3; the file actually states a nearest-crossing rule (argmin |FPR - FNR|, EER = mean of the two rates), no interpolation. I implemented that rule exactly. Candidate thresholds are each distinct score (predict fake when s >= t) plus +inf; a tie in the gap goes to the first candidate in order of rising FPR. This differs from sklearn-roc-curve based interpolated EERs by about 1/n on small sets.
2. Ledger key vs. modality. PROTOCOL section 7 refuses a second final evaluation of the same iteration and dataset. If the visual and the audio scores of one iteration on one dataset are evaluated from two separate files, the second file is refused and needs `--repeat`, which STATUS.md counts as a mistake. I followed the text literally. A file may contain several modalities (they are all evaluated in one call). Alternative: add the modality to the key. Needs a decision.
3. Ledger granularity. One entry per (iteration, dataset, split) in the evaluated file; the check key is (iteration, dataset). `--repeat` given when no earlier entry exists is accepted and the reason is not stored (only real repeats are stored and flagged).
4. `--iteration` must equal the `iteration` column of the final rows, otherwise a made-up tag would defeat the ledger. Not in PROTOCOL, added as a guard strengthening.
5. Frame rows. The score file has no frame column. Rows that repeat (run_id, modality, dataset, split, sample_id) are treated as frames and averaged (section 3); the clip logit is recomputed from the mean probability (a mean of frame logits would be a different aggregation rule).
6. What "candidate" and "seed" mean in a file. One candidate = the runs sharing (iteration, modality) in the file, distinct seeds, same samples. Two different models with the same iteration and modality in one file are an error ("two runs share a seed"): one file per candidate.
7. Which val rows fit the threshold and temperature. PROTOCOL says "val of the training dataset". I pool all `val` rows of the run (normally a single dataset) and list the datasets in the report. The threshold and T are per run (per seed); the report averages the metrics over seeds, each seed using its own threshold.
8. Metrics after calibration. ECE (both versions), Brier and log loss are reported after temperature scaling. AUC and EER are not (unchanged by a monotone map, and sigmoid saturation would create false ties); balanced accuracy is not reported after scaling because the threshold is fixed before scaling (section 5).
9. Metrics on `val` and `train` are reported but flagged: the threshold and T are fitted on val, so val numbers are optimistic. `train` is bootstrapped too; `--splits` can skip it.
10. One-class blocks (for example a test set with only fakes): AUC, EER, balanced accuracy, precision and recall are omitted and noted; the other metrics are bootstrapped without redrawing. A one-class `val` is a hard error.
11. Precision is nan if nothing is predicted fake; nan replicates are dropped and counted in `n_nan`. Recall, balanced accuracy, AUC, EER raise `OneClassError` on one class.
12. ECE top-label with 15 equal-width bins on [0, 1]: confidence is at least 0.5, so the first 7 bins are always empty for top-label ECE; as defined in MATH_NOTES. Bin edges are right-closed, with float noise removed by rounding conf*M to 9 decimals.
13. Log loss clipping to [1e-7, 1 - 1e-7] makes -ln(1 - q) for q = 1 - 1e-7 differ from -ln(1e-7) by about 5e-10 because of float representation of 1 - 1e-7; the test tolerance is 1e-6 for that one case. Agreement with sklearn (given the same clipped input) is below 1e-9.
14. Temperature bounds [0.01, 100] are my choice; perfectly separable logits run to the bound (`temperature_at_bound` in the report). Synthetic probabilities that are clipped at exactly 0 or 1 give logits of about +-16.1 and can push T high; real model outputs should be saved as raw logits in the `logit` column.
15. If the `logit` column disagrees with the `score` column (sigmoid(logit) differs by more than 1e-3) a warning is written to the report; it is not an error. Calibration uses `logit`, the "before" metrics use `score`.
16. In `--splits` mode the guard looks only at the evaluated splits (val is always read to fit).

## 6. Questions for the main agent

- Decide item 2 (ledger key with or without modality) and update PROTOCOL section 7 or tell me to change it.
- Decide item 1: is the nearest-crossing EER acceptable as the protocol's definition (it is what MATH_NOTES states)?
- Should a `--compare other.csv` option (paired comparison of two candidate files, comparison rule 4) be added to `run.py`? The function `paired_bootstrap` exists and is tested but the command line does not call it.
- FILES.md and EXPERIMENTS.md were not touched (outside my scope); they need rows for these files and the A1 to A5 and A8 results above.
- `dfd` is not installed in the venv (tests work through `pythonpath = ["src"]`); `python -m dfd.eval.run` needs `PYTHONPATH=src` unless an editable install is added to pyproject (not touched).

## 7. Follow-up (2026-10-24): split dev_x (D-035, check A9 harness side) and --compare

Test run after the follow-up: `.venv/Scripts/python.exe -m pytest tests/eval -q` gives `89 passed in 12.63s` (80 earlier tests plus 9 new in `tests/eval/test_eval_devx_compare.py`).

dev_x
- `scorefile.py`: `dev_x` is in `SPLITS`; new constant `DEV_SPLITS = ("dev_x",)`; `FINAL_SPLITS` is unchanged, so the guard and the ledger ignore dev_x (verified: a file with val and dev_x rows runs without `--final`, repeatedly, and writes no ledger file; a file that also has test_x rows is still refused without `--final`, and with `--final` only test_x is logged).
- `run.py`: every block now has `role` and `development` fields; dev_x is its own block with the note "development split (D-035) ... never the headline and never pooled with test_x"; the text table shows `dev_x (dev)` and a legend line. `--splits` accepts dev_x (validation list now taken from `SPLITS`).
- Places that assumed the four split names (all handled, listed so you know): `scorefile.SPLITS` (reader check), `run.py` split validation for `--splits` (was a hard-coded set), `run.py` `ROLES` table (a new split name raises a KeyError there until a role text is added), `run.py` `--final` help text and module docstring, and the test helper in `tests/eval/conftest.py` (default layout has four splits; now accepts `layout=`). The A8 test still asserts the exact set of four blocks for its own file, which is correct for that file. Not harness code: A9's leakage part (dev_x vs test_x groups and content) is in the S2-B leakage check, not here.

--compare (`run.py`, function `compare`, paired bootstrap from `bootstrap.paired_from_evaluators`)
- Usage: `PYTHONPATH=src .venv/Scripts/python.exe -m dfd.eval.run --compare --scores A.csv [--scores-b B.csv] --run-a r1,r2,r3 --run-b q1,q2,q3 --modality visual --dataset D --split dev_x --metric auc [--n-boot 2000] [--out compare.json]`. A side is one run_id or a comma list (several = several seeds, section 6 rule). Metric: any name in the report (auc, eer, balanced_accuracy, precision, recall, accuracy_at_0.5, log_loss, brier, ece, ece_class1 and the `_cal` versions). Difference is A minus B, same resampled groups, percentile interval; prints a short text and writes JSON schema `dfd-eval-compare/1`.
- Tests: identical runs (also two seeds each) give estimate 0, interval [0, 0] for auc, eer, brier, ece_cal; a strong versus a weak scorer gives an interval above 0 and the swapped call gives the mirrored interval below 0; one-file comparison of two runs equals the library `paired_bootstrap` result; bad run_id, missing --run-a/--run-b and misaligned samples exit 1.
- Choice I made (please confirm): a comparison on `test_in` or `test_x` needs `--final --iteration` (and `--iteration-b` if side B has another tag) AND every (iteration, dataset) must already have a ledger entry from a regular final evaluation. So a comparison re-analyses logged final scores; it never makes a first look at a final split and writes no ledger entry. dev_x, val and train comparisons need no flag. If you prefer another rule (for example comparisons on final splits not allowed at all, or counted as a final evaluation), say so.
- Sides in two files must cover the same samples, labels and group_ids of the block, else exit 1.
