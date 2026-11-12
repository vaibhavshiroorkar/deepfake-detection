# Evaluation Lab and dashboard contract (2026-10-27)

Shared interface for the parallel work of 2026-10-27 evening (D-087, D-088). Agents implement against this file; changes to it go
through the main agent. File ownership is at the end.

## 1. Module assessment (D-087, implemented, read only for the UI)

Every completed or non-completed module result of a report carries `assessment` (src/dfd/platform/verdict.py):

```
assessment = {
  version: "v1", verdict: <code>, verdict_label: str, meaning: str,
  tier: "detector" | "limited" | "experimental" | "provenance" | "context" | "none",
  basis: "threshold_rule" | "heuristic" | "provenance" | "none", why: str,
  raw: {score: float|null, score_kind: "model_output" | "ranking_score" | "none", threshold: float|null},
  calibration: {is_probability: false, statement: str|null},
  uncertainty: {level: "moderate" | "high" | "not_assessed", text: str},
  reasons: [code], evidence: [{id, experiment, text}], not_established: str, mock?: true
}
```

Verdict codes and labels: `GET /api/vocabulary` -> `assessment.verdicts`. There is NO report-level verdict: the page must not
compute, show or imply one (no "overall", no combined colour, no aggregate number). The report `summary` (counts per group) stays the
report-level element. Job errors carry `error.assessment` (`unsupported_input` or `analysis_failed`).

Card order on the page: (1) verdict label, (2) the module's explanation headline, (3) supporting signals and evidence
(`assessment.evidence`, explanation sentences), (4) model and modality (`model.name`, module title), (5) limitations, (6) uncertainty
(`assessment.uncertainty.text`, `assessment.calibration.statement`). A raw score is shown only inside technical details, labelled
"model output" or "ranking score" (never as a percentage, never "probability", never "confidence"). Experimental cards show
"Experimental signal observed, no reliable verdict available" and never a verdict colour of the detector cards.

## 2. Evaluation Lab backend (package `src/dfd/platform/evallab/`)

Built on `dfd.eval` (scorefile, guard, ledger read only, metrics, bootstrap, threshold). It evaluates recorded score files; it does not
run models, train, fit on the evaluated split, write the ledger, write checkpoints or read or change `APPROVED_HEADS`.

Catalogue `GET /api/evallab/catalogue`:
```
{ score_files: [{id, path (repo-relative), sha256, n_rows, iterations: [str], modalities: [str],
                 blocks: [{dataset, split, role, n_samples, n_fake, n_real, n_groups, runs: [run_id], seeds: [int]}] }],
  roles: {train: "model development", val: "validation (threshold and temperature are fitted here)",
          dev_x: "development slice (repeatable, not final)", test_in: "final held-out (sealed)", test_x: "final held-out (sealed)"},
  default_dirs: ["data/_final/scores", "data/_evallab/imports"] }
```
Score files: the `dfd.eval.scorefile` schema (run_id, iteration, seed, modality, dataset, split, sample_id, group_id, label, score;
optional source or generator column for breakdowns). Search directories are configurable (server option or env `DFD_EVALLAB_DIRS`);
default local only. Nothing is uploaded anywhere.

Run config `POST /api/evallab/runs` (JSON):
```
{ score_file_id, iteration, modality, dataset, split, run_ids: [..] (seeds of one candidate; default all of that iteration+modality),
  threshold_source: "val_fit" (max balanced accuracy on the val rows of the same runs, as dfd.eval.run) | "fixed" (value given, with
  its provenance text), threshold?: float, threshold_provenance?: str, n_boot: int (default 2000, max 5000), seed: int (default 20261027),
  label: str, purpose: "development" | "validation" | "final_reanalysis" }
```
Rules enforced server-side: (a) `split` in test_in/test_x is accepted only if every (iteration, dataset) is already in the ledger
(`dfd.eval.guard.check_compare_guard` semantics): the lab re-analyses a logged final evaluation and never makes a first look; it
writes no ledger entry; purpose must be "final_reanalysis". (b) The threshold is never fitted on the evaluated rows; `val_fit` on a val
block is allowed but marked `fitted_on_these_rows: true` (optimistic). (c) Leakage checks before computing: group_id shared between
the evaluated block and any train or val row of the same runs (refuse for final splits, warn otherwise); duplicate sample_id within
the block; when the score file has a `sha256` or `file_hash` column, duplicate hashes across splits. (d) The config, the score file
hash and the code version are frozen into the run record before computing.

Run record (persisted to `data/_evallab/runs/<run_id>/run.json`, git-ignored; `GET /api/evallab/runs/<id>`, `GET /api/evallab/runs`):
```
{ id, created, started, finished, state: "queued"|"running"|"done"|"failed"|"refused", progress: {stage, done, total},
  config (as above, frozen), config_sha256, score_file {path, sha256, n_rows},
  role, purpose, final_guard: {required: bool, ledger_entries: [..]},
  threshold: {value, source, provenance, fitted_on_these_rows},
  data: {n_samples, n_fake, n_real, n_groups, class_balance, skipped: [{sample_id, reason}], warnings: [str], leakage: {...}},
  metrics: {name: {estimate, low, high}} for auc, eer, precision, recall, specificity, fpr, fnr, f1, accuracy, balanced_accuracy
           (group bootstrap over group_id, seed recorded),
  confusion: {tp, fp, tn, fn}, roc: {fpr: [...], tpr: [...]} (at most 200 points),
  breakdown: {column, rows: [{value, n_fake, n_real, auc|null, fpr|null, fnr|null}]} | null,
  runtime_s, error: {code, message} | null, environment: {python, git_head, dirty_entries} }
```
Comparison `POST /api/evallab/compare {run_a, run_b, metric}`: only when both runs cover the same samples, labels and groups (paired
group bootstrap, `dfd.eval.bootstrap.paired_bootstrap`); otherwise `{available: false, reason}`. Never fabricates a baseline.
Export `GET /api/evallab/runs/<id>/export?format=json|md`: the full run record, or a Markdown report with the same numbers, the
config, hashes and limitations.

## 3. Evaluation Lab page (front end)

A separate view reached from a top navigation entry "Evaluation Lab", distinct from "Analyse a file". Steps: choose a score file and a
block (dataset and split, with its role badge), review the configuration (model iteration, runs and seeds, threshold source, purpose,
samples and class balance), run, watch progress and failures, view results (metrics with intervals, confusion matrix, ROC curve drawn
from `roc` with inline SVG, breakdown table, warnings, skipped rows), compare with a baseline run (or "comparison unavailable" with the
reason), export JSON or Markdown. Final held-out runs get an explicit badge "Final held-out, re-analysis of a logged evaluation".
No metric, chart or table is hard-coded: everything comes from the API.

## 4. Design tokens (dashboard and lab)

```
--bg #F5F3EA (warm ivory)   --card #FFFDF6 (cream)   --primary #244A38 (forest green)   --sage #82977A
--accent #A66A4C (clay, restrained)   --text #282D29 (charcoal)   --border #D8D7CB (stone)
```
Green carries primary actions and navigation emphasis; clay is an accent only. Contrast at least 4.5:1 for text, visible focus rings.

## 5. File ownership

| Owner | Files |
|---|---|
| main agent | src/dfd/platform/verdict.py, verdict_evidence.py, explain.py, schema.py, modules/**, mocks.py, fixtures, tests/platform_verdict/**, docs/DECISIONS.md, docs/EXPERIMENTS.md, docs/STATUS.md, docs/TESTING_RECORD.md, docs/FILES.md, Gemini feasibility documents |
| agent C (Evaluation Lab backend) | src/dfd/platform/evallab/** (new), tests/platform_evallab/** (new), the blueprint registration and static routes in src/dfd/platform/dashboard/serve.py |
| agent D (UI) | src/dfd/platform/dashboard/index.html, dashboard.css, dashboard.js, new front-end files in that folder, tests/platform_dashboard/** |
| agent E (benchmark audit) | docs/PERFORMANCE_BENCHMARK.md, docs/LITERATURE_SURVEY.md, docs/RESEARCH.md (append only) |

Nobody changes: src/dfd/infer/**, src/dfd/web/**, checkpoints/**, ledger/**, data/_final/**, existing folders under data/_results/,
`APPROVED_HEADS`, any threshold, the lip-sync band (-6, +2), git state.
