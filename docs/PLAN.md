# PLAN

Last updated: 2026-10-24 (Session 0)

This plan lists the stages of the project, the question each stage must answer and where its output is recorded. A stage is marked done only when its output exists in the repository and the decisions it needed are logged in DECISIONS.md.

## Stages

| Stage | Question | Output | Status |
|---|---|---|---|
| 0 | What exactly is being detected, from which inputs, and what counts as success | Problem statement, requirements | not started |
| 1 | Which data, which labels, how are train and test kept apart | Data plan, leakage rules | not started |
| 2 | Metrics, splits, seeds, threshold policy, fixed before any model exists | Evaluation protocol and acceptance test | not started |
| 3 | The simplest end-to-end MVP | Baseline v0 with results | not started |
| 4 | Other models in the same harness, so the final choice has evidence | Comparison table with accuracy, cost, ease of fine-tuning | not started |
| 5 | Iterations 1 to 7 or more, each from a written hypothesis | One experiment note per iteration | not started |
| 6 | Whether and where remote heart-rate (rPPG) evidence enters | Feasibility test, then a decision | not started |
| 7 | Paper (IEEE, LaTeX), technical report, presentation | Documents assembled from the logs | not started |

Session 0 (this session) is groundwork only: machine verification, living documents, GitHub setup and the first literature pass. It does not belong to any stage above and it settles no stage question.

## Iteration slots

Each slot receives its hypothesis, success criterion and failure criterion before its run, in docs/EXPERIMENTS.md. Outcomes are written only after the run. Nothing is pre-written here.

| Slot | Role | Hypothesis | Experiment ID | Outcome |
|---|---|---|---|---|
| I-0 | MVP | to be written before the run | none yet | none yet |
| I-1 | Iteration 1 | to be written before the run | none yet | none yet |
| I-2 | Iteration 2 | to be written before the run | none yet | none yet |
| I-3 | Iteration 3 | to be written before the run | none yet | none yet |
| I-4 | Iteration 4 | to be written before the run | none yet | none yet |
| I-5 | Iteration 5 | to be written before the run | none yet | none yet |
| I-6 | Iteration 6 | to be written before the run | none yet | none yet |
| I-7 | Iteration 7 | to be written before the run | none yet | none yet |
| I-8 | Spare iteration (use only if a hypothesis justifies it) | to be written before the run | none yet | none yet |

## Timeline

Decision D-009: more than two weeks are available and the work should finish as soon as is sound. Exact dates are open (see DECISIONS.md). No dates are promised here.

## Fixed constraints taken from the measured machine

These come from ENV-DESKTOP-GPU in EXPERIMENTS.md and apply to every plan:
- One GPU with 12 GB of memory (about 11 GiB usable while the desktop is running).
- 404 GB free on D: where the repository lives; data and checkpoints must stay on D:.
- 24 CPU threads and 31 GB RAM. Video decoding and face cropping are likely to be the slowest part, and are not yet measured.
