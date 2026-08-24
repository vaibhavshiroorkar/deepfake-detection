# Project handbook and research evidence design

**Status:** Approved
**Date:** 2026-08-24
**Branch policy:** Maintain the project on `main`.

## Context

The repository has a tested research platform and a short set of protocol
documents. It does not yet teach the implementation. The current documents
state rules, but they do not explain the required theory, code flow, tensor
shapes, experiments, or findings.

The student knows basic Python and Git. The documentation must teach PyTorch,
deep learning, statistics, and audio-video processing from first principles.
Reading it should provide the knowledge gained by building the project.

The project will use AI assistance for implementation. The documentation must
make every generated choice inspectable and defensible. It must never claim
that an experiment ran when no evidence exists.

## Goals

The documentation system must:

1. Teach the full project in a deliberate learning order.
2. Explain the theory behind each implemented component.
3. Trace research claims to code, data, runs, and artifacts.
4. Record why major decisions were accepted or rejected.
5. Update with each material implementation change.
6. Prepare the student to reproduce, explain, criticize, and defend the work.
7. Support the final paper without creating a separate version of the truth.

## Non-goals

The documentation will not:

- Pretend that implementation volume is a research contribution.
- Copy source code into prose without explaining its purpose.
- Store daily progress reports that become stale.
- Copy all MLflow metrics into Markdown files.
- Publish raw media, face crops, private paths, or restricted artifacts.
- Invent positive findings before the locked experiments run.
- Optimize the system for public deployment or automated moderation.

## Documentation architecture

Use three connected content layers: a handbook, technical references, and
research evidence. Architecture decision records preserve major choices.

```text
README.md
CHANGELOG.md
ROADMAP.md
docs/
  README.md
  handbook/
    README.md
    00-learning-path.md
    01-problem-and-research-question.md
    02-deep-learning-foundations.md
    03-audio-video-foundations.md
    04-data-and-leakage.md
    05-preprocessing-pipeline.md
    06-visual-branch.md
    07-audio-branch.md
    08-sync-branch.md
    09-fusion-and-calibration.md
    10-training-system.md
    11-evaluation-and-statistics.md
    12-inference-and-dashboard.md
    13-reproducing-the-project.md
    14-viva-preparation.md
  reference/
    architecture.md
    configuration.md
    cli.md
    artifact-contracts.md
    testing.md
    hardware-and-compute.md
  research/
    questions-and-hypotheses.md
    experiment-matrix.md
    metrics-and-statistics.md
    result-traceability.md
    findings.md
    error-analysis.md
    paper-outline.md
  decisions/
    ADR-001-local-mlflow.md
    ADR-002-source-disjoint-splits.md
    ADR-003-calibrated-late-fusion.md
    ADR-004-quality-aware-abstention.md
    ADR-005-detector-bakeoff.md
```

### Root documents

`README.md` remains a short entry point. It explains setup, common commands,
and documentation navigation. It must not become the full handbook.

`ROADMAP.md` owns phase order and exit gates. It lists planned work without
claiming completion.

`CHANGELOG.md` records material software and protocol changes. Experiment runs
belong in MLflow. Accepted findings belong in the research layer.

### Handbook

The handbook is the learning path. Its chapters follow the order in which data
moves through the project.

| Chapter | Purpose |
|---|---|
| `00-learning-path.md` | Explain the reading order, prerequisites, exercises, and expected outcomes. |
| `01-problem-and-research-question.md` | Define deepfakes, project scope, research questions, hypotheses, and contribution. |
| `02-deep-learning-foundations.md` | Teach tensors, gradients, losses, optimizers, transfer learning, regularization, and PyTorch modules. |
| `03-audio-video-foundations.md` | Teach frames, sampling rates, timestamps, codecs, synchronization, and temporal windows. |
| `04-data-and-leakage.md` | Teach manifests, cue labels, identity leakage, shortcuts, splits, and dataset limits. |
| `05-preprocessing-pipeline.md` | Follow decoding, timelines, detection, tracking, alignment, views, caching, hashes, and abstention. |
| `06-visual-branch.md` | Explain spatial artifacts, the visual encoder, temporal aggregation, training, and candidate comparisons. |
| `07-audio-branch.md` | Explain waveforms, speech encoders, masks, attentive pooling, spoof cues, and candidate comparisons. |
| `08-sync-branch.md` | Explain correspondence learning, offsets, negative pairs, temporal tokens, and mouth alignment. |
| `09-fusion-and-calibration.md` | Explain out-of-fold predictions, Platt scaling, logistic fusion, alternatives, and missing evidence. |
| `10-training-system.md` | Explain datasets, loaders, seeds, checkpoints, early stopping, mixed precision, and MLflow. |
| `11-evaluation-and-statistics.md` | Explain metrics, thresholds, calibration, bootstrap intervals, paired tests, and subgroup limits. |
| `12-inference-and-dashboard.md` | Trace one video through loading, prediction, coverage, verdicts, and presentation. |
| `13-reproducing-the-project.md` | Rebuild the environment, data protocol, caches, models, results, and paper tables. |
| `14-viva-preparation.md` | Provide expected questions, defensible answers, weaknesses, and live demonstration steps. |

### Technical references

Reference documents answer exact operational questions. They do not repeat the
handbook's lessons.

- `architecture.md` maps packages, dependencies, interfaces, and data flow.
- `configuration.md` defines every configuration field, default, unit, and
  validation rule.
- `cli.md` documents supported commands and realistic examples.
- `artifact-contracts.md` defines manifests, caches, checkpoints, predictions,
  feature stores, thresholds, and hashes.
- `testing.md` maps test layers to research risks and common commands.
- `hardware-and-compute.md` records the verified machine and compute policy.

### Research evidence

The research layer connects experiments to the paper.

- `questions-and-hypotheses.md` preregisters questions and falsifiable claims.
- `experiment-matrix.md` lists comparisons, controls, budgets, seeds, and
  acceptance rules.
- `metrics-and-statistics.md` defines calculations and interpretation.
- `result-traceability.md` maps each paper result to immutable evidence.
- `findings.md` records accepted results after the required runs finish.
- `error-analysis.md` classifies failures and links them to reviewed samples.
- `paper-outline.md` maps evidence to the final paper sections and figures.

`findings.md` starts with the statement that no findings exist. That statement
changes only when complete tracked evidence supports a conclusion.

### Architecture decision records

ADRs record decisions that affect several modules or research claims. Each ADR
contains context, options, trade-offs, decision, consequences, and review
triggers.

Accepted ADRs are not rewritten to hide earlier reasoning. A later ADR
supersedes an earlier decision when evidence changes the choice.

The initial ADRs cover local MLflow, source-disjoint splits, calibrated late
fusion, quality-aware abstention, and the detector comparison.

## Chapter content standard

Every technical handbook chapter uses the following sections when applicable:

1. Learning goals.
2. Required background.
3. Problem definition.
4. Theory in plain language.
5. Equations with every symbol and unit defined.
6. Input and output shapes.
7. A small worked example.
8. Project implementation and symbol-level code path.
9. Design choices and rejected alternatives.
10. Failure modes and debugging checks.
11. Tests that support the described behavior.
12. Exercises that can run without the full dataset when possible.
13. Viva questions with concise expected answers.
14. Primary sources and further reading.

The chapter must separate general theory from project-specific behavior. A
reader should know which statements come from literature and which come from
this implementation.

Use small diagrams when relationships are hard to explain in prose. Use Mermaid
or ASCII diagrams that render on GitHub. Every diagram must teach a concrete
data flow, dependency, or state transition.

## Writing rules

- Use plain English and define every technical term on first use.
- State the point before its explanation.
- Keep examples small enough to calculate by hand.
- Explain code by symbol and responsibility, not by copying whole files.
- Use stable file paths and symbol names. Avoid line-number references in the
  repository because they become stale.
- Cite primary papers, official documentation, dataset papers, and source
  repositories.
- Mark planned behavior as planned. Describe only implemented behavior as
  current.
- Record uncertainty, limitations, and negative findings.
- Follow the repository's ASCII punctuation rules.

## Research questions

The study centers on these questions:

### RQ1: Fusion generalization

Does calibrated cue-specific fusion generalize better than a strong visual
baseline under source-disjoint and shortcut-controlled evaluation?

### RQ2: Branch contribution

Which visual, audio, and synchronization cues remain useful on unseen
manipulation families and an external dataset?

### RQ3: View integrity

How do face detection, tracking, and landmark alignment affect coverage,
synchronization quality, and downstream performance?

### RQ4: Reliability

How do calibration and quality-aware abstention affect confidence and coverage
under missing or degraded evidence?

### RQ5: Cost

Which component choices provide the best validation evidence within the local
compute budget?

These questions permit negative answers. The paper must not redefine them after
the locked test runs.

## Comparison design

Use sequential controlled comparisons instead of a full Cartesian search.

### View pipeline

- Compare MTCNN with YuNet on a reviewed training-only sample.
- Compare greedy IoU tracking with one motion-aware method.
- Compare box-relative mouth crops with landmark-aligned crops.
- Freeze the winning view configuration before branch selection.

### Branch models

- Compare EfficientNet-B0 plus GRU with ConvNeXt-Tiny.
- Treat a frozen DINOv2 encoder as optional when the compute budget permits it.
- Compare Wav2Vec2 Base with WavLM and AASIST.
- Compare the current sync branch with a published SyncNet-style baseline.
- Treat AV-HuBERT as optional because it has higher cost and integration risk.

### Fusion and reliability

- Compare every single branch, each branch pair, and all branches.
- Compare calibrated logistic regression with the small MLP.
- Compare Platt scaling with isotonic regression only when validation size
  supports it.
- Compare quality-aware abstention with silent fallback.
- Compare authentic correspondence learning with global fake-label sync
  training.

Each model comparison uses the same split, seeds, maximum training steps,
selection metric, and tuning budget. Change only the factor under study.

## Measurement plan

### Predictive metrics

Report ROC-AUC, PR-AUC, balanced accuracy, precision, recall, F1, FPR, FNR,
EER, FPR at 95 percent TPR, Brier score, and expected calibration error.

### Generalization metrics

Report source-disjoint, identity-strict, unseen-method, external zero-shot, and
corruption results. Report the worst manipulation family beside pooled scores.

### Evidence availability

Report face coverage, stable-track coverage, audio coverage, fusion coverage,
abstention rate, and failure reasons. Keep failed rows in denominators.

### Efficiency metrics

Report parameter count, peak GPU memory, training time, inference time,
preprocessing throughput, and artifact storage.

### Statistical evidence

Use three fixed training seeds for accepted branch comparisons. Use 1,000
source-identity bootstrap samples for confidence intervals. Use paired source
bootstraps for fusion comparisons. Report subgroup results only with sample
counts and intervals.

Do not claim a meaningful improvement from a single seed or a pooled metric
without uncertainty.

## Experiment lifecycle

Every research comparison follows this sequence:

```text
question
  -> hypothesis and decision rule
  -> frozen comparison configuration
  -> smoke run
  -> complete fixed-seed runs
  -> validation analysis
  -> accepted or rejected decision
  -> locked test evaluation when permitted
  -> finding and paper traceability
```

The experiment record includes:

- Research question and hypothesis.
- Independent, dependent, and controlled variables.
- Dataset, partition role, and split hash.
- Preprocessing and feature-store hashes.
- Git commit and dirty-worktree flag.
- Candidate names and exact model revisions.
- Compute budget, seeds, and stopping rule.
- Primary and secondary metrics.
- MLflow run IDs.
- Checkpoint, prediction, and report hashes.
- Result, confidence interval, and cost.
- Interpretation and threats to validity.
- Decision and rejection reason.

Failed and interrupted runs remain visible. Retries receive new run IDs.

## Result traceability

Every table, figure, and headline number in the paper receives a stable result
identifier. The traceability table maps that identifier to:

```text
paper item
  -> analysis command and configuration
  -> metric report and prediction file hashes
  -> MLflow run IDs
  -> checkpoint hashes
  -> preprocessing and split hashes
  -> Git commit
```

The paper may round values for display. The stored metric report remains the
numeric source of truth.

## Experiment tracking

Use local MLflow with SQLite metadata and filesystem artifacts. Keep current
JSON histories, Parquet feature stores, checkpoints, and audit files. MLflow
indexes these artifacts and records their relationships.

Do not log raw videos or derived biometric crops into MLflow. Store local paths,
dataset identifiers, counts, and hashes instead.

W&B remains a review option if hosted supervisor sharing becomes a firm
requirement before tracker integration. Do not instrument both systems.

## Hardware and compute policy

The verified development machine has:

- AMD Ryzen 5 5600X with 6 cores and 12 threads.
- NVIDIA GeForce RTX 5070 Ti with 16,303 MiB reported VRAM.
- 32 GB system memory.
- Windows 11 Pro.
- More than 1.4 TB free across the inspected local drives.

Use NVIDIA's reported VRAM rather than Windows adapter-memory fields. Windows
reported an incorrect four GB adapter value for this card.

The compute policy is:

- Use automatic mixed precision for supported GPU training.
- Use gradient accumulation when a fair batch size does not fit.
- Prefer base or small backbones for full fine-tuning.
- Freeze large foundation encoders unless a measured need justifies tuning.
- Record peak memory and wall-clock time for every accepted comparison.
- Keep raw media, caches, and large runs on a spacious configurable data drive.
- Keep the repository and small metadata independent of machine-specific paths.
- Limit candidate and hyperparameter counts before experiments begin.

Hardware limits affect batch size and time. They must not change test data,
selection metrics, or statistical rules.

## Documentation update contract

Every material implementation change follows this path:

```text
implementation and tests
  -> related handbook chapter
  -> technical reference
  -> ADR when the decision crosses module boundaries
  -> research matrix when an experiment changes
  -> changelog
  -> documentation validation
```

The same commit should contain code, tests, and related documentation. A later
documentation-only correction is acceptable when it fixes an independent error.

The documentation ownership map in `docs/README.md` will map packages and
project concerns to their handbook and reference files.

## Automated documentation checks

Add a repository-owned documentation checker and tests. They must check:

- Every local Markdown link resolves.
- Referenced repository files exist.
- New documents contain no unfinished placeholder markers.
- Prose follows the ASCII punctuation rules.
- The documented CLI command tree matches `ddf --help`.
- The documentation ownership map covers each top-level source package.
- Required chapter headings exist.
- The changelog changes with material code or protocol updates.

External link checking will run separately. Network failures must not break the
normal local test suite.

Examples that can run on fixtures should become tests. Examples requiring full
datasets must state prerequisites and expected artifact types.

## Existing document ownership

Keep the current documents as canonical policy references:

- `docs/research-design.md` owns the frozen high-level research protocol.
- `docs/data-card.md` owns dataset purpose, fields, handling, and limits.
- `docs/model-selection.md` owns component selection rules.
- `docs/reproducibility.md` owns required run provenance.
- `docs/threat-model.md` owns attack scope, misuse, and safe claim language.

Handbook chapters teach these policies and link to them. They must not maintain
conflicting copies.

## Implementation order

Build the documentation and project in this order:

1. Add the documentation index, ownership map, validation tool, and complete
   navigation for the current platform.
2. Document the current platform from manifest creation through inference.
3. Add local MLflow and versioned configuration files with their documentation.
4. Implement and document landmarks, YuNet, aligned crops, and detector tests.
5. Run and document the view-pipeline comparison before model selection.
6. Add audio masks and strong branch candidates with controlled comparisons.
7. Run fixed-seed branch experiments and record accepted decisions.
8. Produce out-of-fold fusion, calibration, and ablation evidence.
9. Freeze the system and run locked internal and external evaluations.
10. Write findings, error analysis, model card, paper, and viva material from
    stored evidence.

No empty placeholder chapter counts as implementation. A chapter enters the
index only when it contains accurate current material or a clear planned-status
notice with useful background.

## Risks and controls

| Risk | Control |
|---|---|
| Documentation becomes too large to read | Use a learning path, summaries, exercises, and separate references. |
| Documentation becomes stale | Require same-commit updates and automated link and CLI checks. |
| AI invents reasoning or results | Require evidence links, hashes, run IDs, and explicit planned status. |
| Too many model comparisons consume time | Use sequential comparisons and fixed budgets. |
| Validation choices become hidden test tuning | Preregister decision rules and lock final test access. |
| Large artifacts fill the system drive | Use configurable data roots and record storage use. |
| Sensitive media enters tracking or Git | Log identifiers and hashes, not raw media or crops. |
| Paper and repository disagree | Generate every reported result from stored prediction artifacts. |
| Student cannot explain generated code | Require theory, code paths, exercises, and viva questions. |

## Acceptance criteria

The documentation system is complete when:

- A reader with basic Python and Git can follow the learning path.
- Every implemented subsystem has theory, shapes, code paths, tests, and
  failure explanations.
- Every top-level source package has a handbook or reference owner.
- Every major research choice has an ADR or recorded comparison decision.
- Every reported result maps to immutable runs and artifacts.
- A clean environment can run the documented smoke workflow.
- Documentation validation runs with the normal test workflow.
- The student can answer the viva questions without reading source code live.
- The final paper can be rebuilt from the traceability records.

The project itself is complete only after the locked evaluations and clean
reproduction finish. Documentation volume alone does not satisfy that gate.
