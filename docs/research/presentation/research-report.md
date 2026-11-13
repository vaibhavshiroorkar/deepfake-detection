# Deepfake detection research report

## Scope and authors

This report follows the project from the first prototype on 2026-07-04 through the repository integration on 2026-11-12.
The research asks whether multiple detection cues improve performance on unfamiliar manipulated media.
The evidence shows strong performance within some datasets and substantial limits under distribution shift.
Several root-study results require corrected reruns. They cannot establish the corrected system's performance.

Vaibhav Shiroorkar, Shivansh Gupta and Adithya Jithesh share equal credit, one third each, across all work and deliverables.
Author order does not imply rank. Dr. Tejashree Phatak is the project guide.
The department is Electronics and Computer Engineering at K. J. Somaiya School of Engineering, Somaiya Vidyavihar University.

The repository contains two research lines. The root studies cue-specific fusion and generalization.
The team project studies frozen encoders and separate findings for image, face-video and speech analysis.
Their protocols and scores remain separate throughout this report.

Evidence: AUTHORS.md; docs/HANDBOOK.md; docs/IDENTITY.md at tag team/submission-final; docs/research/result-traceability.md.

## Research question and method

The root question is whether visual artifacts, voice spoofing and mouth-audio alignment provide complementary evidence on unfamiliar data.
An in-domain score answers how well a model ranks examples from a familiar distribution.
A cross-corpus score tests a different distribution. A leave-one-generator-out evaluation tests a held-out generation family.
These tests answer different questions and cannot be substituted for one another.

The root pipeline preprocesses video and audio, trains cue-specific branches and combines held-out branch predictions.
Visual training uses a cue-specific label. A genuine video with fake audio remains visually genuine.
Fusion uses the clip-level manipulation label. Branch comparisons must use a common target when measuring fusion gains.

The team system uses frozen encoders, trained heads and validation-selected thresholds.
Its platform reports separate module findings. It does not expose one validated multimodal verdict.
Missing evidence and experimental modules can produce an inconclusive result.

ROC-AUC measures ranking across thresholds. It does not measure the false-alarm rate at the threshold used in a demonstration.
Report the threshold, false positives, missed manipulations and coverage alongside ranking metrics.

Evidence: docs/research-design.md; docs/research/architecture.md; `docs/PROTOCOL.md` at tag `team/submission-final`; results/legacy-team/README.md.

## Development from July through September

The July prototype planned three visual streams and two cross-modal streams for lip-sync and emotion.
Feature fusion and embedding cross-attention replaced simpler score combinations and text-based comparison in the design.
Early August work added training and resume support, with changes among Xception and DINO encoders.
This phase established the implementation path. It did not establish reliable generalization.

The August rebuild made evaluation part of the architecture.
It introduced source-disjoint splits, identity stress tests, preprocessing hashes, run metadata and smoke checks.
The handbook explained the pipeline stages and their assumptions.

September work trained the branches and fusion variants and examined failures.
Checkpoint selection, seed variation, dataset shortcuts and changing capture conditions affected the conclusions.
Some striking historical scores were later withdrawn. Keep their measurements and corrections together when telling this history.

The lesson from this phase is procedural: a trained model and a high score are insufficient evidence.
The evaluation must establish which cue is being measured and which information was available during model selection.

Evidence: docs/HANDBOOK.md section 12; docs/research/findings.md; docs/research/result-traceability.md.

## October methods and sealed evaluation

The root study added seed comparisons, encoder challengers, shortcut probes and a research paper draft.
The team project started a separate evaluation protocol in late October.
It recorded hypotheses and success criteria, selected models on development data and froze the system before sealed evaluation.
Its evaluation ledger records the sealed reads. Statistical intervals use source groups rather than treating related samples as independent.

The team progression included an MVP and iterations on representation, training choices, calibration and integration.
An rPPG feasibility gate failed. A failed gate is a result about that setting, not evidence that all physiological detection is impossible.
The platform kept experimental image and generated-video checks distinct from validated claims.

Three confirmatory comparisons required a multiple-comparison correction.
The Xception comparison supports non-inferiority under the declared margin.
The apparent superiority at 95 percent does not survive the corrected 98.33 percent interval.

The two codebases provide related evidence about shortcut risks.
They are not a controlled replication because their models, data partitions and target labels differ.

Evidence: `docs/ITERATIONS.md` at tag `team/submission-final`; `docs/PROTOCOL.md` at tag `team/submission-final`; results/legacy-team/EXPERIMENTS.md E-006 and E-014; `submission/REHEARSAL.md` at tag `team/submission-final`.

## Sealed results and their limits

The following team results come from E-014. AUROC is on a zero-to-one scale.
These numbers belong to the team protocol and must not be presented as root-study results.

| Evaluation | MVP | Final | Interpretation |
|---|---|---|---|
| Video on the Celeb-DF family | 0.6238 | 0.8218 | Improvement within this sealed protocol |
| Speech on In-the-Wild | 0.7516 | 0.9148 | Improvement within this sealed protocol |

The video difference against the re-run Xception baseline was +0.0176.
Its 95 percent interval was [+0.0025, +0.0327].
Its corrected 98.33 percent interval was [-0.0011, +0.0355], which includes zero.
Report non-inferiority under the declared 0.0100 margin, not corrected evidence of superiority.
The baseline used 224-pixel inputs, which also limits interpretation.

Post hoc false-positive rates at the frozen face-video thresholds were 0.1098 on the Celeb-DF family and 0.1405 on DFD.
These E-032 rates are not confirmatory endpoints. They describe the risk of calling genuine media manipulated.
Published benchmark values from different subsets or training protocols provide context, not a fair ranking of systems.

Evidence: results/legacy-team/EXPERIMENTS.md E-014 and E-032; `submission/REHEARSAL.md` at tag `team/submission-final`; `docs/POST_FREEZE_BENCHMARK_COMPARISON.md` at tag `team/submission-final`.

## Corrections that change the conclusions

The root study's audit found three defects that affect research claims.
First, outer-fold labels selected checkpoints used to produce out-of-fold features.
Fusion needs a nested selection process before its historical comparisons can support a conclusion.

Second, an explicit zero-start LAV-DF window could shift into a manipulated span.
The historical 0.9969 matched-window score requires corrected windows and a rerun.
It cannot support a claim that the model demonstrated audio-visual correspondence.

Third, Perception Encoder and CLIP features used incorrect input normalization.
The historical DF26 score of 0.7527 does not establish corrected performance.
Re-extraction and evaluation remain necessary.

Other interpretations also need restraint. Attention mass near chance does not prove the model ignores timing.
Audio swaps and timing shifts would test that mechanism more directly.
Two classifiers being wrong on the same examples does not prove that their continuous features contain no useful information.
Those error counts describe the evaluated decisions, not an upper bound on every possible fusion model.

Evidence: docs/research/result-traceability.md; docs/reproducibility.md; docs/handoff.md; docs/research/contributions.md.

## Post-freeze exploratory evidence

E-033 and E-034 evaluated previously unread samples from distributions already examined during development.
They are exploratory post-freeze evaluations. They are not new confirmatory tests or fully independent benchmarks.
The old PLANNED wording in their registration blocks preserves the original ordering record; results are appended below it.

| Evaluation and target | Samples | AUROC | 95 percent interval |
|---|---|---|---|
| E-033 LAV-DF video head, any modality changed | 18,383 | 0.2954 | [0.2890, 0.3019] |
| E-033 LAV-DF speech head, any modality changed | 18,383 | 0.6448 | [0.6383, 0.6514] |
| E-034 Community Forensics-Eval, pooled image label | 7,209 | 0.7909 | [0.7803, 0.8014] |
| E-034 PNG-only diagnostic | 368 | 0.5693 | [0.5082, 0.6279] |

Against modality-specific labels, E-033 G4 gives video AUROC 0.3803 and speech AUROC 0.6447 on 10,599 clips.
The target label and sample pool must accompany each score. These values cannot be mixed with the full-pool row above.
The image diagnostic uses a smaller, different population. Its lower score is an association, not proof that format caused the difference.
Nested sample-size groups overlap and are not independent replications.

Evidence: results/legacy-team/EXPERIMENTS.md E-033 and E-034; `docs/POST_FREEZE_EVALUATION_PROTOCOL.md` at tag `team/submission-final`; `docs/POST_FREEZE_BENCHMARK_COMPARISON.md` at tag `team/submission-final`.

## November work and remaining experiments

The November AV-HuBERT work adds mouth crops, encoder support and lip-audio agreement code.
E-035 preregisters a zero-shot baseline on LAV-DF windows.
E-036 preregisters real-only and supervised heads on frozen AV-HuBERT features, with three seeds per trained variant.
Both entries still have empty result slots. No accuracy or improvement is claimed for them.

The next steps are to verify local data and weights, build the preregistered windows and exclude sealed sources.
Then run the registered checks and record the results before choosing the next T-AV variant.
The root work separately needs corrected cross-fitting, corrected LAV-DF windows and corrected PE/CLIP preprocessing.
No fresh sealed test should be used for presentation preparation or threshold adjustment.

The 2026-11-12 integration preserved both histories and linked the early prototype history.
The present report brings the evidence into one narrative while retaining separate protocols.
Equal credit applies to all three student authors across the entire project.

Evidence: results/legacy-team/EXPERIMENTS.md E-035 and E-036; src/deepfake_detection/avsync/; docs/handoff.md; docs/HANDBOOK.md.

## Demonstration and reproducibility

Use the team setup guide for the modular platform and the root README for the research interface.
Run each from its own project directory with its own environment.
Confirm that required local checkpoints and data exist before promising a live run.
A fixture or mock adapter demonstrates software behavior only and must remain visibly labelled.

During the demonstration, explain the input, available modalities, module output and missing evidence.
Show the threshold and limitations. Do not describe a score as the probability that arbitrary internet media is fake.
If real checkpoints are unavailable, present the documented workflow and recorded results without inventing a live inference.

To reproduce a research number, follow its experiment or registry entry to the command, manifest, checkpoint and output hashes.
Some artifacts are excluded from Git because they are large or restricted by dataset and weight licences.
This documentation update did not rerun training, inspect every private artifact or certify the deployed demo.

The presentation's defensible conclusion is that evaluation controls expose failures that aggregate accuracy can hide.
The research contribution is the documented testing and correction process, subject to the listed reruns and limitations.
No claim of state-of-the-art performance, deployment readiness or literature priority follows from this report.

Evidence: README.md; docs/reproducibility.md; docs/model-card.md; `docs/SETUP.md` at tag `team/submission-final`; `docs/FINAL_DEMO_RUNBOOK.md` at tag `team/submission-final`; `docs/RESEARCH_INTEGRITY.md` at tag `team/submission-final`.
