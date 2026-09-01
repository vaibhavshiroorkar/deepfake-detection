# Dataset, MNW, and Research History Design

Date: 2026-09-02

## Goal

Create one safe local data root with four datasets. Replace Deepfake-Eval with
MNW as the locked external benchmark. Keep a durable research trail through
MLflow and versioned evidence records.

## Current state

The repository has two untracked datasets inside the Python package:
`Celeb-DF-v2` and `FakeAVCeleb_v1.2`. The same directory also contains an
untracked FaceForensics++ downloader. Raw media does not belong inside package
source code.

The project uses local MLflow through `mlflow.db` and `mlartifacts/`. It does
not use W&B. The existing ten finished MLflow runs are repeated CPU smoke runs
over a 32-row fixture. They prove that the software path works. They do not
provide research model results.

No research model is fully trained. There are no visual, audio, or sync
checkpoints. There are also no complete three-seed comparisons, out-of-fold
features, fusion results, or locked evaluations.

## Dataset layout

Use the ignored top-level `data/` directory. It will contain exactly these four
dataset directories:

```text
data/
  Celeb-DF-v2/
  FakeAVCeleb_v1.2/
  FaceForensics++/
  MNW/
```

Keep `data/Celeb-DF-v2.zip` beside the directories unless disk pressure makes
the duplicate archive a problem. It is not a fifth dataset directory.

Keep `src/deepfake_detection/data/download.py` as the FaceForensics++
acquisition tool. Dataset code stays in the package. Dataset payloads stay in
the top-level ignored directory.

## FaceForensics++ acquisition

The user confirmed agreement with the FaceForensics++ terms. Run the supplied
downloader for every dataset, with video payloads and `c23` compression. The
target is `data/FaceForensics++`.

The downloader is resumable at the file level. Existing completed files are
skipped. A failed transfer must leave completed files in place so a later run
can resume.

Do not download raw videos, masks, or generator model files in this task.

## MNW acquisition and use

Clone `https://github.com/microsoft/MNW` into `data/MNW` with Git LFS support.
First clone repository metadata without LFS smudging. Calculate the declared
LFS payload size and compare it with free disk space. Keep at least 50 GB free
after acquisition.

Pull the full benchmark when it fits. If it does not fit, stop before the LFS
pull and report the required size. Do not silently install a partial benchmark.

MNW is an evaluation-only dataset. It must never appear in a training or model
selection manifest. The project must also state its non-commercial restriction.
Replace every live Deepfake-Eval protocol reference with MNW and its official
repository or paper.

## Research tracking

Keep local MLflow as the run tracker. Do not add W&B in this change.

Use MLflow for each attempt, including failures. Each research run records its
configuration, Git state, dataset and split hashes, seed, epoch metrics,
runtime, checkpoint hash, predictions, and failure details. Smoke runs retain
the `software_fixture_only` scope and cannot support model selection.

Add the missing research documents already named by the documentation index:

- `docs/research/experiment-matrix.md` defines planned comparisons and fixed
  seeds.
- `docs/research/result-traceability.md` defines the accepted result registry.
- `docs/research/findings.md` remains empty until evidence is accepted.

MLflow stores full run details. Git stores only accepted result IDs, hashes,
analysis commands, and paper locations. Raw media, local databases,
checkpoints, and generated run outputs remain ignored.

## GPU training constraint

All research model training must use the NVIDIA GeForce RTX 5070 Ti with
16,303 MiB of VRAM. A configured research run must fail before training if CUDA
is unavailable. CPU remains allowed for unit tests and labeled smoke fixtures.

Use mixed precision, cached preprocessing, and gradient accumulation where the
model recipe supports them. Track the GPU model, CUDA version, peak memory,
samples per second, and elapsed time in MLflow.

Before the full experiment matrix starts, run one short throughput benchmark
for each branch. Use its measured samples per second and the final manifest row
count to produce a run-time forecast.

## Time estimate

Data acquisition time depends on the remote payload sizes and observed network
speed. Report a measured forecast after the FaceForensics++ file list and MNW
LFS metadata are available.

The current planning estimate for the full model matrix is 8 to 24 days of
continuous work on one RTX 5070 Ti. This includes cached preprocessing, seven
branch candidates with three seeds, fusion, ablations, and locked evaluation.
It excludes manual detector review. The range is intentionally broad because
the final manifest size and branch throughput have not been measured.

No full training starts as part of this dataset task. Training starts only
after data audits, split freezes, detector selection, and the GPU throughput
benchmarks pass.

## Documentation changes

Update the data card, research design, roadmap, handbook links, and README
examples. Commands must use paths under the top-level `data/` directory.
Protocol text must distinguish development datasets from the locked MNW
benchmark.

Record this protocol change in `CHANGELOG.md`. Do not claim that model training
or evaluation is complete.

## Verification

Verify all of the following before reporting completion:

1. The top-level data root has exactly the four named dataset directories.
2. The two existing datasets retain their expected media counts and metadata.
3. FaceForensics++ has every requested `c23` video family and no temporary
   transfer files.
4. MNW is at a recorded commit and has no unresolved LFS pointers.
5. No live Deepfake-Eval reference remains.
6. Documentation links and tests pass.
7. MLflow smoke history remains readable and clearly scoped as fixture-only.
8. No research training completion claim appears without checkpoints and run
   evidence.

## Failure handling

Do not delete completed downloads after a transfer failure. Stop if disk space
would fall below 50 GB. Keep the old dataset location until each move is
verified, then remove only the empty source directories. Never use MNW data in
a training or validation command.
