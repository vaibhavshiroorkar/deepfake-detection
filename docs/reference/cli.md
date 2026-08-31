# CLI reference

## Prerequisites

Install Python 3.11 through 3.13, `uv`, and the project dependencies. Install
FFmpeg and the required optional extras before running media, training, or
inference commands.

In PowerShell, run every command through `uv run ddf`:

```powershell
uv run ddf --help
```

## Generated commands

<!-- BEGIN GENERATED COMMANDS -->
- `ddf cache build`
- `ddf detector compare`
- `ddf detector fetch-yunet`
- `ddf detector run`
- `ddf detector sample`
- `ddf detector validate-annotations`
- `ddf evaluate`
- `ddf features export`
- `ddf features score`
- `ddf manifest build`
- `ddf predict`
- `ddf run`
- `ddf smoke`
- `ddf split build`
- `ddf split crossfit`
- `ddf split method-holdout`
- `ddf threshold`
- `ddf train audio`
- `ddf train fusion`
- `ddf train sync`
- `ddf train visual`
<!-- END GENERATED COMMANDS -->

The command block comes from `build_parser()`. Run `uv run ddf-docs` after a
parser change to detect drift.

## Command help

`ddf run` loads layered configuration files and executes the configured command
from the selected project root. Run `uv run ddf run --help`.

`ddf smoke` runs a deterministic CPU fusion fixture. Its metrics are software
fixture evidence only. Run `uv run ddf smoke --help`.

`ddf manifest build` normalizes source metadata and writes a manifest audit.
Run `uv run ddf manifest build --help`.

`ddf split build` creates source-disjoint train, validation, and test splits.
Run `uv run ddf split build --help`.

`ddf split crossfit` creates source-grouped folds for out-of-fold features.
Run `uv run ddf split crossfit --help`.

`ddf split method-holdout` writes a protocol that holds out selected methods.
Run `uv run ddf split method-holdout --help`.

`ddf cache build` prepares shared audio-video views and records failed clips.
Run `uv run ddf cache build --help`.

`ddf detector fetch-yunet` downloads the pinned YuNet asset and verifies its
size and SHA-256. The model stays local and is never an MLflow artifact.

`ddf detector sample` creates a deterministic training-only review sample and
local PNG review images. It verifies the frozen split directory against its
expected hash and samples only identity-strict training rows. Every source and
target identity must be owned by training. The default is 625 frames from 125
clips. This leaves at least 500 comparison frames from 100 clips after the 20
percent calibration split. It marks at least 10 percent for independent second
review.

`ddf detector validate-annotations` writes an aggregate audit. It returns code
`2` until every review, second review, and required adjudication passes.

`ddf detector run` validates the audit before inference. It calibrates the
candidate threshold on source-disjoint calibration identities, evaluates the
comparison identities, and writes a path-free prediction JSONL plus an
aggregate report. A research report requires a clean worktree, the current
`uv.lock` byte hash, and a source run ID. Tracking supplies the run ID when it is
enabled. Otherwise pass `--source-run-id`. MTCNN provenance comes from its
loaded state. Do not pass an expected model hash for MTCNN. YuNet still requires
a matching asset hash.

`ddf detector compare` requires exactly one MTCNN report and one YuNet report
for research evidence. It reads each report once, then parses and hashes that
same immutable byte buffer. The saved decision therefore identifies the exact
bytes used for selection. It rejects unpaired environments and candidates with
no tracked frames. Exact speed ties remain undecided because no strict
downstream evidence artifact exists. Fixture comparisons remain flexible and
cannot produce a real selection.

Use these commands directly or place the same command and arguments in a YAML
file for `ddf run`. With local tracking enabled, only aggregate detector
reports and path-free predictions become detector artifacts. Review images,
annotations, source media, crops, and model binaries remain local inputs.
Before upload, the prediction byte hash and strictly parsed report artifact must
match the supplied benchmark report.

Run the workflow in order:

```powershell
uv run ddf detector fetch-yunet --report runs/detector/yunet-asset.json
uv run ddf detector sample --split-dir data/private/splits --expected-split-hash <sha256> --dataset-root data/private --dataset training --output data/private/detector-review/sample.jsonl --review-dir data/private/detector-review/images --report runs/detector/sample-report.json
uv run ddf detector validate-annotations --sample data/private/detector-review/sample.jsonl --annotations data/private/detector-review/annotations.jsonl --report runs/detector/annotation-audit.json
uv run ddf detector run --sample data/private/detector-review/sample.jsonl --annotations data/private/detector-review/annotations.jsonl --split-dir data/private/splits --dataset-root data/private --dataset training --predictions runs/detector/mtcnn-predictions.jsonl --report runs/detector/mtcnn-report.json --detector mtcnn --detector-revision <revision> --source-run-id <run-id>
uv run ddf detector run --sample data/private/detector-review/sample.jsonl --annotations data/private/detector-review/annotations.jsonl --split-dir data/private/splits --dataset-root data/private --dataset training --predictions runs/detector/yunet-predictions.jsonl --report runs/detector/yunet-report.json --detector yunet --detector-revision opencv-zoo-47534e27 --model-path models/face_detection_yunet_2026may.onnx --expected-model-hash ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0 --source-run-id <run-id>
uv run ddf detector compare --reports runs/detector/mtcnn-report.json runs/detector/yunet-report.json --output runs/detector/decision.json
```

These commands prepare the comparison. They do not claim that review or the
real comparison has happened.

`ddf train visual` trains the visual cue branch.
Run `uv run ddf train visual --help`.

`ddf train audio` trains the audio cue branch.
Run `uv run ddf train audio --help`.

`ddf train sync` trains the mouth-audio alignment branch.
Run `uv run ddf train sync --help`.

`ddf train fusion` trains calibrated late fusion from out-of-fold features.
Run `uv run ddf train fusion --help`.

`ddf features export` writes branch outputs to a feature store.
Run `uv run ddf features export --help`.

`ddf features score` scores stored features with a fusion model.
Run `uv run ddf features score --help`.

`ddf threshold` selects a validation decision threshold.
Run `uv run ddf threshold --help`.

`ddf evaluate` writes metrics, intervals, method reports, and subgroup reports.
Run `uv run ddf evaluate --help`.

`ddf predict` runs all required artifacts on one video.
Run `uv run ddf predict --help`.

## Exit codes

Code `0` means the command finished successfully. Code `1` is used by
`ddf-docs` when checks find documentation issues. An unhandled CLI runtime
error also normally ends with code `1`.

Code `2` means partial output or invalid CLI input. `ddf cache build` returns
`2` when one or more clips fail after writing its successful outputs.
`ddf features export` returns `2` when one or more rows lack required evidence.
`ddf detector validate-annotations` returns `2` for an incomplete or invalid
audit after writing its aggregate report.
Argparse also uses `2` for invalid command syntax.

## Security

Only load joblib files from trusted sources. Joblib files can execute code
during loading.
