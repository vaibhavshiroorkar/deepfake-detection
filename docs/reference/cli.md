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
- `ddf evaluate`
- `ddf features export`
- `ddf features score`
- `ddf manifest build`
- `ddf predict`
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
Argparse also uses `2` for invalid command syntax.

## Security

Only load joblib files from trusted sources. Joblib files can execute code
during loading.
