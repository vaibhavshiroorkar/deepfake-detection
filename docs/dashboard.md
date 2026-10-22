# Dashboard

The dashboard is a teaching tool. Its pages walk a clip through the pipeline a
stage at a time with models you configure, and report what each stage measured.
Verdicts for users come from the web app (`ddf-web`), which serves the trained
models with their provenance checks. The dashboard used to carry its own verdict
path, the Evidence gate, over an older frozen baseline; it was removed so there
is one place that issues a verdict.

Nothing in the dashboard trains a model or writes into `data/`.

## Running it

```powershell
uv run streamlit run src\deepfake_detection\dashboard\app.py --server.address 127.0.0.1
```

Open `http://127.0.0.1:8501`. The server binds to the local host.

The pages run without checkpoints or data, and report their empty state instead
of raising. Pages that run a model need CUDA-enabled PyTorch. See `docs/obstacles.md` for the
environment install, which must not mix the `cpu` and `cu130` extras.

## Layout

```
src/deepfake_detection/dashboard/
  app.py         shell: page registration, sidebar, the shared stylesheet
  pages/         one file per sidebar entry, each runs top to bottom
  sections/      render_x(embedded=...) bodies that some pages wrap
  lib/           page-independent logic: selection, decoding, checkpoints, MLflow
  configuration.py, evidence.py   the validation record the Experiments page reads
```

The roots every page resolves against (`PROJECT_ROOT`, `DATA_DIR`, `RUNS_DIR`,
`MLFLOW_DB`) live in `deepfake_detection/paths.py`, shared with the web app.

`pages/` holds scripts, `sections/` holds functions. A page can be handed
straight to `streamlit.testing.v1.AppTest`; a section is called from a page, and
carries an `if __name__ == "__main__":` block so it can be tested the same way.

## Navigation

`app.py` registers every page with `st.navigation(..., position="hidden")` and
then draws the sidebar itself with `st.page_link`, because the Streams hub has
to be a page and an indented section header at once.

Order, top to bottom:

| Page | What it does |
|---|---|
| Overview | The problem, the architecture diagram, what is built |
| Preprocessing | Every preprocessing step as a toggle, applied cumulatively |
| Streams | Hub: configure all three visual streams and see what fusion reads |
| Visual | One backbone, one clip, stage by stage |
| Lip-Sync | Audio against mouth motion, the trained stream on one clip |
| Emotion | Vocal affect against facial expression, the trained stream on one clip |
| Audio branch | Teaching page. States the trained branch's results; loads no checkpoint |
| Sync branch | Teaching page. States the trained branch's results; loads no checkpoint |
| Experiments | The frozen validation record, and every MLflow run beside it |
| Fusion | The trained head, its ablation, and both verdicts |
| Explainability | Per-method breakdowns, and the two views not built |
| Documentation | Long-form reference, plus links to the repository records |

`st.Page` and `st.switch_page` paths resolve against `app.py`'s own directory,
so every registration is written as `pages/<name>.py`.

## The teaching path

Preprocessing owns the clip selection and the two numbers every other page
reads: the frame count and the audio window. Streams and its three subpages
inherit that selection rather than rendering a second picker, so an uploaded
video flows straight through and a clip is chosen in exactly one place.

The per-step operations come from `deepfake_detection.preprocessing.ops`, which
is the same code the batch pipeline calls. The stream template comes from
`deepfake_detection.streams`, which gives all three backbones from one
`StreamConfig`, and `streams.introspect`, which hooks the backbone's feature
stages and returns every intermediate as plain NumPy.

## Cross-page state

Streamlit discards the `session_state` entry behind a widget that was not
rendered on the current run. A value read straight off a widget key resets to
its default the moment you navigate away, silently. Anything read on a page
other than the one whose widget wrote it therefore lives in a plain dict:

- `lib/sticky.py`: the clip settings, and each backbone's last run and trace.
- `lib/stream_pages.py`: the architecture settings, per backbone.
- `lib/selectors.py`: the dataset registry, the manifest cache, and the picker
  dialog's open flag.

## Tracking

The dashboard reads the local MLflow store and never writes to it.
`lib/mlflow_runs.py` lists experiments and runs and flattens them into the
Experiments page's comparison table. `lib/checkpoints.py` can pull a checkpoint
out of a run with `from_mlflow`, taking a run id or a `runs:/<run id>/<path>`
URI, cached under `checkpoints/_mlflow/`.

There is no Weights and Biases path. This project tracks with MLflow only.

## Datasets

`lib/datasets.py` scans `data/` on load rather than carrying a registry, so a
freshly dropped dataset appears without a code change. A directory holding a
FakeAVCeleb-style `meta_data.csv` is a raw drop, and its manifest is built in
memory by `deepfake_detection.data.meta.manifest_from_meta`. Any CSV under
`data/` carrying `clip_id`, `video_path` and `label` is a manifest, attached to
the dataset its `video_path` column points into.

An uploaded clip is written to a temp directory, never into `data/`, and carries
label `-1`, which renders as "unknown". `selectors.clip_path` is the only correct
way to resolve a selected row, because an upload's path is already absolute.

## Testing

`tests/test_dashboard_app.py` runs the shell and asserts the sidebar order. `tests/test_dashboard_lib.py` covers selection, dataset
discovery and checkpoint loading without a running app.
`tests/test_dashboard_mlflow_runs.py` covers the tracking reader. The page smoke
tests run with an empty `data/`, which is the state a fresh checkout is in.

## Fusion and Explainability

Fusion reads the trained model in
`runs/program-20260906`: the classifier and its provenance hashes, ROC-AUC on
the held-out in-domain partition and on DFDC, and the ablation over every branch
subset. The ablation leads, because objective 3 asks whether combining beats the
parts and the answer depends on the partition: in-domain the combination wins,
on DFDC it does not, and the page reports both verdicts rather than the flattering
one.

Explainability carries the one specified view with data behind it, per-method
and per-manipulation-type accuracy with subgroup coverage, and names the two
that do not exist. Grad-CAM needs a backward pass the dashboard does not run,
and embedding shift needs stream embeddings on matched pairs that no run has
exported. Both are listed as not built rather than stubbed.

Neither page fits or runs a model. Every figure is read from a run directory, so
it traces to a file and a checkpoint hash.
