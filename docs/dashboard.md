# Dashboard

The dashboard is two things sharing one shell. The teaching pages walk a clip
through the pipeline a stage at a time with models you configure. The Evidence
gate runs the one frozen baseline that has provenance behind it. Both are in the
same sidebar so the difference between "what a stage does" and "what this
project can currently claim" is visible rather than implied.

Nothing in the dashboard trains a model or writes into `data/`.

## Running it

```powershell
uv run streamlit run src\deepfake_detection\dashboard\app.py --server.address 127.0.0.1
```

Open `http://127.0.0.1:8501`. The server binds to the local host.

The gate's preprocessing and prediction steps need CUDA-enabled PyTorch and the
ignored `runs\initial-20260902` artifacts. The teaching pages run without either,
and report their empty state instead of raising. See `docs/obstacles.md` for the
environment install, which must not mix the `cpu` and `cu130` extras.

## Layout

```
src/deepfake_detection/dashboard/
  app.py         shell: page registration, sidebar, the shared stylesheet
  paths.py       PROJECT_ROOT, DATA_DIR, CHECKPOINT_DIR, RUNS_DIR, MLFLOW_DB
  pages/         one file per sidebar entry, each runs top to bottom
  sections/      reusable render_x(embedded=...) bodies the gate composes
  lib/           page-independent logic: selection, decoding, checkpoints, MLflow
  configuration.py, evidence.py, runtime.py, state.py, view_model.py, workflow.py
```

`pages/` holds scripts, `sections/` holds functions. A page can be handed
straight to `streamlit.testing.v1.AppTest`; a section is called from a page, and
carries an `if __name__ == "__main__":` block so it can be tested the same way.

## Navigation

`app.py` registers every page with `st.navigation(..., position="hidden")` and
then draws the sidebar itself with `st.page_link`. Streamlit's built-in
navigation has no disabled entry, and Fusion and Explainability have to appear
in their pipeline positions, dimmed and unclickable, rather than vanish. Their
routes stay registered, so a direct visit lands on a body saying what will land
there and what unlocks it.

Order, top to bottom:

| Page | What it does |
|---|---|
| Overview | The problem, the architecture diagram, what is built |
| Evidence gate | The four-step workflow over the frozen baseline |
| Preprocessing | Every preprocessing step as a toggle, applied cumulatively |
| Streams | Hub: configure all three visual streams and see what fusion reads |
| Visual | One backbone, one clip, stage by stage |
| Lip-Sync | Audio against mouth motion. Stage 4, not built |
| Emotion | Vocal affect against facial expression. Stage 5, not built |
| Audio branch | Prototype. Trained on a fixture, not evaluated |
| Sync branch | Prototype. Trained on a fixture, not evaluated |
| Experiments | The frozen validation record, and every MLflow run beside it |
| Fusion | Locked until Stage 6 |
| Explainability | Locked until Stage 10 |
| Documentation | Long-form reference, plus links to the repository records |

`st.Page` and `st.switch_page` paths resolve against `app.py`'s own directory,
so every registration is written as `pages/<name>.py`.

## The Evidence gate

`pages/gate.py` composes four sections in one page: video input, preprocessing,
visual model, prediction. Each is an expander showing its own
waiting/ready/complete state from `workflow.py`, so the order is visible without
navigating.

Derived state is keyed by the upload's SHA-256 in `state.py`, stored as
`(clip_sha256, value)` tuples. A new upload therefore invalidates preprocessing
and the prediction rather than leaving a stale result on screen.

The gate is the only part of the dashboard bound to the frozen provenance in
`configuration.py`. It loads the visual engine only after the checkpoint hash,
run id, split hash, commit, seed and preprocessing hash all match. A mismatch is
reported; nothing is substituted. `evidence.py` applies the same rule to the
saved validation record.

The sidebar shows those provenance values under "Frozen baseline", with a note
that they bind the gate only. The teaching pages build their own models and load
whatever checkpoint you pick.

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

`tests/test_dashboard_app.py` runs the shell and asserts the sidebar order and
the locked entries. `tests/test_dashboard_lib.py` covers selection, dataset
discovery and checkpoint loading without a running app.
`tests/test_dashboard_mlflow_runs.py` covers the tracking reader. The page smoke
tests run with an empty `data/`, which is the state a fresh checkout is in.
