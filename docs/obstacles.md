# Obstacles

Constraints and traps that cost real time in this repository. Each entry names
the constraint, what breaks when you ignore it, and what to do instead. This is
a reference to read before touching an area, not a record of when anything
happened. Git history holds the chronology.

## Environment

### The `cpu` and `cu130` extras conflict

`pyproject.toml` declares them as a `[tool.uv] conflicts` pair. Installing both
fails, and installing `cpu` silently gives you a Torch build where
`torch.cuda.is_available()` is `False` on a machine with a working GPU.

Nothing in the code says "CUDA is missing because you installed the wrong
extra". You get a `RuntimeError` from a `.cuda()` call, or the dashboard's
prediction step reports that CUDA is unavailable on this server.

Install the full research environment as one command:

```powershell
uv sync --extra cu130 --extra ml --extra media --extra dashboard --extra tracking --group dev
```

Dropping `--extra tracking` leaves MLflow out, so `ddf run` cannot record and
the Experiments page cannot read the store. Dropping `--extra media` leaves out
`av`, `librosa` and `opencv-python`, which the preprocessing ops need.

### A running server keeps the torch it imported

Upgrading the environment under a live Streamlit server changes nothing for that
process. A dashboard started before `uv sync --extra cu130` keeps the CPU-only
torch it imported and fails with `AssertionError: Torch not compiled with CUDA
enabled` from inside a vendored model, while a fresh shell in the same checkout
reports `torch.cuda.is_available()` as True. The fix is to restart the server,
not to reinstall.

`runtime.require_cuda` runs before anything touches a device and names which of
the two states it is, so this reaches the page as a sentence rather than a stack
trace.

### Run the dashboard from the checkout you edited

`.worktrees/multipage-teaching-dashboard` holds an older branch with its own
`dashboard/` package, including a `pages/preprocessing.py` that does not exist
on `main`. A traceback whose paths run through `.worktrees/` is that checkout,
not this one. `git worktree list` says which is which.

### Branch training refuses a non-CUDA device

`require_research_cuda` gates every branch training command. This is
deliberate: a CPU run would produce a checkpoint with different numerics from
every recorded run, and there is no way to tell the two apart afterwards. Fix
the environment rather than the gate.

## Repository shape

### `main` and `old` share no history

`git merge-base main origin/old` exits 1. The two branches are unrelated
implementations of the same project, so no merge, rebase or cherry-pick applies
between them. Moving code across is a port: rewrite the imports, drop the
`sys.path` bootstrap, and re-run the tests.

The `old` branch is a flat script repository with `[tool.uv] package = false`,
where imports resolve through a per-file `sys.path.insert` of the repository
root and an intentionally empty root `conftest.py`. `main` is an installed
src-layout package. Code copied across without that rewrite imports cleanly on
the author's machine and fails everywhere else.

### Every Python file under `src/` must be tracked by Git

`tests/test_repository_integrity.py` compares `git ls-files -- src` against the
files on disk and fails on anything untracked. A new module therefore has to be
`git add`ed before the suite passes, even while the work is still in progress.

### The CLI reference is drift-checked

`docs/reference/cli.md` carries a generated block that `uv run ddf-docs`
compares against the live parser. Adding or renaming a CLI command without
regenerating that block fails the check.

## Preprocessing

### The detector is not a pipeline version

`PIPELINE_VERSION` is 4 and is bumped only when the cached crop or audio pixels
change. Which detector ran varies per run rather than moving forward, so it is
stamped beside the number as `4:mtcnn` in the cache's version file. Bumping the
version for a detector swap invalidates every cached clip for no reason.

### Landmark order is image-left first, for both detectors

Left eye, right eye, nose, mouth-left, mouth-right, with image-left first.
YuNet's own documentation calls its first point the right eye, but that is the
subject's right, which is the image-left point facenet-pytorch calls the left
eye. Same physical order, opposite naming convention. Getting this wrong mirrors
every face without changing a single shape, so no assertion catches it.

### The confidence threshold lives in one place

`faces.detect` applies `conf_thresh`; the detectors themselves are built with a
floor of 0.05 and gate nothing. A backend threshold would be a second, invisible
threshold that the dashboard slider could not reach.

### YuNet bakes the input size into its anchor grid

`cv2.FaceDetectorYN` returns plausible-looking nonsense boxes if it is fed a
frame shape it was not told about. The wrapper calls `setInputSize` whenever the
shape changes. YuNet also reads BGR, while the ops interface is RGB.

### YuNet weights are not in the repository

Fetch them with `uv run ddf detector fetch-yunet`, which writes
`models/face_detection_yunet_2026may.onnx` and verifies size and SHA-256.
`preprocessing.ops.detectors.YuNetDetector` raises `FileNotFoundError` naming
that command when the file is absent. MTCNN needs no fetch and is the default.

### Five-point alignment was removed, and carried two traps

The face crop used to be a similarity warp onto a canonical ArcFace template, so
eyes and mouth landed on the same pixels in every frame. It is gone, and
`PIPELINE_VERSION` went to 4 because the cached crop pixels changed. If it comes
back, these are the two things that went wrong the first time:

- **Padding must be black, not reflected.** A warped canvas reaches past the
  frame edge, and OpenCV's default reflect border mirrored a second face into
  every crop.
- **The template inset shrinks the face.** The ArcFace template leaves margin
  around the face, so applying it unchanged fed the backbone a smaller face than
  the bbox crop did, and the two are not comparable.

The current crop is a margin-padded bounding box clamped to the frame, so it can
never introduce padding of its own. The cost is that head roll and off-centre
framing survive into the tensor.

### Frame and audio sampling share one set of timestamps

`sample_timestamps` produces the timestamps for both modalities, and
`start_offset` pushes them past FakeAVCeleb's leading-silence shortcut. Sampling
the two independently makes every clip look desynchronised, so the cross-modal
streams would measure the pipeline rather than the forgery.

### The synchronisation window is pinned near the start of a clip

`Preprocessor.prepare` places it at `content_start + sync_max_offset_seconds`,
which is fine for a corpus that manipulates whole clips and wrong for one that
hides a short forgery somewhere inside a longer recording. Measured on LAV-DF,
**56,582 of 99,873 forgeries fall entirely outside that window**, so 57 percent
of fake clips would train the model on genuinely unmanipulated video carrying a
fake label.

`ClipRecord.sync_start_sec` overrides it per clip. The field enters
`cache_fingerprint` **only when non-zero**, so every entry cached before the
option existed keeps its key. A key present on every clip would have invalidated
the whole cache to record a value that is zero almost everywhere.

### The duplicate quarantine keys on the window, not just the path

`load_manifest` drops any `video_path` whose rows disagree on manipulation type
or method, which is what catches FakeAVCeleb's roughly 22 duplicate listings.
Matched pairs are two rows on one file with different labels **by design**, so
the guard keys on `(video_path, sync_start_sec)`. Two windows are distinct
clips; two rows on the same window with conflicting labels still quarantine.
Group on the path alone and both halves of every pair disappear silently into
`quarantined_paths`.

### Celeb-DF-v2 and MNW have no audio at all

The 638 `missing_audio` blockers in the full cache audit are 518 Celeb-DF clips
plus 120 MNW clips, not a detector failure. Neither corpus can evaluate an
audio, sync, or cross-modal stream. Check `audio_present` before assuming a
dataset can carry an audiovisual claim.

## Models

### DINOv3 must pool over the class token

`timm` builds `fc_norm` in place of `norm` whenever `global_pool="avg"`, and
DINOv3's released weights carry `norm`, so a strict pretrained load fails
outright. `dinov3_config()` sets `global_pool="token"` for that reason, not as a
tuning choice. `stream_pages.GLOBAL_POOL` pins the same value per backbone,
because building a stream with the wrong one still loads every tensor (the
shapes match either way) and then quietly computes a different vector than the
one that was trained.

### DINOv3 needs an explicit `img_size`

Its pretrained config is 256 pixels and this pipeline feeds 224. Without
`img_size` the positional embedding is sized for an input that never arrives.
CNN backbones raise `TypeError` on the keyword, which `_create_backbone` uses as
the signal to build them plainly rather than keeping a list of which backbones
are transformers.

### `legacy_xception` does not support gradient checkpointing

`timm` exposes `set_grad_checkpointing` on it, and the method asserts on enable.
`VisualStream` records the failure in `self.grad_checkpointing` instead of
raising: losing it costs memory, not correctness.

### Gradient checkpointing hides the activations from the tracer

A checkpointed `timm` backbone runs through a flattened functional segment, so
the stage modules are never called as modules and their forward hooks never
fire. `trace_visual_stream` turns checkpointing off for the pass and back on
afterwards. Without that the trace comes back empty with no error.

### A ViT stage output is not a channel map

A CNN stage emits `[C, H, W]` per frame; a ViT stage emits `[prefix + P, D]`.
`StageTrace.kind` says which, and callers must branch: a channel grid is
meaningless for tokens and a patch grid is meaningless for channels.
`token_grid` infers the prefix by finding the smallest one that leaves a perfect
square, so 201 tokens decompose as 5 + 14x14, not 65 + 8x8.

### Whisper's encoder does not take a waveform

Every other audio path here feeds `Wav2Vec2Model` a raw `[batch, samples]`
waveform, so it is natural to assume Whisper is a drop-in swap. It is not. Its
encoder takes `[batch, 80, 3000]` log-mel and rejects anything else outright:

```
ValueError: Whisper expects the mel input features to be of length 3000,
but found 32000.
```

3000 frames is exactly 30 seconds and the positional embeddings are sized for
it. The synchronisation window is 2 seconds, so using Whisper there means 28
seconds of padding and 100 useful tokens out of 1500. `stream_spec.py` names
Whisper for the lip-sync stream; that is a target, not something that works
today.

### fairseq cannot be installed here

AV-HuBERT's released implementation needs fairseq, which pins PyTorch to 1.08,
1.13 or 2.0. This environment runs torch 2.12.1+cu130. Installing fairseq takes
the working CUDA stack with it, so AV-HuBERT needs either a separate environment
that pre-extracts features offline or a community port. Do not pip install it
into the project venv to "see if it works".

### Pooling a cross-attention output erases the attention

A cross-modal stream that returns `attended.mean(dim=1)` barely trains. Attention
is near-uniform at initialisation, so every attended step collapses onto the mean
value token and averaging over time removes what remains. Measured on an
untrained stream, shifting the audio by 320 ms moved the embedding by 2e-7,
which is float noise rather than a gradient.

Return the residual, `query - attended`, instead. That is what
`stream_spec.py` means by a mismatch vector, and it moved the same embedding by
0.072 against a scale of 1.0.

### An untrained CNN is nearly input-blind

A randomly initialised ResNet in eval mode produces almost the same pooled
feature for every frame: measured token spread across time was 0.011 against the
audio encoder's 0.587, and two differently seeded noise clips moved a stream's
embedding by 0.06 percent. This is a property of untrained weights, not of the
wiring, and it makes naive "does the video path work" tests fail for the wrong
reason. Test with a distribution-level difference, or with pretrained weights
outside the automated suite.

### Cross-modal streams are order-blind until attention sharpens

With uniform attention the attended vector is a mean over frames, and a mean
cannot see order. Reversing the frames of an untrained stream changes its
embedding by ~1e-7. That is correct, not a bug: becoming order-sensitive is what
training has to achieve, so the check belongs on a trained checkpoint where it
should invert.

## Training runs

### Never overwrite a checkpoint or history file

`docs/reproducibility.md` requires a new run to write new paths. The comparison
configs under the ignored `runs/comparison-20260904/` keep the baseline's `cache-index`,
`cache-root`, `split-hash` and `preprocessing-hash` and change only the
checkpoint and history filenames, so the runs are comparable and none of them
destroys the frozen evidence.

### `run-id: configured-by-mlflow` is a placeholder

`execute_configured_run` replaces `arguments.run_id` with the live MLflow run id
when tracking is enabled. A literal run id in the config is silently ignored.

### Training reads the cache, not the raw video

`ddf train visual` reads the preprocessed `.npz` cache through
`CachedBranchDataset`. The 1,999-clip cache under
`runs/initial-20260902/cache/` therefore keeps training reproducible while
`data/` is empty. What an empty `data/` does block is a fresh real-video
inference pass, because that path decodes from the original file.

### MLflow records CLI flags under an `arguments.` prefix

A run's seed is `arguments.seed`, not `seed`.
`dashboard/lib/mlflow_runs.py` strips the prefix for its comparison table.
Querying the bare name returns nothing and looks like a run that recorded no
parameters.

### Runs are sequential on one GPU

Train the comparison configs one at a time. A second concurrent run only
contends for the same 16 GB.

### Give pretrained encoders their own learning rate

Training a cross-modal stream with one learning rate across a randomly
initialised attention head and 100M pretrained encoder parameters drove the
training loss to 0.039 while validation climbed to 2.56. `StreamConfig` has
carried `lr_head` and `lr_backbone` since it was written for exactly this.
`ddf train stream --encoder-learning-rate` defaults to 5e-6 against the head's
1e-4; use `parameter_groups` rather than `model.parameters()`.

### `workers: 0` starves the GPU on cached clips

A cached lip-sync clip holds 50 mouth crops at 112x112, which is 7.5 MB of
float32 to decompress per item. With a single-process loader the GPU averaged
17 percent utilisation and epochs took three times longer than necessary.
`workers: 4` raised it to about 43 percent. The cost is invisible without
per-epoch output, which is why `fit_stream` prints as well as logging: MLflow
only receives metrics when the run ends, so a stalled epoch and a slow one look
identical until then.

### A trained fusion model is not an evaluated one

The first version of `run_program.ps1` trained fusion and stopped. Fusion does
not read clips, so scoring it is a three-step chain that is easy to forget:
`features export` turns clips into branch logits with the final checkpoints,
`features score` turns those into probabilities, and `evaluate predictions`
turns those into metrics. Without all three the pipeline produces a
`.joblib` nobody can quote a number from.

Each partition needs its **own** feature store. `train fusion` rejects a store
containing anything but out-of-fold rows, so appending test or external
features to the store the model was fitted on breaks the next fusion run.

### Fusion cannot be evaluated on a corpus without audio

Celeb-DF-v2 and MNW have no audio track, so the audio and sync branches produce
unavailable rows and `FeatureStore.assemble` finds no clip with full branch
coverage. Cross-corpus fusion evidence needs a corpus that carries both
modalities; here that is DFDC alone.

## Streamlit

### `st.navigation` has no disabled entry

Fusion and Explainability have to appear in the sidebar in their pipeline
positions, dimmed and unclickable. The only way to express that is to hide the
built-in navigation with `position="hidden"` and draw the list by hand with
`st.page_link(..., disabled=True)`. Their routes stay registered so a direct
visit lands on a body explaining what unlocks the section rather than on a dead
end.

### `st.Page` paths are relative to the app file's directory

`st.Page("pages/overview.py")` and `st.switch_page("pages/stream_visual.py")`
resolve against `src/deepfake_detection/dashboard/`, where `app.py` lives.
Moving `app.py` breaks every registration and every switch at once.

### Session state disappears for widgets that did not render

Streamlit discards the `session_state` entry behind a widget that was not
rendered on the current run. A value read straight off a widget key resets to
its default the moment you navigate to another page, silently and without an
error. Anything read on a page other than the one whose widget wrote it lives in
a plain dict instead: `lib/sticky.py` for the clip settings and each backbone's
last run, `lib/stream_pages.py` for the architecture settings.

`sticky.widget_default` exists because passing both an explicit default and a
live `session_state` value makes Streamlit warn, so a stored value is offered
only when the widget key is absent.

### Closing a dialog needs an app-scoped rerun

`st.dialog` is a fragment. A button inside it reruns only the dialog body, so
the app-level `if sel_picker_open:` never re-evaluates and Streamlit keeps the
dialog on screen. Clearing the flag in a callback closes nothing. The clip
picker clears the flag and calls `st.rerun()` in the dialog body.

### Browsers cannot play FakeAVCeleb's fake-audio clips

The wav2lip generator wrote MPEG-4 Part 2, which no browser decodes, and those
are exactly the lip-sync forgeries this project studies. `media.playable_video_bytes`
re-encodes to H.264/AAC through PyAV and reports which codec forced it, so the
player never silently serves different pixels than are on disk.

### Decode and detection must be cached or the page crawls

Streamlit reruns the whole script on every widget interaction, including ones
that touch nothing on the clip path. `media.cached_decode_frames`,
`media.cached_face_mouth` and `media.cached_playable_video` memoize on
`(path, mtime, settings)`. `cached_face_mouth` measures detection time inside
the memoized body so a cache hit reports the time detection really took rather
than the near-zero the cache hit cost.

## Testing

### `AppTest.from_file` needs a page that runs standalone

Each file under `dashboard/pages/` executes top to bottom, so it can be handed
straight to `AppTest`. The reusable bodies under `dashboard/sections/` are
functions, and each carries an `if __name__ == "__main__":` block for the same
reason.

### `ElementList` does not concatenate

`page.info + page.error` raises `TypeError`. Wrap each in `list()` first.

### `pytest.importorskip` before imports trips E402

Ruff accepts a bare `pytest.importorskip("torch")` followed by `import torch`,
but flags the assignment form `torch = pytest.importorskip("torch")` when module
imports follow it.

### The dashboard tests run with an empty `data/`

Every page has to render its empty state rather than raise. That is what the
page smoke tests actually check, and it is the state a new checkout is in.

## The dashboard reads checkpoints from a directory nothing writes to

`dashboard/lib/checkpoints.discover` looked in `checkpoints/<stream_name>/` at
the project root. `ddf run` writes into whatever run directory it is given, at
`runs/<run>/checkpoints/`, and nothing copies between the two. The top-level
directory has never existed here, so every stream page showed
"(untrained, random weights)" as the only option while twenty trained
checkpoints sat on disk. Nothing failed and nothing was logged: an empty picker
is a valid state, because no checkpoint is a normal state early in a project.

Two things follow for anyone adding a stream page. Discovery has to search the
runs tree, and a checkpoint's filename is not an identity: `ddf run` writes the
same `fold0-visual.pt` into every run it is pointed at, so a picker keyed on the
name silently keeps one of them.

## A branch checkpoint half-loads into a stream, and the half that fails is the head

`ddf train visual` trains `branches/visual.py`, which ends in a classifier
straight to one logit. The dashboard builds `streams/visual_stream.py`, which
ends in a projection to the shared fusion width with a development head after
it. The backbone and temporal tensors share their names, so 362 of 364 land, and
the load reports six missing and two unexpected.

The trap is the number on the page. Loading is non-strict by design, so the page
happily shows a "fake probability" computed from a randomly initialised head on
top of trained features. Check the load report for a missing `temp_head` before
treating any stream probability as a detection.

Also worth knowing: `ddf train visual` builds a *unidirectional GRU*, while the
dashboard defaults to a BiLSTM. Both load without raising, because the size
mismatch is filtered out first, and the result is a trained backbone feeding a
random temporal model. The picker now reads the gate count out of the tensor
shapes and names the setting to change.

## Visual training is I/O bound, not GPU bound

A visual batch is 16 frames of 3x224x224 float32, about 77 MB, and it arrives as
a compressed npz. Decompression on the main thread runs at 65 ms per clip, so an
epoch over 7,637 clips spends 8 minutes doing nothing but reading, and the GPU
measured 15 percent utilisation across a sampling window.

The tempting fix, `--workers 0`, is the one that causes this. Workers are the
fix, and they can also die on batches this size, which is why both
`run_program.ps1` and `scripts/train_design_b.ps1` attempt a worker count and
retry single-process rather than picking one. A dead worker fails the run
outright instead of degrading the model, so the retry costs an epoch and never
correctness.

Worth knowing before diagnosing a slow run: no per-batch progress is printed, so
a first epoch that has not finished after an hour looks identical to a hang.
Check GPU utilisation and the process CPU time before assuming either.

Worker count is bounded by host RAM, not by the GPU. Each worker holds its
prefetch, and a visual batch of 8 is 616 MB, so three workers reached 2.5 GB
each and drove free memory from 13 GB down to 3.8 GB. The first symptom was not
a dead worker: it was an unrelated `pytest` run failing to start with
`OSError: [WinError 1455] The paging file is too small`. Two workers is the
setting that fits on a 32 GB host.

## Early stopping on loss keeps a checkpoint that scores below chance

Stream training selected its checkpoint on validation BCE, and the two visual
streams showed what that costs. `visual-efficientnet` reached its lowest loss at
epoch 1, with the backbone still frozen, and every later epoch looked worse:
0.8421, then 1.2327, then 1.8231. The saved epoch-1 checkpoint measured 0.4668
ROC-AUC in-domain, which is below chance.

The two metrics answer different questions. BCE measures calibration and
punishes a confident mistake very hard, so a model that ranks clips correctly
while being overconfident scores worse than one that hedges and ranks badly.
Every objective in this project is stated in ROC-AUC.

The symptom is easy to misread as divergence, and it was misread here first: a
rising validation loss beside a collapsing training loss looks exactly like
overfitting. Score the checkpoint before concluding anything from the curve.
`training/ranking.py` now supplies the AUC that both stream trainers select on.
