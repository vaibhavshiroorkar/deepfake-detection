# Multipage Teaching Dashboard Design

## Goal

Build a multipage Streamlit dashboard that explains the deepfake detector in
pipeline order. A viewer should understand what enters the system, what each
implemented stage does, what the trained visual model returns, and which
research stages remain incomplete.

## Audience

The main audience is a project reviewer, supervisor, or student who needs to
see how the system works without reading the source first. The dashboard is a
research demonstration. It is not a public prediction service.

## Design reference

The `origin/old` dashboard supplies the interaction reference. Its useful
patterns are a visible page hierarchy, pipeline-order navigation, cumulative
step explanations, clip previews, tensor shapes, and locked pages that explain
what will unlock them.

The implementation will not copy obsolete model code, W&B controls, arbitrary
checkpoint selection, or claims from the old branch. It will use the current
preprocessor, visual checkpoint, MLflow history, and research limits.

## Navigation

The sidebar will present pages in this order:

1. Overview
2. Video input
3. Preprocessing
4. Visual model
5. Prediction
6. Experiments
7. Audio branch
8. Sync branch
9. Fusion
10. Documentation

The order describes the real workflow. Audio, sync, and fusion pages remain
visible because they are part of the research design. Their pages clearly
state their current prototype or locked status.

## Shared state

The video input page accepts one local upload and stores its bytes, filename,
suffix, and stable content hash in Streamlit session state. Later pages rebuild
a temporary file only while they need it and delete it in a `finally` block.

Preprocessing output may be cached by the uploaded content hash and frozen
preprocessing identity. Pages never accept a client-controlled model path.
The only runnable checkpoint is the frozen visual development baseline.

If a viewer opens a later page before uploading a clip, the page explains that
the first step is incomplete and links back to Video input.

## Page behavior

### Overview

Explain the research question, the four datasets, the implemented visual
baseline, and the planned multimodal system. Show a compact pipeline diagram
made with Streamlit layout and text. State that FaceForensics++ is paused and
that MNW is evaluation-only.

### Video input

Show one upload control, supported formats, a video preview, filename, size,
and content hash. Explain that the upload stays in the local Streamlit session
and is not added to a dataset or training run.

### Preprocessing

Run the current visual-only preprocessor. Present its work as numbered stages:
media probe, uniform timestamp sampling, face detection and tracking, face
crop, resize and normalization, and the final `[16, 3, 224, 224]` tensor.
Show sampled face crops when preprocessing succeeds. Explain failures in plain
language and do not fabricate missing views.

### Visual model

Explain the current EfficientNet-B0 plus GRU architecture. Show the input shape,
frame feature flow, temporal sequence role, classifier logit, and sigmoid
probability. This page may run the frozen visual branch but must not claim that
intermediate heatmaps explain the model unless a real explanation method is
implemented.

### Prediction

Run the same provenance-checked visual inference engine used by the current
dashboard. Show the verdict, probability, fixed threshold `0.5`, visual
evidence availability, and research limits. Store the latest result in session
state so expanding technical details does not rerun the model.

### Experiments

Show the saved training summary and fixed-threshold validation metrics from
the local run records. Provide the local MLflow URL and the exact training and
evaluation run IDs. Label the metrics as FakeAVCeleb development validation,
not cross-dataset evidence.

### Audio branch

Explain the audio spoof branch input, architecture, and existing prototype
status. Do not offer a final audio prediction because full training is not
complete.

### Sync branch

Explain mouth-audio alignment, offset classification, and the existing
prototype status. Do not present a research score before full training and
evaluation.

### Fusion

Show how visual, audio, and sync evidence will enter calibrated late fusion.
Keep the page locked for runnable prediction. State that the current fusion
file is a software fixture and cannot support a research claim.

### Documentation

Link only to repository documents that exist. Include the handoff, research
design, data card, reproducibility guide, model selection rules, and CLI
reference.

## Visual direction

The visual language is a laboratory field notebook. It uses cool paper,
graphite text, cobalt controls, teal available evidence, amber incomplete
evidence, and muted red manipulated evidence. Bahnschrift carries headings,
Aptos carries prose, and Cascadia Mono carries hashes, shapes, and scores.

The signature element is a vertical pipeline index in the sidebar. Completed,
current, prototype, and locked states use words and color together. Color never
carries status alone.

Pages use a bounded reading column, clear numbered stages, restrained borders,
and visible focus states. The layout must work on narrow screens and respect
reduced-motion preferences. No animation is required.

## Security and integrity

- Bind Streamlit to `127.0.0.1`.
- Accept no client-controlled checkpoint, model, or fusion paths.
- Verify checkpoint SHA-256, run ID, split hash, training commit, seed, branch,
  and preprocessing hash before inference.
- Keep the decision threshold fixed at `0.5` and display it with each result.
- Use `torch.load(..., weights_only=True)` through the existing checkpoint
  loader.
- Do not expose the `joblib` fusion loader from the dashboard.
- Delete temporary uploads after each operation.

## Component boundaries

- `dashboard/app.py` owns navigation and global styling.
- `dashboard/state.py` owns uploaded clip and cached result session contracts.
- `dashboard/components.py` owns shared status, stage, and prerequisite UI.
- `dashboard/pages/` contains one focused module per page.
- `dashboard/configuration.py` remains the frozen checkpoint authority.
- `inference/` and `views/` remain the only model and preprocessing
  implementations. Dashboard pages call them instead of duplicating them.

## Error handling

Missing uploads produce a direct next step. Missing local artifacts name the
missing file. Decode, face detection, provenance, CUDA, and inference errors
show plain recovery guidance. The interface never replaces an error with a
score.

## Testing

Tests use Streamlit `AppTest` for navigation order, locked states, prerequisite
messages, absence of model path controls, and static-page rendering. Unit tests
cover shared state, metric parsing, and result presentation. Existing
provenance, preprocessor, and inference tests remain the model-path contract.

The final verification runs the full test suite, Ruff lint and formatting,
documentation checks, lock validation, a Streamlit component smoke, and one
real CUDA prediction with the frozen checkpoint.

## Completion criteria

- All ten pages appear in pipeline order.
- A clip selected on Video input is available to later pages.
- Preprocessing, visual model, and prediction pages show real outputs.
- Experiments shows the saved MLflow-linked development evidence.
- Audio, sync, and fusion status cannot be mistaken for completed research.
- No browser control can choose a local artifact path or threshold.
- All automated and runtime checks pass.
