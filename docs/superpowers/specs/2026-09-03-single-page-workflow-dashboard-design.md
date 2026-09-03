# Single-page workflow dashboard design

## Problem

The multipage teaching dashboard exposes the full project, but its navigation
weakens the focused analysis flow. The original dashboard has a clearer visual
identity and a faster path from upload to verdict.

The revised dashboard will keep the original design and add the teaching
features as a guided, single-page workflow.

## Goals

- Preserve the original hero, typography, colors, result card, evidence gate,
  and frozen-baseline sidebar.
- Present video analysis as four numbered steps on one page.
- Keep later steps unavailable until their required data exists.
- Retain the research, prototype, and documentation content without competing
  with the main workflow.
- Reuse the tested upload, preprocessing, prediction, runtime, state, and
  evidence code from the multipage implementation.
- Keep the dashboard bound to the local host and preserve artifact provenance
  checks.

## Non-goals

- Do not add new model capabilities.
- Do not unlock audio, sync, or fusion inference.
- Do not change checkpoint metadata, thresholds, or preprocessing behavior.
- Do not add artifact path controls or load untrusted model files.
- Do not redesign the original visual language.

## Page structure

The dashboard will use one Streamlit page with this order:

1. Original masthead and introduction.
2. Frozen-baseline details in the sidebar.
3. Step 1, Video input.
4. Step 2, Preprocessing.
5. Step 3, Visual model.
6. Step 4, Prediction.
7. Research and methodology.

Each workflow step will use an expander. The first incomplete actionable step
will open by default. Completed steps will remain available for review. A short
status line will identify each step as waiting, ready, complete, prototype, or
locked.

The Research and methodology expander will contain five tabs:

- Experiments
- Audio branch
- Sync branch
- Fusion
- Documentation

These tabs retain the current teaching and status content. Audio and sync stay
marked as prototypes. Fusion stays locked.

## Workflow behavior

### Step 1: Video input

The user uploads one supported video and can preview or remove it. Replacing or
removing the upload clears derived preprocessing and prediction state.

### Step 2: Preprocessing

This step stays disabled until a video exists. It explains the processing
stages and runs the existing visual preprocessing path. It then shows coverage,
quality blockers, clip metadata, and sampled face evidence.

### Step 3: Visual model

This step explains the verified visual model and its tensor flow. It shows the
frozen checkpoint, run, split, commit, seed, threshold, and device details. The
section can be read before preprocessing finishes, but it cannot imply that a
specific upload has been scored.

### Step 4: Prediction

This step stays disabled until preprocessing succeeds. It runs the existing
prediction path and uses the original result card and three-channel evidence
gate. It also shows blockers, limitations, branch scores, and the preprocessing
fingerprint.

## State and data flow

`dashboard/state.py` remains the source of truth for uploaded, prepared, and
predicted clip state. `dashboard/runtime.py` remains responsible for cached
preprocessing and prediction services. The single page reads those modules and
passes their values to focused rendering functions.

The page will not duplicate model loading or temporary-file handling. Upload
identity remains tied to the content hash. Derived state must match that hash
before the page displays it.

## Code structure

`dashboard/app.py` will own the page shell, original CSS, workflow order, and
expander state. Existing page modules will expose rendering functions instead
of executing only at import time. The app will call those functions inside the
matching workflow section or research tab.

`dashboard/navigation.py` will no longer drive runtime navigation. Small status
types may remain if the single-page components use them. Dead page-link code
will be removed.

## Errors and blocked states

Expected media, CUDA, checkpoint, and provenance failures will appear inside
the step that caused them. A failure must not erase the upload. Failed
preprocessing clears stale prepared and prediction values. Failed prediction
clears only stale prediction values.

Disabled steps will explain the required prior action. Prototype and locked
research sections will state their limits without showing controls that cannot
work.

## Accessibility and responsive behavior

The original high-contrast palette and visible focus rings will remain. Status
must use text as well as color. The evidence gate will collapse to one column
on narrow screens. Expanders and tabs will keep their native keyboard behavior.
Reduced-motion preferences will remain respected.

## Testing

Tests will cover:

- The original hero, baseline sidebar, and evidence-gate styling.
- The four workflow steps and their order.
- Disabled and enabled states for upload, preprocessing, and prediction.
- Upload replacement and removal state clearing.
- Original result rendering with the new prediction state.
- The five research tabs and prototype or locked labels.
- Expected error handling and stale-state clearing.
- A Streamlit application smoke test with no uncaught exceptions.

The existing dashboard runtime, state, evidence, preprocessing, and prediction
tests will remain in use. Multipage navigation tests will be replaced with
single-page workflow tests.
