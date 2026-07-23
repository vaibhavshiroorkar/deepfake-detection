# Preprocessing Single-Page Redesign — Design

**Date:** 2026-07-23
**Status:** approved, ready to implement
**Supersedes:** the two-page split (`dashboard/pages/preprocess_visual.py`,
`dashboard/pages/preprocess_audio.py`) from the multi-page dashboard.

## Purpose

Replace the two separate preprocessing pages with **one organized, sequential page**
that reads top-to-bottom as a pipeline. Each preprocessing step is applied
**cumulatively** and shows the result **after** it runs, so toggling any step visibly
changes everything downstream. The page ends with the exact tensor the model receives.

Unchanged constraints (PROJECT_OVERVIEW.md §7): read-only, never writes
`data/processed/`, never trains. Reuses the pure ops in
`dashboard/lib/{visual_ops,audio_ops}.py` verbatim — only page composition changes.

## Layout (top to bottom)

1. **Selection** — shared Dataset / Target / Clip selector (`selectors.render_selection`)
   in the main area, plus a compact "pipeline settings" row: N frames, audio window (s),
   and a "preview frame" index slider (which of the N frames the per-step thumbnails show).
2. **Visual pipeline** — one bordered step card per stage, in pipeline order. Each card:
   controls (toggle + params) on the left, the **cumulative result up to that stage** on
   the right (the preview frame, with an expander for all N). Stages:
   - Decode (baseline "Original" full frame, resized to 224 for display)
   - Face detect + crop (toggle; conf, margin) — off = full-frame passthrough
   - Quality & robustness (sharpen, denoise, CLAHE, blur, JPEG, downscale — each a toggle,
     applied in that fixed order)
   - Mouth-region crop (toggle)
   - ImageNet normalize (toggle; renders the de-normalized frame + prints the pixel range)
   Disabled stages render a "skipped — passthrough" note and pass their input through.
3. **Audio pipeline** — same bordered-card treatment; each card shows the waveform
   **after** that stage. Stages: Decode (baseline) → Mono downmix → Resample →
   Trim leading silence → Quality (noise reduction, RMS normalize, bandpass, add noise) →
   Window extraction (→ `[N, win]`) → Mel-spectrogram view (toggle; drawn below the final
   waveform).
4. **Model input (final step)** — clearly labelled *"this is what the model receives."*
   Shows the final face tensor grid `[N, 3, 224, 224]` (de-normalized for display), the
   final audio windows `[N, win]`, and the label — with `shape`, `dtype`, and value range
   printed for each. This mirrors the `ClipDataset`/`make_dataloader` contract.

## Structure & non-messiness

- Single vertical flow — **no sidebar controls, no tabs.** `st.container(border=True)`
  per step, `st.divider()` between the three sections, section headers via `st.header`.
- Cumulative computation is explicit: the page walks the frames/waveform through the
  stages in Python, snapshotting after each stage, then renders each snapshot in its card.
  One representative frame per card keeps it compact; an expander reveals all N.

## Files

- Create: `dashboard/pages/preprocess.py`
- Modify: `dashboard/app.py` — nav "Data Preprocessing" section now holds the single
  `preprocess.py` page (title "Preprocessing").
- Delete: `dashboard/pages/preprocess_visual.py`, `dashboard/pages/preprocess_audio.py`,
  `tests/dashboard/test_preprocess_visual_smoke.py`,
  `tests/dashboard/test_preprocess_audio_smoke.py`.
- Create: `tests/dashboard/test_preprocess_smoke.py` — AppTest smoke: page runs without
  exception and renders the "Visual pipeline", "Audio pipeline", and "Model input" headers.

## Out of scope (YAGNI)

- No step reordering UI (fixed sensible order).
- No change to the pure ops or to `selectors`/`media`.
- Stream pages and the streams work (sub-projects B, C) are unchanged here.
