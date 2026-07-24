"""Canonical, pure preprocessing ops shared by the real pipeline and the dashboard.

No Streamlit, no Torch, no disk writes — just NumPy/OpenCV/librosa functions so
every step is unit-testable and there is exactly ONE implementation of each.

Layout:
  constants      — the single home for shapes, sample rates, ImageNet stats, the
                   5-point alignment template, and PIPELINE_VERSION.
  faces          — MAIN visual steps: detect, align, crop, normalize, mouth ROI.
  audio          — MAIN audio steps: decode, downmix, resample, silence, windows.
  extras_visual  — robustness/augmentation image ops (off by default).
  extras_audio   — robustness/augmentation audio ops (off by default).

See docs/preprocessing.md for what each step does and why.
"""
