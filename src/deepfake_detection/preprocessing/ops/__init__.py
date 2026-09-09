"""Pure preprocessing ops. No Streamlit, no Torch, no disk writes.

Every step is a NumPy/OpenCV/librosa function, so each one is unit-testable and
there is exactly one implementation of it.

Layout:
  constants      shapes, sample rates, ImageNet stats, PIPELINE_VERSION.
  faces          main visual steps: detect, crop, normalize, mouth ROI.
  audio          main audio steps: decode, downmix, resample, silence, windows.
  detectors      the face detectors behind one small interface.
  extras_visual  robustness and augmentation image ops (off by default).
  extras_audio   robustness and augmentation audio ops (off by default).
"""
