"""
Pure functions for turning FakeAVCeleb's meta_data.csv into our manifest.

Kept separate from audit_dataset.py (which does the file walking and CSV
writing) so this logic can be tested without touching the 21,544-clip dataset.
"""


# The four categories FakeAVCeleb ships, as they appear in meta_data.csv's
# `type` column. Anything outside this set is a bug, not a new category.
CLIP_TYPES = (
    "RealVideo-RealAudio",
    "RealVideo-FakeAudio",
    "FakeVideo-RealAudio",
    "FakeVideo-FakeAudio",
)


def clip_label(clip_type: str) -> int:
    """
    Binary CLIP-LEVEL label: 1 = fake, 0 = real.

    A clip is real only if both its tracks are real, so RealVideo-FakeAudio
    counts as fake here even though every pixel is genuine.

    Note this is NOT the label a visual-only stream trains on -- that stream
    cannot see the audio, so RealVideo-FakeAudio is "real" to it. See
    docs/PROJECT_OVERVIEW.md section 6.
    """
    if clip_type not in CLIP_TYPES:
        raise ValueError(
            f"Unrecognised FakeAVCeleb type {clip_type!r}. "
            f"Expected one of {CLIP_TYPES}."
        )
    return 0 if clip_type == "RealVideo-RealAudio" else 1
