"""End-to-end contract check for extract_clip on a real sample clip.

Skips when the dataset isn't extracted (CI without data), so it never blocks a
run, but when clips are present it verifies the aligned pipeline still emits the
exact tensors every downstream stage depends on.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from preprocessing.extract_clip import extract_clip
from preprocessing.ops.constants import NUM_FRAMES, FRAME_SIZE, AUDIO_SR, AUDIO_WINDOW_SEC

_DATA = Path(__file__).resolve().parents[2] / "data"
_WIN = int(AUDIO_WINDOW_SEC * AUDIO_SR)


def _first_existing_clip():
    for split in ("val", "train", "test"):
        csv = _DATA / f"{split}.csv"
        if not csv.exists():
            continue
        for _, row in pd.read_csv(csv).iterrows():
            vp = _DATA / row["video_path"]
            if vp.exists():
                return row["clip_id"], vp
    return None


@pytest.fixture(scope="module")
def sample_clip():
    clip = _first_existing_clip()
    if clip is None:
        pytest.skip("No extracted dataset clip available.")
    return clip


def test_extract_clip_contract(sample_clip):
    clip_id, video_path = sample_clip
    out = extract_clip(video_path, clip_id, force=True, device="cpu")

    frames, audio = out["frames"], out["audio"]
    assert frames.shape == (NUM_FRAMES, FRAME_SIZE, FRAME_SIZE, 3)
    assert frames.dtype == np.uint8 and 0 <= frames.min() and frames.max() <= 255
    assert audio.shape == (NUM_FRAMES, _WIN) and audio.dtype == np.float32
    assert out["timestamps"].shape == (NUM_FRAMES,)
    assert out["leading_silence_sec"] >= 0.0


def test_alignment_changes_pixels_vs_bbox_crop(sample_clip):
    """When a face is found, 5-point alignment yields different pixels than the
    plain bbox crop, i.e. alignment is actually doing something."""
    clip_id, video_path = sample_clip
    aligned = extract_clip(video_path, clip_id, force=True, device="cpu", align=True)
    if aligned["num_faces_detected"] == 0:
        pytest.skip("No face detected in this clip; alignment path not exercised.")
    bbox = extract_clip(video_path, clip_id, force=True, device="cpu", align=False)
    assert not np.array_equal(aligned["frames"], bbox["frames"])
