"""
Tests for the clip-level labelling convention.

FakeAVCeleb's `type` column is the ground truth for what was manipulated. This
module pins down how that becomes a binary label, because getting it wrong is
silent -- the pipeline still runs, the metrics still print, and every number
downstream is meaningless.
"""
import pytest

from preprocessing.manifest import clip_label


def test_only_real_video_real_audio_is_labelled_real():
    """A clip is real only if BOTH tracks are real."""
    assert clip_label("RealVideo-RealAudio") == 0
    assert clip_label("RealVideo-FakeAudio") == 1
    assert clip_label("FakeVideo-RealAudio") == 1
    assert clip_label("FakeVideo-FakeAudio") == 1


def test_unrecognised_type_raises_instead_of_guessing():
    """
    A typo must not quietly become "fake". Silently mislabelling is the
    failure mode that produces confident, meaningless metrics.
    """
    with pytest.raises(ValueError, match="RealVideo-Realaudio"):
        clip_label("RealVideo-Realaudio")
