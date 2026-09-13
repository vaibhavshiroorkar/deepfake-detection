"""The gate exists because of a measured inversion, so the tests name it.

The visual stream reads motion as evidence of authenticity. Generated video is
typically smooth. So the class the system has no mechanism for is also the class
its features push towards "real", and a confident verdict there is the failure
this flag is meant to make visible.
"""

import numpy as np

from deepfake_detection.inference.distribution_gate import (
    TRAINING_MOTION_QUARTILES,
    check,
    frame_motion,
)


def _clip(step: float, frames: int = 16) -> np.ndarray:
    """A view whose consecutive frames differ by exactly `step`."""
    return np.stack(
        [np.full((3, 8, 8), index * step, dtype=np.float32) for index in range(frames)]
    )


def test_an_ordinary_clip_sits_inside_the_training_range() -> None:
    result = check(_clip(TRAINING_MOTION_QUARTILES[1]))

    assert result.inside
    assert result.blocker is None


def test_unnaturally_smooth_footage_is_flagged_and_the_reason_says_why() -> None:
    result = check(_clip(0.001))

    assert not result.inside
    assert result.blocker == "outside_training_distribution_below"
    assert "smoother" in result.reason


def test_violent_motion_is_flagged_as_above_the_range() -> None:
    result = check(_clip(2.0))

    assert not result.inside
    assert result.blocker == "outside_training_distribution_above"


def test_a_single_frame_carries_no_temporal_statistic() -> None:
    result = check(_clip(0.2, frames=1))

    assert result.inside
    assert np.isnan(result.motion)
    assert "single frame" in result.reason


def test_motion_is_the_mean_absolute_difference_between_frames() -> None:
    assert frame_motion(_clip(0.25)) == 0.25
