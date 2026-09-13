"""Refuse to answer on input unlike anything the model was trained on.

The measured problem this exists for: the visual stream reads frame-to-frame
motion as evidence of authenticity, correlation -0.316 among manipulated DFDC
clips, and fully generated video is typically smooth. So the one class the
system has no mechanism for is also the class its learned features push towards
"real". On MNW it detects 0 of 10 `vasa_1` clips while reporting ordinary
probabilities, and expected calibration error cross-corpus is 0.6366 against
0.0022 in-domain.

A confident number on input the model has never seen anything like is worse than
a refusal, so this measures the gap and lets the caller abstain. It is not a
deepfake detector for generated video and must not be described as one. It
answers a narrower question: does this clip's low-level statistics sit inside
the range the training corpus covered?

The statistic is mean absolute difference between consecutive frames of the
prepared view, which is the tensor the model is handed, and the reference
quantiles come from scoring the training corpus with
`scripts/motion_vs_error.py`. One statistic is not a distribution test. It
catches the obvious cases, which are the ones being reported as confident
verdicts today, and it will miss a generated clip whose motion happens to look
ordinary. That limit is the reason the result is a flag beside the verdict
rather than a replacement for it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Measured on the 1,648-clip FakeAVCeleb in-domain test partition, the corpus
# every served model was trained on. Regenerate with
# `python scripts/motion_vs_error.py --dataset in-domain`.
TRAINING_MOTION_QUARTILES = (0.1751, 0.2278, 0.3038)
TRAINING_MOTION_MEAN = 0.2429
TRAINING_MOTION_RANGE = (0.0864, 0.5394)

# The first and ninety-ninth percentiles of the training corpus, used as the
# bounds directly.
#
# The obvious choice was a 1.5 IQR fence, and a test caught it: the quartiles
# are 0.1751 and 0.3038, so the lower fence lands at -0.018, and motion is an
# absolute difference that cannot go below zero. The "too smooth" case could
# never have fired, and that is the case this gate exists for, since generated
# video is typically smoother than any real recording.
#
# Percentiles instead, which flag about 2 percent of training clips as outside
# their own corpus. That rate is the honest cost of the check and is the reason
# the result is a flag beside the verdict rather than a refusal.
TRAINING_MOTION_BOUNDS = (0.1017, 0.4789)


@dataclass(frozen=True, slots=True)
class DistributionCheck:
    """Whether one clip sits inside the training corpus's motion range."""

    motion: float
    inside: bool
    position: str
    reason: str = ""

    @property
    def blocker(self) -> str | None:
        return None if self.inside else f"outside_training_distribution_{self.position}"


def frame_motion(view: np.ndarray) -> float:
    """Mean absolute difference between consecutive frames of a prepared view."""
    frames = np.asarray(view, dtype=np.float32)
    if frames.ndim < 2 or len(frames) < 2:
        # A single frame has no temporal statistic. An image upload is already
        # marked degraded by the routing table, so this reports rather than
        # invents a value.
        return float("nan")
    return float(np.abs(np.diff(frames, axis=0)).mean())


def check(view: np.ndarray) -> DistributionCheck:
    """Compare a clip's motion against the training corpus's range."""
    motion = frame_motion(view)
    if not np.isfinite(motion):
        return DistributionCheck(
            motion=float("nan"),
            inside=True,
            position="unknown",
            reason="A single frame carries no motion statistic to compare.",
        )

    low, high = TRAINING_MOTION_BOUNDS

    if motion > high:
        return DistributionCheck(
            motion=motion,
            inside=False,
            position="above",
            reason=(
                f"Frame-to-frame motion is {motion:.3f}, above the {high:.3f} "
                "that 99 percent of the training corpus sits under. Hand-held "
                "or fast-cut footage lands here, and the model was trained on "
                "seated talking heads."
            ),
        )
    if motion < low:
        return DistributionCheck(
            motion=motion,
            inside=False,
            position="below",
            reason=(
                f"Frame-to-frame motion is {motion:.3f}, below the {low:.3f} "
                "that 99 percent of the training corpus sits above. Synthetic "
                "video is often smoother than any real recording, and the model "
                "reads smoothness as evidence of authenticity, so a verdict "
                "here would lean towards 'real' for the wrong reason."
            ),
        )
    return DistributionCheck(motion=motion, inside=True, position="inside")
