"""Checks a corpus must pass before any model reads it (P-002 design, handbook mistakes 67 to 69).

P-001 found three preprocessing faults only after reading results: clips with less
video than audio, LAV-DF windows that missed the edited words (A9), and small faces
the landmarker missed (A10). Each check here turns one of them, plus the shortcut
ceiling, into a pass or fail with the measured value and the bar it was held to.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

FRAME_SEC = 0.04  # one video frame at 25 fps


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    value: float
    bar: float
    detail: str


def overlap(start: float, end: float, periods: Sequence[Sequence[float]]) -> float:
    """Seconds of the window [start, end] covered by labelled segments."""
    return sum(max(0.0, min(end, b) - max(start, a)) for a, b in periods)


def _share(flags: Sequence[bool]) -> float:
    if not flags:
        raise ValueError("a check needs at least one item")
    return sum(flags) / len(flags)


def face_coverage(
    face_shares: Sequence[float], *, min_face: float = 0.8, bar: float = 0.9
) -> CheckResult:
    """Share of windows whose face was found in at least `min_face` of frames."""
    value = _share([share >= min_face for share in face_shares])
    return CheckResult(
        "face coverage",
        value >= bar,
        value,
        bar,
        f"share of windows with a face in at least {min_face} of frames",
    )


def full_span(
    durations: Sequence[tuple[float, float]],
    *,
    span: float = 3.0,
    tolerance: float = FRAME_SEC,
    bar: float = 1.0,
) -> CheckResult:
    """Share of clips whose video and audio, (video_sec, audio_sec), both last the full fixed span.

    Comparing the streams with each other is not enough: when a source video ends early,
    muxing with -shortest cuts the audio too, so both streams match and both are short.
    """
    value = _share(
        [min(video, audio) >= span - tolerance for video, audio in durations]
    )
    return CheckResult(
        "full span",
        value >= bar,
        value,
        bar,
        f"share of clips whose video and audio both last {span} s, within {tolerance} s",
    )


def label_in_window(
    windows: Sequence[tuple[float, float]],
    periods: Sequence[Sequence[Sequence[float]]],
    *,
    min_overlap: float = FRAME_SEC,
    bar: float = 1.0,
) -> CheckResult:
    """Share of fake clips whose scored window holds at least `min_overlap` s of labelled manipulation."""
    value = _share(
        [
            overlap(start, end, segs) >= min_overlap
            for (start, end), segs in zip(windows, periods, strict=True)
        ]
    )
    return CheckResult(
        "label in window",
        value >= bar,
        value,
        bar,
        f"share of fake clips with at least {min_overlap} s of edit inside the scored window",
    )


def shortcut_ceiling(ceiling: float, *, bar: float = 0.60) -> CheckResult:
    """The cue-only classifier's AUC on the neutral copies must stay at or below `bar`."""
    return CheckResult(
        "shortcut ceiling",
        ceiling <= bar,
        ceiling,
        bar,
        "out-of-fold AUC from file and audio-edge cues alone, at most the bar",
    )
