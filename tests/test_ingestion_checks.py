"""The ingestion checks that P-002 runs on every corpus before any model (handbook mistakes 67 to 69)."""

import pytest

from deepfake_detection.data.ingestion_checks import (
    face_coverage,
    full_span,
    label_in_window,
    overlap,
    shortcut_ceiling,
)


def test_overlap_sums_every_segment_inside_the_window():
    assert overlap(1.0, 4.0, [[0.5, 1.5], [3.5, 5.0]]) == pytest.approx(1.0)
    assert overlap(1.0, 4.0, []) == 0.0


def test_face_coverage_fails_when_too_few_windows_have_a_face():
    # DFDC before A10: most windows had no face.
    result = face_coverage([0.0, 0.0, 1.0, 0.9, 0.5], bar=0.9)
    assert not result.passed
    assert result.value == pytest.approx(0.4)


def test_face_coverage_passes_on_clean_windows():
    assert face_coverage([1.0, 0.95, 0.85], bar=0.9).passed


def test_a_clip_shorter_than_the_span_fails_even_when_its_streams_match():
    # Mistake 67: the source video ended early and -shortest cut the audio with it,
    # so both streams were 2.24 s where the span is 3.00 s.
    result = full_span([(3.0, 3.0), (2.24, 2.24), (2.98, 3.0)], span=3.0)
    assert not result.passed
    assert result.value == pytest.approx(2 / 3)


def test_a_clip_missing_its_video_fails():
    assert not full_span([(2.0, 3.0)], span=3.0).passed


def test_label_outside_the_window_fails():
    # A9: the kept window missed the edited words.
    windows = [(0.0, 3.0), (0.0, 3.0), (2.0, 5.0)]
    periods = [[[4.0, 4.5]], [[1.0, 1.4]], [[4.0, 4.5]]]
    result = label_in_window(windows, periods)
    assert not result.passed
    assert result.value == pytest.approx(2 / 3)


def test_less_than_a_frame_of_overlap_does_not_count():
    result = label_in_window([(0.0, 3.0)], [[[2.98, 3.5]]])
    assert result.value == 0.0


def test_shortcut_ceiling_is_a_maximum():
    assert shortcut_ceiling(0.58).passed
    failed = shortcut_ceiling(0.632)
    assert not failed.passed and failed.bar == 0.60


def test_results_say_what_they_measured():
    result = face_coverage([1.0], bar=0.9)
    assert result.name == "face coverage" and "0.8" in result.detail
