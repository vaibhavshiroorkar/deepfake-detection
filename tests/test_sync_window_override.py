"""Per-clip placement of the synchronisation window.

LAV-DF hides a 0.8 to 1.6 second forgery somewhere inside an otherwise genuine
clip. The default window is pinned near the start, so for most fake clips it
would contain no manipulation while still carrying a fake label. The override
lets the manifest say where to look, which is also what makes it possible to cut
a fake window and a genuine window from the same recording.
"""

from pathlib import Path

import pytest

from deepfake_detection.data.manifest import ClipRecord
from deepfake_detection.views.cache import cache_fingerprint
from deepfake_detection.views.timeline import ViewConfig


def record(**overrides) -> ClipRecord:
    values = {
        "clip_id": "clip-1",
        "dataset": "fixture",
        "video_path": "clip-1.mp4",
        "manipulation_type": "RealVideo-RealAudio",
        "method": "real",
        "source": "identity-1",
    }
    values.update(overrides)
    return ClipRecord.from_mapping(values)


@pytest.fixture
def media(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"fixture-media")
    return path


def fingerprint(media: Path, **kwargs) -> str:
    return cache_fingerprint(
        media,
        dataset="fixture",
        config=ViewConfig(),
        code_version="test-v1",
        **kwargs,
    )


def test_default_records_carry_no_override() -> None:
    assert record().sync_start_sec == 0.0


def test_override_survives_the_manifest_round_trip() -> None:
    assert record(sync_start_sec="3.75").sync_start_sec == pytest.approx(3.75)


def test_no_override_leaves_the_fingerprint_unchanged(media: Path) -> None:
    """The load-bearing test. Every clip cached before this option existed must
    keep its key, or adding the feature silently invalidates the whole cache."""
    before = fingerprint(media)
    after = fingerprint(media, sync_start_sec=0.0)

    assert before == after


def test_two_windows_of_one_clip_get_different_keys(media: Path) -> None:
    """Matched pairs depend on this. The same media, config and dataset must not
    collide, or the second window overwrites the first in the cache."""
    forged = fingerprint(media, sync_start_sec=6.4)
    genuine = fingerprint(media, sync_start_sec=1.2)

    assert forged != genuine
    assert forged != fingerprint(media)


def test_override_is_independent_of_leading_silence(media: Path) -> None:
    assert fingerprint(media, sync_start_sec=2.0) != fingerprint(
        media, leading_silence_sec=2.0
    )


def test_negative_window_start_is_rejected() -> None:
    with pytest.raises(ValueError, match="Sync start"):
        record(sync_start_sec="-1.0")


def test_non_finite_window_start_is_rejected() -> None:
    with pytest.raises(ValueError, match="Sync start"):
        record(sync_start_sec="nan")
