"""Routing an upload to the streams that can actually read it.

The rule under test: a stream runs only when every view it needs is present.
Without it the dashboard would hand a photograph to a lip-sync model and report
whatever number fell out, which looks like a result and is not one.
"""

import pytest

from deepfake_detection.dashboard.lib import media_kind


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("clip.mp4", media_kind.VIDEO),
        ("clip.MOV", media_kind.VIDEO),
        ("face.jpg", media_kind.IMAGE),
        ("face.PNG", media_kind.IMAGE),
        ("voice.wav", media_kind.AUDIO),
        ("voice.mp3", media_kind.AUDIO),
        ("notes.txt", media_kind.UNKNOWN),
        ("no_suffix", media_kind.UNKNOWN),
    ],
)
def test_classify_reads_the_suffix(filename: str, expected: str) -> None:
    assert media_kind.classify(filename) == expected


def test_a_video_drives_every_stream() -> None:
    assert set(media_kind.runnable(media_kind.VIDEO)) == {
        "visual",
        "audio",
        "lipsync",
        "emotion",
    }


def test_an_image_drives_only_the_visual_stream() -> None:
    """No sound and no time axis, so three of the four have nothing to read."""
    assert media_kind.runnable(media_kind.IMAGE) == ("visual",)


def test_an_image_marks_the_visual_stream_degraded() -> None:
    """It runs, but the model was trained on sixteen frames and reads how the
    face changes between them. Saying "runs" without that caveat would overstate
    what the number means."""
    visual = next(
        item
        for item in media_kind.availability(media_kind.IMAGE)
        if item.stream == "visual"
    )

    assert visual.available
    assert visual.degraded
    assert "one frame" in visual.reason


def test_audio_drives_only_the_audio_branch() -> None:
    assert media_kind.runnable(media_kind.AUDIO) == ("audio",)


def test_a_cross_modal_stream_never_runs_on_one_modality() -> None:
    """Lip-sync and emotion compare two modalities. With one, there is nothing
    to compare, so they must not run at all rather than run on a zero tensor."""
    for kind in (media_kind.IMAGE, media_kind.AUDIO):
        runnable = media_kind.runnable(kind)
        assert "lipsync" not in runnable
        assert "emotion" not in runnable


def test_every_unavailable_stream_gives_a_reason() -> None:
    """An empty panel with no explanation reads as a fault."""
    for kind in (media_kind.IMAGE, media_kind.AUDIO):
        for entry in media_kind.availability(kind):
            if not entry.available:
                assert entry.reason, f"{kind}/{entry.stream} has no reason"


def test_an_unknown_kind_runs_nothing() -> None:
    assert media_kind.runnable(media_kind.UNKNOWN) == ()
    assert "nothing can run" in media_kind.describe(media_kind.UNKNOWN).lower()


def test_describe_names_what_will_run() -> None:
    assert media_kind.describe(media_kind.IMAGE) == "Visual only."
    assert media_kind.describe(media_kind.AUDIO) == "Audio only."
    assert "Lip-sync" in media_kind.describe(media_kind.VIDEO)


def test_upload_suffixes_cover_all_three_kinds() -> None:
    """The file_uploader accepts exactly what classify can route."""
    suffixes = set(media_kind.UPLOAD_SUFFIXES)

    assert {"mp4", "jpg", "wav"} <= suffixes
    for suffix in suffixes:
        assert media_kind.classify(f"x.{suffix}") != media_kind.UNKNOWN
