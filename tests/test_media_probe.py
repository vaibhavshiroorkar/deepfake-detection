"""The real decoder reads a sound file, and the video entry points refuse it.

Probing used to raise on any file without a video stream, so `prepare_audio`
could never run on an uploaded wav outside the stub-decoder tests.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("av")

from deepfake_detection.data.manifest import ClipRecord
from deepfake_detection.views.media import FFmpegMediaDecoder
from deepfake_detection.views.preprocessor import Preprocessor
from deepfake_detection.views.timeline import ViewConfig

FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg is not installed")


@pytest.fixture
def tone(tmp_path: Path) -> Path:
    path = tmp_path / "tone.wav"
    subprocess.run(  # noqa: S603
        [
            FFMPEG,
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1.5",
            "-ar",
            "16000",
            str(path),
        ],
        check=True,
    )
    return path


def record(path: Path) -> ClipRecord:
    return ClipRecord(
        clip_id="tone",
        dataset="test",
        video_path=path,
        manipulation_type="RealVideo-RealAudio",
        method="real",
        source="test",
        targets=(),
        clip_fake=False,
        video_fake=False,
        audio_fake=False,
    )


def test_probe_describes_a_sound_file(tone):
    info = FFmpegMediaDecoder().probe(tone)
    assert info.audio_present
    assert not info.video_present
    assert info.duration_sec == pytest.approx(1.5, abs=0.05)
    assert info.video_fps == 0.0


def test_a_sound_file_prepares_as_audio_and_is_refused_as_video(tone):
    preprocessor = Preprocessor(
        decoder=FFmpegMediaDecoder(),
        detector=None,
        config=ViewConfig(),
        code_version="test",
    )
    prepared = preprocessor.prepare_audio(record(tone), tone)
    assert prepared.audio_view is not None
    assert prepared.visual_view is None
    with pytest.raises(ValueError, match="no video stream"):
        preprocessor.prepare_visual(record(tone), tone)
