import shutil
import subprocess
from pathlib import Path

import pytest

from deepfake_detection.views.media import FFmpegMediaDecoder


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_decoder_reads_real_video_frames_and_aligned_audio(tmp_path: Path) -> None:
    media = tmp_path / "fixture.mp4"
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    # The test controls every argument and never invokes a shell.
    subprocess.run(  # noqa: S603
        [
            ffmpeg,
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=32x32:r=10",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000",
            "-t",
            "1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(media),
        ],
        check=True,
    )
    decoder = FFmpegMediaDecoder()

    info = decoder.probe(media)
    frames = decoder.read_frames(media, (0.1, 0.8))
    audio = decoder.read_audio(
        media,
        start_sec=0.25,
        duration_sec=0.5,
        sample_rate=16_000,
    )

    assert info.duration_sec == pytest.approx(1.0, abs=0.15)
    assert info.audio_present
    assert len(frames) == 2
    assert frames[0].shape == (32, 32, 3)
    assert len(audio) == 8_000
