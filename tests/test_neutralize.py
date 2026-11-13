import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from deepfake_detection.preprocessing.neutralize import (
    NeutralSettings,
    cut_window,
    ffmpeg_command,
    neutralize_clip,
    silence_bounds,
)


def test_silence_bounds_find_digital_silence_at_both_ends() -> None:
    rate = 1000
    wave = np.concatenate([np.zeros(30), 0.5 * np.ones(900), np.zeros(70)]).astype(np.float32)
    leading, trailing = silence_bounds(wave, rate)
    assert leading == pytest.approx(0.030)
    assert trailing == pytest.approx(0.070)


def test_quiet_speech_is_not_silence() -> None:
    # -50 dBFS is quiet but above the -60 dBFS cut.
    wave = np.full(1000, 10 ** (-50 / 20), np.float32)
    assert silence_bounds(wave, 1000) == (0.0, 0.0)


def test_a_clip_that_would_become_too_short_is_kept_whole() -> None:
    assert cut_window(1.2, 0.5, 0.5, minimum=1.0) == (0.0, 1.2)
    assert cut_window(6.0, 0.03, 1.2, minimum=1.0) == (0.03, pytest.approx(4.8))


def test_one_seek_cuts_both_streams() -> None:
    command = ffmpeg_command(Path("a.mp4"), Path("b.mp4"), 0.03, 4.8, True, NeutralSettings())
    # The seek sits before -i, so it applies to the input and therefore to both streams.
    assert command.index("-ss") < command.index("-i")
    assert "-map_metadata" in command and "libx264" in command and "loudnorm" in " ".join(command)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_neutral_copy_drops_the_silence_and_changes_the_codec(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    # 0.5 s silence, 2 s tone, 1 s silence, under an MPEG-4 Part 2 video.
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=size=64x64:rate=25:duration=3.5",
         "-f", "lavfi", "-i",
         "aevalsrc='if(between(t,0.5,2.5),0.3*sin(2*PI*440*t),0)':s=16000:d=3.5",
         "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(source)],
        check=True,
    )
    target = tmp_path / "neutral.mp4"
    report = neutralize_clip(source, target, NeutralSettings())
    assert "error" not in report
    assert report["offset_sec"] == pytest.approx(0.5, abs=0.06)
    assert report["kept_sec"] == pytest.approx(2.0, abs=0.1)
    codec = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name", "-of", "csv=p=0", str(target)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert codec == "h264"
