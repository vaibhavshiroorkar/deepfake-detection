import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from deepfake_detection.preprocessing.neutralize import (
    SAMPLE_RATE,
    NeutralSettings,
    cut_window,
    dither,
    ffmpeg_command,
    neutralize_clip,
    normalize_level,
    silence_bounds,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def test_silence_bounds_find_digital_silence_at_both_ends() -> None:
    rate = 1000
    wave = np.concatenate([np.zeros(30), 0.5 * np.ones(900), np.zeros(70)]).astype(
        np.float32
    )
    leading, trailing = silence_bounds(wave, rate)
    assert leading == pytest.approx(0.030)
    assert trailing == pytest.approx(0.070)


def test_quiet_speech_is_not_silence() -> None:
    wave = np.full(1000, 10 ** (-50 / 20), np.float32)
    assert silence_bounds(wave, 1000) == (0.0, 0.0)


def test_cuts_start_on_a_whole_frame_and_have_one_length() -> None:
    # 31 ms of leading silence rounds up to frame 1 (40 ms); the span is fixed.
    start, end = cut_window(6.0, 0.031, 1.19, fixed=3.0, fps=25)
    assert start == pytest.approx(0.04)
    assert end == pytest.approx(3.04)
    # 1.2 s of sound cannot give a 3 s span.
    assert cut_window(2.0, 0.4, 0.4, fixed=3.0, fps=25) is None


def test_level_normalization_is_a_plain_gain() -> None:
    wave = np.sin(np.linspace(0, 200, 16000)).astype(np.float32) * 0.01
    out = normalize_level(wave, -23.0)
    assert 20 * np.log10(
        np.sqrt(np.mean(out.astype(np.float64) ** 2))
    ) == pytest.approx(-23.0, abs=0.05)
    # Same shape, so no delay: the peak sits where it was.
    assert int(np.argmax(out)) == int(np.argmax(wave))


def test_audio_comes_from_the_cut_wav_not_from_the_source() -> None:
    command = ffmpeg_command(
        Path("a.mp4"), Path("b.mp4"), 0.04, 4.8, Path("a.wav"), NeutralSettings()
    )
    assert command.index("-ss") < command.index("-i")
    assert command[command.index("1:a:0") - 1] == "-map"
    assert "loudnorm" not in " ".join(command) and "-bf" in command


def _decode(path: Path) -> np.ndarray:
    out = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "f32le",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(out.stdout, np.float32)


def _starts(path: Path) -> list[str]:
    return subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,start_time",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()


def _make(path: Path, video_codec: str, audio_codec: str, silence: float) -> None:
    tone = f"if(gte(t,{silence}),0.3*sin(2*PI*440*t),0)"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=25:duration=3",
            "-f",
            "lavfi",
            "-i",
            f"aevalsrc='{tone}':s=44100:d=3",
            "-c:v",
            video_codec,
            "-c:a",
            audio_codec,
            "-shortest",
            str(path),
        ],
        check=True,
    )


@pytest.mark.skipif(not HAS_FFMPEG, reason="needs ffmpeg")
def test_differently_built_sources_leave_identical_edges(tmp_path: Path) -> None:
    # Version 1 let each source's AAC priming leak through the re-mux: face-swapped
    # FakeAVCeleb clips gained 28 ms of leading silence that real clips did not.
    first, second = tmp_path / "a.mp4", tmp_path / "b.mp4"
    _make(first, "mpeg4", "aac", silence=0.5)
    _make(second, "libx264", "libmp3lame", silence=0.0)
    outputs = []
    for source in (first, second):
        target = tmp_path / f"n-{source.name}"
        report = neutralize_clip(source, target, NeutralSettings(fixed_seconds=2.0))
        assert "error" not in report
        outputs.append(target)
    for target in outputs:
        leading, _ = silence_bounds(_decode(target))
        assert leading < 0.002
    assert _starts(outputs[0]) == _starts(outputs[1])


@pytest.mark.skipif(not HAS_FFMPEG, reason="needs ffmpeg")
def test_neutral_copy_drops_the_silence_and_changes_the_codec(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=25:duration=3.5",
            "-f",
            "lavfi",
            "-i",
            "aevalsrc='if(between(t,0.5,2.5),0.3*sin(2*PI*440*t),0)':s=16000:d=3.5",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ],
        check=True,
    )
    target = tmp_path / "neutral.mp4"
    report = neutralize_clip(source, target, NeutralSettings(fixed_seconds=1.5))
    assert "error" not in report
    assert report["offset_sec"] == pytest.approx(0.52, abs=0.05)
    assert report["kept_sec"] == pytest.approx(1.5, abs=0.01)
    codec = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
            str(target),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert codec == "h264"


@pytest.mark.skipif(not HAS_FFMPEG, reason="needs ffmpeg")
def test_a_clip_with_too_little_sound_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "short.mp4"
    _make(source, "libx264", "aac", silence=2.5)  # 0.5 s of tone in a 3 s clip
    report = neutralize_clip(
        source, tmp_path / "out.mp4", NeutralSettings(fixed_seconds=2.0)
    )
    assert report["error"].startswith("too short")


def test_dither_removes_digital_zeros_and_repeats_per_file() -> None:
    zeros = np.zeros(16000, np.float32)
    once, again = dither(zeros, -66.0, "a.mp4"), dither(zeros, -66.0, "a.mp4")
    assert np.count_nonzero(once) == once.size
    assert np.array_equal(once, again)
    assert not np.array_equal(once, dither(zeros, -66.0, "b.mp4"))
    level = 20 * np.log10(np.sqrt(np.mean(once.astype(np.float64) ** 2)))
    assert level == pytest.approx(-66.0, abs=0.5)


def test_a_fixed_size_scales_every_clip_to_one_frame() -> None:
    # HiFi-AVDF: generated clips are 1920x1080 or 1920x1072, real ones 1280x720.
    # Stretching, not padding: padding would add black bars to one generator only.
    command = ffmpeg_command(
        Path("a.mp4"), Path("b.mp4"), 0.0, 3.0, None, NeutralSettings(size="1280x720")
    )
    video_filter = command[command.index("-vf") + 1]
    assert video_filter.startswith("scale=1280:720") and "pad" not in video_filter
    plain = ffmpeg_command(
        Path("a.mp4"), Path("b.mp4"), 0.0, 3.0, None, NeutralSettings()
    )
    assert "scale" not in plain[plain.index("-vf") + 1]


def test_a_fixed_video_bitrate_replaces_constant_quality() -> None:
    """P-001 amendment A18: the same neutralizer, encoding video at one constant bitrate."""
    fixed = ffmpeg_command(
        Path("a.mp4"),
        Path("b.mp4"),
        0.0,
        3.0,
        None,
        NeutralSettings(video_bitrate="160k"),
    )
    assert "-crf" not in fixed
    for flag in ("-b:v", "-minrate", "-maxrate", "-bufsize"):
        assert fixed[fixed.index(flag) + 1] == "160k"
    assert "nal-hrd=cbr" in fixed[fixed.index("-x264-params") + 1]
    plain = ffmpeg_command(
        Path("a.mp4"), Path("b.mp4"), 0.0, 3.0, None, NeutralSettings()
    )
    assert "-crf" in plain and "-b:v" not in plain  # v4 itself is unchanged


def test_sound_windows_cover_all_the_sound_in_fixed_parts() -> None:
    from deepfake_detection.preprocessing.neutralize import sound_windows

    # 10 s of sound from 0.5 s: parts at 0.52, 3.52, 6.52, and a last one ending where the sound ends
    parts = sound_windows(duration=11.0, leading=0.5, trailing=0.5, fixed=3.0)
    assert [round(s, 2) for s, _ in parts] == [0.52, 3.52, 6.52, 7.48]
    assert all(round(e - s, 6) == 3.0 for s, e in parts)
    assert parts[-1][1] <= 10.5 + 1e-9
    assert sound_windows(duration=3.52, leading=0.52, trailing=0.0, fixed=3.0) == [
        (0.52, 3.52)
    ]
    assert len(sound_windows(duration=60.0, leading=0.0, trailing=0.0, fixed=3.0)) == 20
    assert sound_windows(duration=3.0, leading=0.5, trailing=0.0, fixed=3.0) == []
    first = cut_window(11.0, 0.5, 0.5, 3.0)
    assert parts[0] == first


@pytest.mark.skipif(not HAS_FFMPEG, reason="needs ffmpeg")
def test_a_given_start_cleans_that_part_of_the_clip(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _make(source, "libx264", "aac", silence=0.0)
    report = neutralize_clip(
        source, tmp_path / "out.mp4", NeutralSettings(fixed_seconds=1.0), start=1.0
    )
    assert "error" not in report
    assert report["offset_sec"] == pytest.approx(1.0)
    assert report["kept_sec"] == pytest.approx(1.0)
