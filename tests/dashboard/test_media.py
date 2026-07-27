import av
import io
import numpy as np
import pytest

from dashboard.lib import media
from dashboard.lib.media import sample_timestamps


def test_sample_timestamps_count_and_inset():
    ts = sample_timestamps(duration_sec=8.0, num_frames=16, window_sec=0.35)
    assert ts.shape == (16,)
    assert ts[0] >= 0.35 / 2 - 1e-9          # inset by half a window
    assert ts[-1] <= 8.0 - 0.35 / 2 + 1e-9
    assert np.all(np.diff(ts) > 0)           # strictly increasing


def test_sample_timestamps_degenerate_short_clip():
    ts = sample_timestamps(duration_sec=0.1, num_frames=4, window_sec=0.35)
    assert ts.shape == (4,)
    assert np.all(ts >= 0)


# --------------------------------------------------------- clip playback path
#
# The dataset's wav2lip clips are mpeg4-encoded, which no browser decodes, so
# st.video showed a blank player for exactly the forgeries that matter. These
# tests build synthetic clips rather than reading data/, so they run anywhere.

def _write_clip(path, video_codec, with_audio=False, n_frames=12, size=64):
    """A tiny valid mp4 at `path`, encoded with `video_codec`."""
    out = av.open(str(path), mode="w", format="mp4")
    vs = out.add_stream(video_codec, rate=25)
    vs.width = vs.height = size
    vs.pix_fmt = "yuv420p"
    aus = res = None
    if with_audio:
        aus = out.add_stream("aac", rate=16000)
        res = av.AudioResampler(format=aus.format.name, layout=aus.layout.name, rate=aus.rate)

    rng = np.random.default_rng(0)
    for _ in range(n_frames):
        arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        for packet in vs.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
            out.mux(packet)
    for packet in vs.encode():
        out.mux(packet)

    if aus is not None:
        samples = (rng.standard_normal((1, 16000)) * 0.1).astype(np.float32)
        frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="mono")
        frame.rate = 16000
        for resampled in res.resample(frame):
            for packet in aus.encode(resampled):
                out.mux(packet)
        for packet in aus.encode():
            out.mux(packet)
    out.close()
    return path


def _streams(data: bytes) -> dict[str, str]:
    """{'video': codec, 'audio': codec} for an in-memory mp4."""
    with av.open(io.BytesIO(data)) as c:
        found = {}
        if c.streams.video:
            found["video"] = c.streams.video[0].codec_context.name
        if c.streams.audio:
            found["audio"] = c.streams.audio[0].codec_context.name
        return found


def test_video_codec_names_the_stream(tmp_path):
    assert media.video_codec(_write_clip(tmp_path / "a.mp4", "mpeg4")) == "mpeg4"
    assert media.video_codec(_write_clip(tmp_path / "b.mp4", "libx264")) == "h264"


def test_browser_codec_is_passed_through_untouched(tmp_path):
    """An h264 clip must not be re-encoded — that would be pure wasted latency."""
    path = _write_clip(tmp_path / "ok.mp4", "libx264")
    data, reencoded_from = media.playable_video_bytes(path)
    assert reencoded_from is None
    assert data == path.read_bytes()


def test_mpeg4_clip_is_reencoded_to_h264(tmp_path):
    """The actual bug: mpeg4 in, h264 out, and the caller is told why."""
    path = _write_clip(tmp_path / "wavtolip.mp4", "mpeg4")
    data, reencoded_from = media.playable_video_bytes(path)
    assert reencoded_from == "mpeg4"
    assert _streams(data)["video"] == "h264"
    with av.open(io.BytesIO(data)) as c:
        assert c.streams.video[0].codec_context.pix_fmt == "yuv420p"


def test_reencode_keeps_the_audio_track(tmp_path):
    """A silent player is as broken as a blank one — the clip must keep sound."""
    path = _write_clip(tmp_path / "av.mp4", "mpeg4", with_audio=True)
    data, _ = media.playable_video_bytes(path)
    assert _streams(data) == {"video": "h264", "audio": "aac"}


def test_reencode_is_faststart(tmp_path):
    """moov ahead of mdat, so the browser can start without the whole file."""
    path = _write_clip(tmp_path / "fs.mp4", "mpeg4")
    data, _ = media.playable_video_bytes(path)
    assert 0 < data.find(b"moov") < data.find(b"mdat")


def test_reencode_of_video_only_clip(tmp_path):
    """No audio stream must not crash the audio pass."""
    path = _write_clip(tmp_path / "mute.mp4", "mpeg4", with_audio=False)
    data, _ = media.playable_video_bytes(path)
    assert _streams(data) == {"video": "h264"}


def test_transcode_rejects_a_file_with_no_video(tmp_path):
    path = tmp_path / "audio_only.m4a"
    out = av.open(str(path), mode="w", format="mp4")
    aus = out.add_stream("aac", rate=16000)
    res = av.AudioResampler(format=aus.format.name, layout=aus.layout.name, rate=aus.rate)
    frame = av.AudioFrame.from_ndarray(
        np.zeros((1, 16000), dtype=np.float32), format="fltp", layout="mono")
    frame.rate = 16000
    for resampled in res.resample(frame):
        for packet in aus.encode(resampled):
            out.mux(packet)
    for packet in aus.encode():
        out.mux(packet)
    out.close()

    assert media.video_codec(path) is None
    with pytest.raises(ValueError, match="no video stream"):
        media.transcode_to_h264(path)
