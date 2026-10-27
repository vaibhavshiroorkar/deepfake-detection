"""Re-encoding a clip so a browser can play it.

OpenCV and PyAV decode a far wider range of codecs than a browser will play, so
a clip the pipeline reads fine can still arrive at a video player as a blank
frame. Both the dashboard and the web app serve clips through this.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import av

# Video codecs that every current browser can decode natively. FakeAVCeleb is
# NOT uniformly one of them: the wav2lip generator wrote its output with an
# MPEG-4 Part 2 encoder, so every FakeVideo-FakeAudio clip (and some
# FakeVideo-RealAudio ones) is `mpeg4`, which no browser will play. Those are
# exactly the lip-sync forgeries this project is built around, so the clip
# player has to re-encode rather than shrug.
BROWSER_VIDEO_CODECS = frozenset({"h264", "vp8", "vp9", "av1"})


def video_codec(video_path) -> str | None:
    """Name of the file's video codec, or None if it carries no video stream."""
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            return None
        return container.streams.video[0].codec_context.name


def transcode_to_h264(video_path) -> bytes:
    """Re-encode a clip to browser-playable H.264/AAC MP4 and return its bytes.

    Goes through PyAV rather than a system ffmpeg binary, for the same reason the
    audio path does (see deepfake_detection/preprocessing/ops/audio.decode): PyAV bundles ffmpeg's
    libraries, so this needs no external install.

    Written to a real temp file rather than a BytesIO because `+faststart` moves
    the moov atom to the front in a second pass, which needs a seekable,
    re-openable output. Video and audio are encoded in separate passes over the
    input, which keeps the muxer's interleaving simple and preserves duration.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "playable.mp4"
        with av.open(str(video_path)) as inp:
            if not inp.streams.video:
                raise ValueError(f"{video_path} has no video stream to re-encode")
            src_v = inp.streams.video[0]
            src_a = inp.streams.audio[0] if inp.streams.audio else None

            out = av.open(
                str(out_path),
                mode="w",
                format="mp4",
                options={"movflags": "+faststart"},
            )
            dst_v = out.add_stream("libx264", rate=src_v.average_rate or 25)
            dst_v.width, dst_v.height = src_v.width, src_v.height
            dst_v.pix_fmt = "yuv420p"  # the only pix_fmt browsers all decode
            dst_v.options = {"crf": "23", "preset": "veryfast"}

            dst_a = resampler = None
            if src_a is not None:
                dst_a = out.add_stream("aac", rate=src_a.rate)
                # Some clips carry mp3 audio, whose frame format the AAC encoder
                # will not take directly.
                resampler = av.AudioResampler(
                    format=dst_a.format.name, layout=dst_a.layout.name, rate=dst_a.rate
                )

            for frame in inp.decode(video=0):
                for packet in dst_v.encode(frame):
                    out.mux(packet)
            for packet in dst_v.encode():
                out.mux(packet)

            if dst_a is not None:
                inp.seek(0)
                for frame in inp.decode(audio=0):
                    for resampled in resampler.resample(frame):
                        for packet in dst_a.encode(resampled):
                            out.mux(packet)
                for packet in dst_a.encode():
                    out.mux(packet)
            out.close()

        return out_path.read_bytes()


def playable_video_bytes(video_path) -> tuple[bytes, str | None]:
    """(mp4 bytes st.video can play, the codec that forced a re-encode or None).

    A clip already in a browser codec is handed over untouched; anything else is
    re-encoded, and the second element names what it was so the caller can say
    so rather than silently serving different pixels than are on disk.
    """
    codec = video_codec(video_path)
    if codec in BROWSER_VIDEO_CODECS:
        return Path(video_path).read_bytes(), None
    return transcode_to_h264(video_path), codec
