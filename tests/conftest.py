"""Small synthetic media files shared by the inference and web tests (CPU only, ffmpeg for video)."""
import shutil
import subprocess

import cv2
import numpy as np
import pytest
import soundfile as sf


def _ff(args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", *args], check=True)


@pytest.fixture(scope="session")
def media(tmp_path_factory):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    d = tmp_path_factory.mktemp("media")
    rng = np.random.default_rng(0)
    cv2.imwrite(str(d / "img.png"), rng.integers(0, 255, (96, 128, 3), dtype=np.uint8))
    cv2.imwrite(str(d / "tiny.png"), rng.integers(0, 255, (32, 32, 3), dtype=np.uint8))
    sf.write(d / "speech.wav", rng.uniform(-0.3, 0.3, 16000 * 5).astype(np.float32), 16000)
    sf.write(d / "short.wav", rng.uniform(-0.3, 0.3, 8000).astype(np.float32), 16000)
    _ff(["-f", "lavfi", "-i", "testsrc=size=128x96:rate=10:duration=1", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(d / "clip_audio.mp4")])
    _ff(["-f", "lavfi", "-i", "testsrc=size=128x96:rate=10:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(d / "clip_silent.mp4")])
    (d / "corrupt.png").write_bytes(b"not an image at all")
    (d / "corrupt.wav").write_bytes(b"RIFF....garbage")
    (d / "corrupt.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42 broken")
    (d / "empty.png").write_bytes(b"")
    (d / "notes.txt").write_text("hello")
    return d
