"""Helpers: tiny synthetic videos made with ffmpeg."""
import subprocess

import numpy as np

LEVEL0, STEP = 10, 4  # frame i is a flat grey image of level LEVEL0 + STEP * i


def make_video(path, n_frames=50, w=64, h=48, fps=25, gop=10, with_audio=False):
    """Lossless H.264 video whose frame i has grey level LEVEL0 + STEP*i, so the index can be read back."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "pipe:0"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={n_frames / fps}"]
    cmd += ["-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", "-g", str(gop)]
    if with_audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(path)]
    data = b"".join(np.full((h, w), LEVEL0 + STEP * i, np.uint8).tobytes() for i in range(n_frames))
    subprocess.run(cmd, input=data, check=True)
    return path


def frame_number(frame_bgr):
    return int(round((float(frame_bgr.mean()) - LEVEL0) / STEP))
