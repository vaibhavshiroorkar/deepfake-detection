"""Copy the demonstration inputs into data/_demo_inputs/ (git-ignored) under stable names, so the demo in docs/DEMO.md uses fixed files.

Only non-sealed material is used, resolved exactly as the live end-to-end checks do (live_face_fixtures.build: train or val splits through
loaders.select_records(final=False), the LAV-DF clip from the official train directory, plus synthetic files). Nothing is scored here.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/platform/make_demo_inputs.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_face_fixtures as F  # noqa: E402

OUT = F.ROOT / "data" / "_demo_inputs"
NAMES = {  # fixture name -> demo file name (the number orders the demo)
    "ffpp_fake_video": "1_face_swap_video_ffpp_val.mp4",
    "ffpp_real_video": "2_original_video_ffpp_val.mp4",
    "lavdf_face_audio_video": "3_talking_head_with_audio_lavdf_train.mp4",
    "asvspoof_fake_audio": "4_speech_audio_asvspoof_dev.flac",
    "ffpp_fake_frame_image": "5_face_image_frame_ffpp_val.png",
    "synthetic_noface_video": "6_synthetic_no_face_tone_audio.mp4",
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        fixtures, skipped = F.build(Path(td))
        for fx in fixtures:
            name = NAMES.get(fx["name"])
            if name:
                shutil.copyfile(fx["path"], OUT / name)
                print(f"{name:<45} <- {fx['name']} ({fx['split']})")
    (OUT / "7_unsupported_notes.txt").write_text("plain text, not media\n", encoding="utf-8")
    (OUT / "8_corrupt_video.mp4").write_bytes(b"this is not a video" * 100)
    print("7_unsupported_notes.txt, 8_corrupt_video.mp4  <- synthetic error cases")
    for s in skipped:
        print("SKIPPED:", s)
    print(f"written to {OUT}")
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
