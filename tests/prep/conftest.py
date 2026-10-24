import shutil

import pytest

from prep_test_helpers import make_video


@pytest.fixture(scope="session")
def ffmpeg_ok():
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")


@pytest.fixture(scope="session")
def video50(tmp_path_factory, ffmpeg_ok):
    return make_video(tmp_path_factory.mktemp("vid") / "v50.mp4", n_frames=50)
