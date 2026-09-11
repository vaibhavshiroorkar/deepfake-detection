"""Every dashboard page has to survive a real clip selection.

The rest of the dashboard tests run pages with nothing selected, so each one
returns at its "pick a clip first" guard and the whole body below it goes
unexecuted. That is how a page could reference a name that does not exist and
still pass the suite. These tests hand every page a selected clip so the body
runs.

The clip is a synthetic ffmpeg pattern with no face in it. Pages must cope with
that anyway (the detector finds nothing on plenty of real frames), and it keeps
the test off the dataset in data/.
"""

import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from deepfake_detection.dashboard.lib import selectors

PAGES = Path("src/deepfake_detection/dashboard/pages")
ALL_PAGES = sorted(p.name for p in PAGES.glob("*.py") if p.name != "__init__.py")

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is required"
)


@pytest.fixture(scope="session")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("clip") / "fixture.mp4"
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    # The test controls every argument and never invokes a shell.
    subprocess.run(  # noqa: S603
        [
            ffmpeg, "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=s=320x240:r=25",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000",
            "-t", "2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-shortest", str(path),
        ],
        check=True,
    )
    return path


@pytest.mark.parametrize("name", ALL_PAGES)
def test_page_renders_with_a_clip_selected(
    name: str, clip: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = pd.Series(
        {
            "clip_id": "fixture",
            "video_path": str(clip),
            "label": 0,
            "manipulation_type": "RealVideo-RealAudio",
            "method": "real",
            "source": "fixture",
        }
    )
    # **kwargs because the real one takes allow_preprocessing and key: the
    # stream pages offer three clip sources, so they pass both.
    monkeypatch.setattr(selectors, "render_selection", lambda **_: row)

    page = AppTest.from_file(PAGES / name, default_timeout=600).run()

    assert not page.exception, [e.message for e in page.exception]
