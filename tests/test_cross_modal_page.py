"""The cross-modal pages carry the project's central finding, so they get tests.

The number on them, diagonal mass against chance, is the evidence that a stream
scoring 0.9978 is not using the cue it is named for. A page that computed the
baseline wrong would invert that conclusion, and this project has already made
exactly that mistake once by comparing an eight-step map against a fifty-step
baseline.
"""

import numpy as np
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from deepfake_detection.dashboard.lib.cross_modal_live import (
    chance_diagonal_mass,
    measured_diagonal_mass,
)


def test_chance_matches_the_values_recorded_during_training() -> None:
    """The two shapes the streams actually produce, from findings.md."""
    assert round(chance_diagonal_mass(8, 8), 4) == 0.3438
    assert round(chance_diagonal_mass(50, 50), 4) == 0.0592


def test_the_baseline_depends_on_the_query_length() -> None:
    """The mistake this guards: one baseline reused across shapes."""
    assert chance_diagonal_mass(8, 8) > 5 * chance_diagonal_mass(50, 50)


def test_a_uniform_map_measures_its_own_chance_value() -> None:
    for size in (8, 50):
        uniform = np.full((size, size), 1.0 / size)

        assert measured_diagonal_mass(uniform) == pytest.approx(
            chance_diagonal_mass(size, size), abs=1e-6
        )


def test_a_perfectly_diagonal_map_measures_one() -> None:
    assert measured_diagonal_mass(np.eye(16)) == pytest.approx(1.0)


@pytest.mark.parametrize("page", ["stream_lipsync", "stream_emotion"])
def test_the_page_renders_without_a_clip_selected(page: str) -> None:
    rendered = AppTest.from_file(
        f"src/deepfake_detection/dashboard/pages/{page}.py", default_timeout=60
    ).run()

    assert not rendered.exception
    assert rendered.title


def test_partial_checkpoint_is_refused_before_forward(tmp_path, monkeypatch):
    import torch

    from deepfake_detection.dashboard.lib import cross_modal_live
    from deepfake_detection.streams import cross_modal_stream

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(2, 1)

        def forward(self, **kwargs):
            pytest.fail("Partial weights reached inference")

    monkeypatch.setattr(
        cross_modal_stream, "build_lipsync_stream", lambda **kwargs: TinyModel()
    )
    checkpoint = tmp_path / "stream-lipsync.pt"
    torch.save({"model": {"head.weight": torch.zeros(1, 3)}}, checkpoint)
    result = cross_modal_live.run_stream("lipsync", checkpoint, object(), device="cpu")
    assert result["incompatible_checkpoint"]
    assert result["report"]["mismatched"]
    assert "probability" not in result
