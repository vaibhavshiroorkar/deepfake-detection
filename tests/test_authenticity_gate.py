"""The gate that turns a false accusation into an abstention.

It exists because the detector calls 90.97 percent of genuine Celeb-DF clips
fake, and no threshold fixes that. What matters in the serving path is narrow:
unfamiliar footage must produce no verdict, familiar footage must be untouched,
and an absent model must not break serving.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from deepfake_detection.inference.authenticity import (
    AuthenticityDensity,
    load,
    load_gate,
    save,
)


def _model(threshold: float = 5.0) -> AuthenticityDensity:
    width = 4
    return AuthenticityDensity(
        mean=np.zeros(width),
        components=np.eye(width),
        centre=np.zeros(width),
        inverse=np.eye(width),
        threshold=threshold,
        encoder="dinov3",
        reference="real",
        flag_rate=0.1,
    )


def test_familiar_footage_passes_and_names_no_blocker() -> None:
    check = _model().check(np.array([1.0, 0.0, 0.0, 0.0]))

    assert check.familiar
    assert check.blocker is None
    assert check.reason == ""


def test_unfamiliar_footage_is_flagged_and_says_why() -> None:
    check = _model().check(np.array([10.0, 0.0, 0.0, 0.0]))

    assert not check.familiar
    assert check.blocker == "unfamiliar_capture_conditions"
    assert "nine times in ten" in check.reason


def test_the_model_survives_a_round_trip(tmp_path: Path) -> None:
    """The fitted model and the served one must be the same model."""
    original = _model(threshold=7.25)
    path = tmp_path / "density.npz"

    save(path, original)
    restored = load(path)

    assert restored.threshold == pytest.approx(7.25)
    assert restored.reference == "real"
    assert restored.encoder == "dinov3"
    probe = np.array([3.0, 0.0, 0.0, 0.0])
    assert restored.distance(probe) == pytest.approx(original.distance(probe))


def test_a_missing_model_serves_without_the_gate(tmp_path: Path) -> None:
    """Absence is normal: a checkout with no fitted model still answers."""
    assert load_gate(tmp_path / "nothing.npz") is None
