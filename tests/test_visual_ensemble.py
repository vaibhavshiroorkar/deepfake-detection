"""The visual ensemble averages logits, not probabilities, and can serve a
subset of its members without loading them twice."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
import torch
from torch import nn

from deepfake_detection.branches.registry import architecture_of
from deepfake_detection.inference.ensemble import (
    VisualEnsembleEngine,
    program_members,
)
from deepfake_detection.inference.predictor import _InferenceRecord
from deepfake_detection.views.contracts import PreparedClip, QualityReport


class Fixed(nn.Module):
    def __init__(self, logit: float) -> None:
        super().__init__()
        self.logit = logit

    def forward(self, values):
        class Output:
            logits = torch.full((values.shape[0],), self.logit)

        return Output()


class StubPreprocessor:
    def __init__(self, view):
        self.view = view

    def prepare_visual(self, record, path):
        return PreparedClip(
            clip_id=record.clip_id,
            visual_view=self.view,
            audio_view=None,
            sync_video_view=None,
            sync_audio_view=None,
            quality=QualityReport(
                face_coverage=1.0,
                stable_face_track=True,
                audio_present=False,
                audio_clipped=False,
                av_duration_delta_sec=0.0,
            ),
            preprocessing_fingerprint="fingerprint",
            preprocessing_config_hash="hash",
        )


FACES = np.zeros((16, 3, 8, 8), dtype=np.float32)


def engine(view=FACES):
    return VisualEnsembleEngine(
        preprocessor=StubPreprocessor(view),
        models=[("a", Fixed(2.0)), ("b", Fixed(-1.0)), ("c", Fixed(0.5))],
        threshold=0.5,
        device="cpu",
    )


def test_probability_is_the_sigmoid_of_the_mean_logit():
    prediction = engine().predict(Path("clip.mp4"))
    expected = 1 / (1 + np.exp(-(2.0 - 1.0 + 0.5) / 3))
    assert prediction.result.probability == pytest.approx(expected)
    assert prediction.result.verdict == "fake"
    assert set(prediction.member_probabilities) == {"a", "b", "c"}
    assert prediction.member_probabilities["b"] == pytest.approx(1 / (1 + np.e))


def test_a_subset_scores_only_the_named_members():
    prediction = engine().predict(Path("clip.mp4"), members=["b"])
    assert prediction.result.branch_logits == {"b": pytest.approx(-1.0)}
    assert prediction.result.verdict == "real"


def test_the_threshold_can_be_set_per_call():
    prediction = engine().predict(Path("clip.mp4"), threshold=0.7)
    assert prediction.result.threshold == 0.7
    assert prediction.result.verdict == "real"


def test_unknown_members_are_refused():
    with pytest.raises(ValueError, match="Unknown ensemble members"):
        engine().predict(Path("clip.mp4"), members=["z"])


def test_no_face_gives_no_verdict():
    prediction = engine(view=None).predict(Path("clip.mp4"))
    assert prediction.result.verdict == "indeterminate"
    assert prediction.result.blockers == ("missing_visual",)
    assert prediction.visual_view is None


def test_program_members_name_six_checkpoints_by_architecture():
    members = program_members(Path("checkpoints"))
    assert len(members) == 6
    for member in members:
        # Old checkpoints record no architecture, so the name has to decide.
        assert member.name.startswith(architecture_of(member.checkpoint, "visual"))


def test_an_upload_record_carries_every_field_the_preprocessor_reads():
    # prepare_visual reads sync_start_sec for the fingerprint. Without it the
    # visual-only engine raised AttributeError on every upload.
    record = _InferenceRecord(clip_id="upload")
    assert record.sync_start_sec is None
    assert record.leading_silence_sec == 0.0
