"""Scoring a video, an image or a sound file through the streams it supports.

The rule under test is that routing decides what runs, not the views. An image
produces a visual_view, and the emotion stream reads that same view, so without
asking the routing table an image would silently reach a model that also expects
a voice and score on half its input.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
import torch
from torch import nn

from deepfake_detection.fusion.deep import StreamFusion
from deepfake_detection.fusion.deep_loading import load_fusion
from deepfake_detection.fusion.stream_export import StreamSpec
from deepfake_detection.inference.multimodal import MultimodalEngine, _modality
from deepfake_detection.views.contracts import PreparedClip, QualityReport

DIMS = {"visual-efficientnet": 4, "stream-emotion": 4, "final-audio-seed17": 4}


class Stub(nn.Module):
    """Stands in for any stream shape, recording that it ran."""

    def __init__(self, kind: str, width: int = 4) -> None:
        super().__init__()
        self.kind = kind
        self.width = width
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        batch = 1

        class Output:
            logit = torch.full((batch,), 0.3)
            logits = torch.full((batch,), 0.3)
            embedding = torch.ones(batch, self.width)

        if self.kind == "visual":
            return torch.full((batch,), 0.3), torch.ones(batch, self.width)
        return Output()


class StubPreprocessor:
    """Returns views directly, so nothing here decodes a real file."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def _clip(self, **views) -> PreparedClip:
        return PreparedClip(
            clip_id="upload",
            visual_view=views.get("visual"),
            audio_view=views.get("audio"),
            sync_video_view=views.get("sync_video"),
            sync_audio_view=views.get("sync_audio"),
            quality=QualityReport(1.0, True, True, False, 0.0),
            preprocessing_fingerprint="fingerprint",
            preprocessing_config_hash="hash",
        )

    def prepare(self, record, path):
        self.calls.append("video")
        return self._clip(
            visual=np.zeros((16, 3, 8, 8), dtype=np.float32),
            audio=np.zeros((64,), dtype=np.float32),
            sync_video=np.zeros((4, 3, 8, 8), dtype=np.float32),
            sync_audio=np.zeros((64,), dtype=np.float32),
        )

    def prepare_image(self, record, path):
        self.calls.append("image")
        return self._clip(visual=np.zeros((1, 3, 8, 8), dtype=np.float32))

    def prepare_audio(self, record, path):
        self.calls.append("audio")
        return self._clip(audio=np.zeros((64,), dtype=np.float32))


@pytest.fixture
def engine(tmp_path: Path) -> MultimodalEngine:
    torch.manual_seed(17)
    head = StreamFusion(stream_dims=DIMS, common_dim=8, hidden_sizes=(4,))
    path = tmp_path / "fusion.pt"
    torch.save(
        {
            "model": head.state_dict(),
            "stream_dims": DIMS,
            "common_dim": 8,
            "hidden_sizes": [4],
        },
        path,
    )
    streams = {
        "visual-efficientnet": StreamSpec(
            "visual-efficientnet", Stub("visual"), "h", "visual"
        ),
        "stream-emotion": StreamSpec("stream-emotion", Stub("cross"), "h", "emotion"),
        "final-audio-seed17": StreamSpec(
            "final-audio-seed17", Stub("audio"), "h", "audio"
        ),
    }
    return MultimodalEngine(
        preprocessor=StubPreprocessor(),
        streams=streams,
        fusion=load_fusion(path),
        device="cpu",
    )


def test_a_video_drives_every_stream(engine: MultimodalEngine) -> None:
    result = engine.predict(Path("clip.mp4"))

    assert set(result.branch_logits) == set(DIMS)
    assert result.probability is not None
    assert result.verdict in {"fake", "real"}


def test_an_image_drives_only_the_visual_stream(engine: MultimodalEngine) -> None:
    """Routing decides, not the views. An image has a visual_view and the
    emotion stream reads that same view, so without the routing table emotion
    would run on half its input."""
    result = engine.predict(Path("face.jpg"))

    assert set(result.branch_logits) == {"visual-efficientnet"}
    assert result.probability is not None


def test_audio_drives_only_the_audio_branch(engine: MultimodalEngine) -> None:
    result = engine.predict(Path("voice.wav"))

    assert set(result.branch_logits) == {"final-audio-seed17"}
    assert result.probability is not None


def test_each_kind_uses_its_own_entry_point(engine: MultimodalEngine) -> None:
    engine.predict(Path("clip.mp4"))
    engine.predict(Path("face.jpg"))
    engine.predict(Path("voice.wav"))

    assert engine.preprocessor.calls == ["video", "image", "audio"]


def test_an_unsupported_file_abstains_with_a_reason(engine: MultimodalEngine) -> None:
    result = engine.predict(Path("notes.txt"))

    assert result.verdict == "indeterminate"
    assert result.probability is None
    assert "unsupported_media_type" in result.blockers


def test_a_head_that_never_saw_a_stream_is_refused(tmp_path: Path) -> None:
    torch.manual_seed(17)
    head = StreamFusion(stream_dims={"visual-efficientnet": 4}, common_dim=8)
    path = tmp_path / "small.pt"
    torch.save(
        {
            "model": head.state_dict(),
            "stream_dims": {"visual-efficientnet": 4},
            "common_dim": 8,
        },
        path,
    )

    with pytest.raises(ValueError, match="not trained on"):
        MultimodalEngine(
            preprocessor=StubPreprocessor(),
            streams={
                "stream-emotion": StreamSpec(
                    "stream-emotion", Stub("cross"), "h", "emotion"
                )
            },
            fusion=load_fusion(path),
            device="cpu",
        )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("visual-dinov3", "visual"),
        ("visual-efficientnet", "visual"),
        ("stream-lipsync", "lipsync"),
        ("stream-emotion", "emotion"),
        ("final-audio-seed17", "audio"),
    ],
)
def test_stream_names_map_to_their_modality(name: str, expected: str) -> None:
    assert _modality(name) == expected
