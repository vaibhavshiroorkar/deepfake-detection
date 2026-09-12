"""Loading a trained fusion head and scoring a partial set of streams.

The deep path could train and never predict: `_deep_fusion` saved a checkpoint
and nothing read it back. These tests pin the round trip, and the two cases that
matter at inference are a clip that drove only some streams and a clip that
drove none.
"""

from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch

from deepfake_detection.fusion.deep import StreamFusion
from deepfake_detection.fusion.deep_loading import fuse, load_fusion

DIMS = {"visual": 6, "audio": 4, "lipsync": 5}


def save(path: Path, dims=None) -> Path:
    dims = DIMS if dims is None else dims
    torch.manual_seed(17)
    model = StreamFusion(stream_dims=dims, common_dim=8, hidden_sizes=(4,))
    torch.save(
        {
            "model": model.state_dict(),
            "stream_dims": dims,
            "common_dim": 8,
            "hidden_sizes": [4],
            "split_hash": "split",
            "preprocessing_hash": "prep",
        },
        path,
    )
    return path


def test_round_trip_rebuilds_the_head_it_was_trained_as(tmp_path: Path) -> None:
    loaded = load_fusion(save(tmp_path / "fusion.pt"))

    assert loaded.stream_dims == DIMS
    assert loaded.streams == ("audio", "lipsync", "visual")
    assert loaded.split_hash == "split"
    assert loaded.preprocessing_hash == "prep"


def test_a_video_uses_every_stream(tmp_path: Path) -> None:
    loaded = load_fusion(save(tmp_path / "fusion.pt"))

    probability = fuse(
        loaded, {name: tuple([0.1] * width) for name, width in DIMS.items()}
    )

    assert 0.0 <= probability <= 1.0


def test_an_image_supplies_only_the_visual_stream(tmp_path: Path) -> None:
    """The whole point of the presence flags: a partial input still scores."""
    loaded = load_fusion(save(tmp_path / "fusion.pt"))

    probability = fuse(loaded, {"visual": tuple([0.1] * 6)})

    assert 0.0 <= probability <= 1.0


def test_an_absent_stream_cannot_change_the_answer(tmp_path: Path) -> None:
    """Masking happens after the projection, so an omitted stream contributes
    an exact zero including the bias. Omitting it and passing an empty tuple
    must agree."""
    loaded = load_fusion(save(tmp_path / "fusion.pt"))
    visual = tuple([0.1] * 6)

    omitted = fuse(loaded, {"visual": visual})
    empty = fuse(loaded, {"visual": visual, "audio": (), "lipsync": ()})

    assert omitted == pytest.approx(empty)


def test_nothing_running_is_an_error_not_a_number(tmp_path: Path) -> None:
    """A head handed an all-absent input returns whatever its biases encode.
    That is a number with no evidence behind it, so the caller must abstain."""
    loaded = load_fusion(save(tmp_path / "fusion.pt"))

    with pytest.raises(ValueError, match="nothing to fuse"):
        fuse(loaded, {"visual": (), "audio": ()})


def test_a_stream_the_head_never_saw_is_refused(tmp_path: Path) -> None:
    loaded = load_fusion(save(tmp_path / "fusion.pt"))

    with pytest.raises(ValueError, match="was not trained on"):
        fuse(loaded, {"emotion": (0.1, 0.2)})


def test_a_wrong_width_is_refused_rather_than_truncated(tmp_path: Path) -> None:
    """Two checkpoints exported into one store would silently disagree here."""
    loaded = load_fusion(save(tmp_path / "fusion.pt"))

    with pytest.raises(ValueError, match="dimensions, head expects"):
        fuse(loaded, {"visual": (0.1, 0.2)})


def test_a_payload_that_is_not_a_fusion_checkpoint_says_so(tmp_path: Path) -> None:
    path = tmp_path / "other.pt"
    torch.save({"model": {}}, path)

    with pytest.raises(ValueError, match="not a deep fusion checkpoint"):
        load_fusion(path)
