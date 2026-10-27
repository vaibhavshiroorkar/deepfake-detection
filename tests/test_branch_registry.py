from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from deepfake_detection.branches.registry import (  # noqa: E402
    DEFAULT_AUDIO_MODEL,
    WAVLM_AUDIO_MODEL,
    architecture_of,
    build_branch,
    load_branch,
)
from deepfake_detection.training.checkpoints import (  # noqa: E402
    RunMetadata,
    read_metadata,
    save_checkpoint,
)


def _metadata(branch: str, architecture: str = "") -> RunMetadata:
    return RunMetadata(
        run_id="run",
        branch=branch,
        git_commit="commit",
        split_hash="split",
        preprocessing_hash="views",
        config_hash="config",
        seed=17,
        architecture=architecture,
    )


def test_a_recorded_architecture_beats_the_file_name() -> None:
    assert architecture_of(Path("final-visual.pt"), "visual", "convnext-tiny") == (
        "convnext-tiny"
    )


def test_old_checkpoints_fall_back_to_the_file_name() -> None:
    assert architecture_of(Path("final-visual-convnext-seed43.pt"), "visual") == (
        "convnext-tiny"
    )
    assert architecture_of(Path("final-visual-seed17.pt"), "visual") == (
        "efficientnet-b0"
    )
    assert architecture_of(Path("final-audio-wavlm-seed17.pt"), "audio") == (
        WAVLM_AUDIO_MODEL
    )
    assert architecture_of(Path("final-sync-seed17.pt"), "sync") == (
        DEFAULT_AUDIO_MODEL
    )


def test_a_convnext_checkpoint_loads_as_convnext_whatever_its_name(
    tmp_path: Path,
) -> None:
    model = build_branch("visual", "convnext-tiny")
    path = tmp_path / "unhelpful-name.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        metadata=_metadata("visual", "convnext-tiny"),
        epoch=1,
    )

    loaded, state = load_branch(path, "visual", fallback="efficientnet-b0")

    assert read_metadata(path).architecture == "convnext-tiny"
    assert state.metadata.architecture == "convnext-tiny"
    assert type(loaded.backbone).__name__ == "ConvNeXt"


def test_loading_the_wrong_branch_is_refused(tmp_path: Path) -> None:
    model = build_branch("visual", "efficientnet-b0")
    path = tmp_path / "visual.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        metadata=_metadata("visual", "efficientnet-b0"),
        epoch=1,
    )

    with pytest.raises(ValueError, match="is a visual checkpoint, not audio"):
        load_branch(path, "audio")
