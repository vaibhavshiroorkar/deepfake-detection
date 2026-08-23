from pathlib import Path

import pytest
import torch
from torch import nn

from deepfake_detection.training.checkpoints import (
    CheckpointState,
    RunMetadata,
    load_checkpoint,
    save_checkpoint,
    validate_branch_states,
)


def test_checkpoint_round_trip_keeps_model_and_research_provenance(
    tmp_path: Path,
) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    metadata = RunMetadata(
        run_id="run-1",
        branch="visual",
        git_commit="abc123",
        split_hash="split123",
        preprocessing_hash="prep123",
        config_hash="config123",
        seed=17,
    )
    path = tmp_path / "checkpoint.pt"

    digest = save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        metadata=metadata,
        epoch=3,
    )
    restored_model = nn.Linear(2, 1)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1)
    restored = load_checkpoint(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
    )

    assert len(digest) == 64
    assert restored.metadata == metadata
    assert restored.epoch == 3
    for expected, actual in zip(
        model.parameters(), restored_model.parameters(), strict=True
    ):
        assert torch.equal(expected, actual)


def test_branch_provenance_rejects_mixed_preprocessing_hashes() -> None:
    states = {
        branch: CheckpointState(
            metadata=RunMetadata(
                run_id=f"run-{branch}",
                branch=branch,
                git_commit="abc123",
                split_hash="split123",
                preprocessing_hash=("prep-other" if branch == "audio" else "prep123"),
                config_hash="config123",
                seed=17,
            ),
            epoch=1,
        )
        for branch in ("visual", "audio", "sync")
    }

    with pytest.raises(ValueError, match="preprocessing hashes"):
        validate_branch_states(states)
