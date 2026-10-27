from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from deepfake_detection.digest import file_sha256


@dataclass(frozen=True, slots=True)
class RunMetadata:
    run_id: str
    branch: str
    git_commit: str
    split_hash: str
    preprocessing_hash: str
    config_hash: str
    seed: int
    trained_source_identities: tuple[str, ...] = ()
    # None identifies legacy checkpoints with no model-selection provenance.
    selection_source_identities: tuple[str, ...] | None = None
    # The backbone or audio model this checkpoint was trained as, so loaders
    # rebuild the right model (`branches/registry.py`). Empty in checkpoints
    # written before the field existed; the registry falls back to the name.
    architecture: str = ""
    # Missing in historical files, whose forwards never used padding masks.
    masking_mode: str = "legacy_none"

    def __post_init__(self) -> None:
        from .input_contract import MASKING_MODES

        if self.masking_mode not in MASKING_MODES:
            raise ValueError(f"Unknown checkpoint masking mode: {self.masking_mode}")


@dataclass(frozen=True, slots=True)
class CheckpointState:
    metadata: RunMetadata
    epoch: int


@dataclass(frozen=True, slots=True)
class BranchProvenance:
    split_hash: str
    preprocessing_hash: str


def validate_branch_states(
    states: Mapping[str, CheckpointState],
) -> BranchProvenance:
    if not states:
        raise ValueError("At least one branch checkpoint is required")
    for branch, state in states.items():
        if state.metadata.branch != branch:
            raise ValueError(
                f"{branch} checkpoint contains {state.metadata.branch} metadata"
            )
    split_hashes = {state.metadata.split_hash for state in states.values()}
    if len(split_hashes) != 1:
        raise ValueError("Branch checkpoints use different split hashes")
    preprocessing_hashes = {
        state.metadata.preprocessing_hash for state in states.values()
    }
    if len(preprocessing_hashes) != 1:
        raise ValueError("Branch checkpoints use different preprocessing hashes")
    return BranchProvenance(
        split_hash=split_hashes.pop(),
        preprocessing_hash=preprocessing_hashes.pop(),
    )


def hash_config(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    metadata: RunMetadata,
    epoch: int,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            suffix=".pt",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metadata": asdict(metadata),
                "epoch": epoch,
            },
            temporary,
        )
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return file_sha256(path)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
) -> CheckpointState:
    # Without an optimizer to restore, map the file instead of reading it. A
    # checkpoint is about two thirds AdamW state, 1.1 GB for the audio branch,
    # and a full read commits all of it although only the weights are used.
    # On Windows that commit charge is what has run this host out of memory.
    payload = torch.load(
        path, map_location="cpu", weights_only=True, mmap=optimizer is None
    )
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    metadata = _metadata(payload["metadata"])
    model.masking_mode = metadata.masking_mode
    model.trained_source_identities = metadata.trained_source_identities
    model.selection_source_identities = metadata.selection_source_identities
    return CheckpointState(
        metadata=metadata,
        epoch=int(payload["epoch"]),
    )


def read_metadata(path: Path) -> RunMetadata:
    """The run metadata alone, without building a model. The file is mapped."""
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return _metadata(payload["metadata"])


def _metadata(stored: Mapping[str, Any]) -> RunMetadata:
    metadata = dict(stored)
    # `asdict` writes the tuple as a list, and a checkpoint written before the
    # field existed has no key at all. Normalise both so callers always get a
    # tuple and the dataclass default covers the old files.
    metadata["trained_source_identities"] = tuple(
        metadata.get("trained_source_identities") or ()
    )
    selected = metadata.get("selection_source_identities")
    metadata["selection_source_identities"] = (
        tuple(selected) if selected is not None else None
    )
    return RunMetadata(**metadata)
