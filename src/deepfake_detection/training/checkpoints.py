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


@dataclass(frozen=True, slots=True)
class RunMetadata:
    run_id: str
    branch: str
    git_commit: str
    split_hash: str
    preprocessing_hash: str
    config_hash: str
    seed: int


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


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    return _file_hash(path)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
) -> CheckpointState:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return CheckpointState(
        metadata=RunMetadata(**payload["metadata"]),
        epoch=int(payload["epoch"]),
    )
