"""The real adapter: a stub that fails loudly until a checkpoint exists.

It never returns a number. Constructing it without a checkpoint folder holding ``metadata.json`` (D-028) raises
``NoCheckpointError``; even with a checkpoint it raises ``NotImplementedError`` when asked to score, because the
scoring path (frozen encoder, linear head, temperature scaling, preprocess_id check) is built with the MVP (D-040).
There is no fallback to the mock: a missing model is an error, not a reason to show a made-up result.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from dfd.infer.adapters import AdapterResult, Media

INFERENCE_SOURCE = "real"


class NoCheckpointError(RuntimeError):
    pass


class RealAdapter:
    inference_source = INFERENCE_SOURCE

    def __init__(self, checkpoint_dir: str | Path | None = None):
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir is None or not (self.checkpoint_dir / "metadata.json").is_file():
            raise NoCheckpointError(
                f"no checkpoint: {self.checkpoint_dir!s} has no metadata.json. The real adapter cannot run until the MVP "
                "checkpoint (D-040) exists; use --adapter mock only for interface development, its results are not real.")

    def analyse(self, media: Media, progress: Optional[Callable[[str], None]] = None) -> AdapterResult:
        raise NotImplementedError("real scoring is not implemented yet: it is built with the MVP checkpoint (D-040)")
