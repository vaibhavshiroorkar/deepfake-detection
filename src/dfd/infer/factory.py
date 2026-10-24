"""Choose an adapter by name. There is no automatic fallback from real to mock: a missing checkpoint raises."""
from __future__ import annotations

from dfd.infer.adapters import Adapter


def make_adapter(kind: str, checkpoint_dir: str | None = None) -> Adapter:
    if kind == "mock":
        from dfd.infer.mock_adapter import MockAdapter
        return MockAdapter()
    if kind == "real":
        from dfd.infer.real_adapter import RealAdapter
        return RealAdapter(checkpoint_dir)  # raises NoCheckpointError until a checkpoint exists
    raise ValueError(f"unknown adapter {kind!r}; expected 'mock' or 'real'")
