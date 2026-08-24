import json
from pathlib import Path

import numpy as np
import pytest

from deepfake_detection.experiments import runtime
from deepfake_detection.experiments.runtime import capture_runtime, seed_everything


def test_runtime_snapshot_captures_identity_and_software() -> None:
    snapshot = capture_runtime(Path.cwd())

    assert snapshot.started_at_utc.endswith("+00:00")
    assert snapshot.git_commit
    assert isinstance(snapshot.git_dirty, bool)
    assert snapshot.python_version
    assert snapshot.platform
    assert "scikit-learn" in snapshot.packages
    assert snapshot.cpu
    assert json.loads(json.dumps(snapshot.as_dict())) == snapshot.as_dict()


def test_seed_everything_repeats_numpy_values() -> None:
    seed_everything(23, deterministic=True)
    first = np.random.random(4)
    seed_everything(23, deterministic=True)
    second = np.random.random(4)

    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("values", [(0, 10), (-1, 10), (4096, -1)])
def test_available_memory_ignores_unavailable_posix_sysconf_values(
    values: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    sysconf_values = iter(values)
    monkeypatch.setattr(runtime.os, "name", "posix")
    monkeypatch.setattr(
        runtime.os, "sysconf", lambda _: next(sysconf_values), raising=False
    )

    assert runtime._available_memory_mib() is None
