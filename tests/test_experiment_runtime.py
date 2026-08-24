import json
from pathlib import Path

import numpy as np

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
