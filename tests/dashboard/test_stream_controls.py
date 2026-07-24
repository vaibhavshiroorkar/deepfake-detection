"""Unit tests for the pure/testable pieces behind the Train and Run tabs."""
import numpy as np

from dashboard.lib.stream_ui import TRAIN_DEFAULTS, train_command
from dashboard.lib.inference import frames_to_tensor


def test_train_command_includes_hyperparameters():
    cmd = train_command("efficientnet", "tf_efficientnet_b0.ns_jft_in1k", TRAIN_DEFAULTS)
    assert cmd.startswith("uv run python -m training.train_visual")
    assert "--stream efficientnet" in cmd
    assert "--backbone tf_efficientnet_b0.ns_jft_in1k" in cmd
    assert "--epochs 8" in cmd
    assert "--lr-backbone 5e-06" in cmd     # scientific formatting, no trailing zeros
    assert "--seed 42" in cmd


def test_frames_to_tensor_shape_and_normalization():
    torch = __import__("torch")
    faces = [np.full((224, 224, 3), 128, np.uint8) for _ in range(16)]
    t = frames_to_tensor(faces)
    assert t.shape == (1, 16, 3, 224, 224)
    assert t.dtype == torch.float32
    # 128/255 normalized by ImageNet stats lands roughly in [-2, 2], not [0, 255]
    assert float(t.abs().max()) < 3.0
