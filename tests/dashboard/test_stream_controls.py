"""Unit tests for the pure/testable pieces behind the Train and Run tabs."""
import numpy as np

from dashboard.lib.stream_ui import DATA_DEFAULTS, TRAIN_DEFAULTS, train_command
from dashboard.lib.inference import frames_to_tensor


def test_train_command_includes_hyperparameters():
    cmd = train_command("efficientnet", "tf_efficientnet_b0.ns_jft_in1k", TRAIN_DEFAULTS)
    assert cmd.startswith("uv run python -m training.train_visual")
    assert "--stream efficientnet" in cmd
    assert "--backbone tf_efficientnet_b0.ns_jft_in1k" in cmd
    assert "--epochs 8" in cmd
    assert "--lr-backbone 5e-06" in cmd     # scientific formatting, no trailing zeros
    assert "--seed 42" in cmd


def test_train_command_carries_the_chosen_dataset_and_splits():
    """What to train on is a training decision, so it must be in the command."""
    settings = {**TRAIN_DEFAULTS, "dataset": "FakeAVCeleb_v1.2",
                "train_split": "train", "val_split": "val"}
    cmd = train_command("xception", "legacy_xception", settings)
    assert "--dataset FakeAVCeleb_v1.2" in cmd
    assert "--train-split train" in cmd
    assert "--val-split val" in cmd
    # data flags precede the hyperparameters, so the command reads in that order
    assert cmd.index("--dataset") < cmd.index("--epochs")


def test_train_command_omits_data_flags_when_no_dataset_chosen():
    """No dataset discovered is not the same as an empty --dataset argument."""
    cmd = train_command("xception", "legacy_xception", {**TRAIN_DEFAULTS, **DATA_DEFAULTS})
    assert "--dataset" not in cmd
    assert "--train-split" not in cmd
    assert "--epochs 8" in cmd


def test_frames_to_tensor_shape_and_normalization():
    torch = __import__("torch")
    faces = [np.full((224, 224, 3), 128, np.uint8) for _ in range(16)]
    t = frames_to_tensor(faces)
    assert t.shape == (1, 16, 3, 224, 224)
    assert t.dtype == torch.float32
    # 128/255 normalized by ImageNet stats lands roughly in [-2, 2], not [0, 255]
    assert float(t.abs().max()) < 3.0
