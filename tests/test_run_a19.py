import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    "run_a19", Path(__file__).parents[1] / "scripts" / "run_a19.py"
)
a19 = importlib.util.module_from_spec(spec)
sys.modules["run_a19"] = a19
spec.loader.exec_module(a19)


def test_audio_windows_centre_on_the_frame_and_pad_with_zeros() -> None:
    audio = torch.arange(1, 5 * 3 + 1, dtype=torch.float32).reshape(
        1, 5, 3
    )  # 5 frames, width 3
    got = a19.audio_windows(audio, torch.tensor([0, 0]), torch.tensor([0, 4]))
    assert got.shape == (2, 2 * a19.TAU + 1, 3)
    assert torch.equal(got[0, a19.TAU], audio[0, 0])  # the centre is the frame itself
    assert torch.equal(got[1, a19.TAU], audio[0, 4])
    assert torch.equal(got[0, a19.TAU - 1], torch.zeros(3))  # before the window: zeros
    assert torch.equal(got[1, a19.TAU + 1], torch.zeros(3))  # after the window: zeros
    assert torch.equal(got[0, a19.TAU + 2], audio[0, 2])


def test_normalize_gives_unit_length_frames() -> None:
    x = a19.normalize(np.array([[[3.0, 4.0], [0.0, 2.0]]], dtype=np.float16))
    assert torch.allclose(torch.linalg.norm(x, dim=-1), torch.ones(1, 2))


def test_window_score_is_logsumexp_of_minus_the_aligned_output() -> None:
    class Fixed(torch.nn.Module):
        def forward(self, visual, audio):
            return visual[..., :1]  # output = first feature of the lips frame

    lips = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])
    score = a19.window_scores(Fixed(), lips, lips)
    assert np.allclose(score, [np.log(np.exp(-1.0) + np.exp(-2.0))])


def test_training_lowers_the_loss_on_a_learnable_toy() -> None:
    torch.manual_seed(0)
    lips = a19.normalize(
        np.random.default_rng(0).normal(size=(4, 20, a19.WIDTH)).astype(np.float16)
    )
    model = a19.FusionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = a19.epoch_loss(model, lips, lips)
    for _ in range(30):
        a19.epoch_loss(model, lips, lips, optimizer)
    assert a19.epoch_loss(model, lips, lips) < before
