"""The face heatmaps: Grad-CAM on the face model, the cited frames, and the words for where it looked."""

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from deepfake_detection.webapp import explain


class BottomLooker(nn.Module):
    """A face model whose "fake" logit grows with the brightness of the lower third of each frame."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Conv2d(3, 1, 1, bias=False)
        nn.init.ones_(self.backbone.weight)

    def forward(self, frames):
        batch, time, channels, height, width = frames.shape
        maps = self.backbone(frames.reshape(batch * time, channels, height, width))
        mask = torch.zeros(height, width)
        mask[2 * height // 3 :, :] = 1.0
        return SimpleNamespace(
            logits=(maps * mask).mean(dim=(1, 2, 3)).reshape(batch, time).sum(1)
        )


def _frames(bright: int, count: int = 4, size: int = 24) -> torch.Tensor:
    frames = torch.zeros(1, count, 3, size, size)
    for t in range(count):
        frames[0, t, :, 2 * size // 3 :, size // 3 : 2 * size // 3] = 0.5
    frames[0, bright, :, 2 * size // 3 :, size // 3 : 2 * size // 3] = 2.0
    return frames


def test_the_heatmap_lights_the_part_of_the_face_the_model_used():
    cams, weights = explain.grad_cam([BottomLooker()], _frames(bright=2))
    assert cams.shape == (4, 24, 24)
    peak = np.unravel_index(cams[2].argmax(), cams[2].shape)
    assert peak[0] >= 16  # lower third
    assert explain.region(cams[2]) == "mouth and jaw"
    assert int(np.argmax(weights)) == 2


def test_cited_frames_carry_their_time_in_the_upload_and_a_picture():
    found = explain.explain_face(
        [BottomLooker()],
        _frames(bright=1),
        times=[0.25, 1.0, 1.75, 2.5],
        offset=6.0,
        top=2,
    )
    assert found["frames"][0]["time"] == 7.0  # 1.0 s into the part that starts at 6 s
    assert found["frames"][0]["image"].startswith("data:image/jpeg;base64,")
    assert found["frames"][0]["region"] == "mouth and jaw"
    assert len(found["frames"]) == 2
    assert "mouth and jaw" in found["summary"] and "not proof" in found["summary"]


def test_regions_are_named_by_where_the_heat_is():
    cam = np.zeros((30, 30))
    cam[5:9, 13:17] = 1.0
    assert explain.region(cam) == "eyes and brow"
    cam = np.zeros((30, 30))
    cam[13:17, 13:17] = 1.0
    assert explain.region(cam) == "nose and cheeks"
    cam = np.zeros((30, 30))
    cam[13:17, 0:3] = 1.0
    assert explain.region(cam) == "edge of the face"
