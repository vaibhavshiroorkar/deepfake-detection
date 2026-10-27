"""AV-HuBERT without fairseq: the pretrained audio-visual encoder as plain torch.

Meta's release (facebookresearch/av_hubert) runs inside fairseq, and fairseq
pins a PyTorch too old for this GPU (see `streams/encoders.py`). The model
itself is small enough to restate:

    video  [B, 1, T, 88, 88] grey lips -> 3D conv + ResNet-18 -> [B, 512, T] -> 768
    audio  [B, 104, T] stacked fbank   -> linear                       -> 768
    concat [B, T, 1536] -> LayerNorm -> linear 768 -> positional conv
                        -> 12 pre-norm transformer layers -> LayerNorm

Each module mirrors its fairseq counterpart parameter for parameter, so the
released checkpoint loads with `strict=True` after one rename (the attention
projections, which torch stores stacked). `scripts/check_encoder_ports.py`
compares this module's output against the official code on the same inputs.

Either modality may be missing. AV-HuBERT was pretrained with modality dropout,
which replaces the absent stream's features with zeros, and the lip-sync stream
relies on exactly that: audio-only and video-only passes land in one feature
space, so the two can be compared frame by frame.

AV-HuBERT is licensed for non-commercial research only (the AV-HuBERT license,
Copyright (c) Meta Platforms, Inc.). The weights are not redistributed here.
"""

from __future__ import annotations

import math
import pickle
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

WIDTH = 768
LIP_SIZE = 88
# The grey-level statistics AV-HuBERT normalises lip crops with, from
# `avhubert/hubert_dataset.py`.
LIP_MEAN = 0.421
LIP_STD = 0.165
# python_speech_features.logfbank defaults, which AV-HuBERT uses unchanged: 26
# mel filters over 25 ms windows every 10 ms. Four frames are stacked into one
# 104-dim vector, so audio runs at 25 Hz, the video rate.
SAMPLE_RATE = 16_000
FILTERS = 26
STACK = 4
WINDOW = 400
STEP = 160
FFT = 512
PREEMPHASIS = 0.97


class _BasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.PReLU(num_parameters=planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.PReLU(num_parameters=planes)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu2(out + residual)


class _Trunk(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(_BasicBlock(64, 64), _BasicBlock(64, 64))
        self.layer2 = nn.Sequential(_BasicBlock(64, 128, 2), _BasicBlock(128, 128))
        self.layer3 = nn.Sequential(_BasicBlock(128, 256, 2), _BasicBlock(256, 256))
        self.layer4 = nn.Sequential(_BasicBlock(256, 512, 2), _BasicBlock(512, 512))

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


class _ResEncoder(nn.Module):
    """`avhubert/resnet.py::ResEncoder` with PReLU, as the release uses."""

    def __init__(self) -> None:
        super().__init__()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, (5, 7, 7), (1, 2, 2), (2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.PReLU(num_parameters=64),
            nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)),
        )
        self.trunk = _Trunk()

    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        x = self.frontend3D(x)
        time = x.shape[2]
        # Every frame through the 2D trunk on its own.
        x = x.transpose(1, 2).reshape(batch * time, x.shape[1], x.shape[3], x.shape[4])
        return self.trunk(x).view(batch, time, -1).transpose(1, 2)


class _SubModel(nn.Module):
    def __init__(self, input_dim: int, resnet: nn.Module | None = None) -> None:
        super().__init__()
        self.resnet = resnet
        self.proj = nn.Linear(input_dim, WIDTH)

    def forward(self, x: Tensor) -> Tensor:
        if self.resnet is not None:
            x = self.resnet(x)
        return self.proj(x.transpose(1, 2)).transpose(1, 2)


class _SamePadConv(nn.Module):
    """fairseq's positional convolution: weight-normed, grouped, one sample
    trimmed so an even kernel keeps the sequence length."""

    def __init__(self) -> None:
        super().__init__()
        conv = nn.Conv1d(WIDTH, WIDTH, kernel_size=128, padding=64, groups=16)
        self.conv = nn.utils.parametrizations.weight_norm(conv, name="weight", dim=2)

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(self.conv(x)[:, :, :-1])


class _Layer(nn.Module):
    """`TransformerSentenceEncoderLayer` with `layer_norm_first=True`."""

    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(WIDTH, 12, batch_first=True)
        self.self_attn_layer_norm = nn.LayerNorm(WIDTH)
        self.fc1 = nn.Linear(WIDTH, 3072)
        self.fc2 = nn.Linear(3072, WIDTH)
        self.final_layer_norm = nn.LayerNorm(WIDTH)

    def forward(self, x: Tensor, padding_mask: Tensor | None) -> Tensor:
        normed = self.self_attn_layer_norm(x)
        attended, _ = self.self_attn(
            normed, normed, normed, key_padding_mask=padding_mask, need_weights=False
        )
        x = x + attended
        return x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))


class AVHubert(nn.Module):
    """The AV-HuBERT Base encoder. `forward` returns the last layer, [B, T, 768]."""

    def __init__(self, layers: int = 12) -> None:
        super().__init__()
        self.feature_extractor_audio = _SubModel(FILTERS * STACK)
        self.feature_extractor_video = _SubModel(512, _ResEncoder())
        self.layer_norm = nn.LayerNorm(2 * WIDTH)
        self.post_extract_proj = nn.Linear(2 * WIDTH, WIDTH)
        self.pos_conv = _SamePadConv()
        self.layers = nn.ModuleList(_Layer() for _ in range(layers))
        self.encoder_layer_norm = nn.LayerNorm(WIDTH)

    def forward(
        self,
        *,
        video: Tensor | None = None,
        audio: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """`video` is [B, 1, T, 88, 88] normalised grey lips, `audio` is
        [B, 104, T] from `audio_features`. At least one must be given."""
        if video is None and audio is None:
            raise ValueError("AV-HuBERT needs audio, video or both")
        video_features = (
            self.feature_extractor_video(video) if video is not None else None
        )
        audio_features = (
            self.feature_extractor_audio(audio) if audio is not None else None
        )
        # Modality dropout at pretraining zeroed the missing stream's features,
        # so zeros are what the encoder expects in its place.
        if video_features is None:
            video_features = torch.zeros_like(audio_features)
        if audio_features is None:
            audio_features = torch.zeros_like(video_features)
        features = torch.cat([audio_features, video_features], dim=1).transpose(1, 2)
        x = self.post_extract_proj(self.layer_norm(features))
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        x = x + self.pos_conv(x.transpose(1, 2)).transpose(1, 2)
        for layer in self.layers:
            x = layer(x, padding_mask)
        return self.encoder_layer_norm(x)


def audio_features(waveform: Tensor) -> Tensor:
    """[B, samples] at 16 kHz to [B, 104, T] stacked, normalised log filterbanks.

    `python_speech_features.logfbank` restated in torch so it batches on the
    GPU; `tests/test_avhubert.py` holds it to the numpy original. Each stacked
    frame is then layer-normalised, as AV-HuBERT's dataset does, which also
    makes the result independent of the waveform's gain: the cache stores
    z-normalised audio, and a constant gain only shifts every log energy by the
    same amount. The one exception is a final step that holds zero padding,
    which the gain does not shift, as in the released loader.
    """
    if waveform.ndim != 2:
        raise ValueError("Waveform must be [batch, samples]")
    signal = torch.cat(
        [waveform[:, :1], waveform[:, 1:] - PREEMPHASIS * waveform[:, :-1]], dim=1
    )
    samples = signal.shape[1]
    frames = 1 if samples <= WINDOW else 1 + math.ceil((samples - WINDOW) / STEP)
    padded = F.pad(signal, (0, (frames - 1) * STEP + WINDOW - samples))
    windows = padded.unfold(1, WINDOW, STEP)
    power = torch.fft.rfft(windows, n=FFT).abs() ** 2 / FFT
    filters = torch.as_tensor(_mel_filterbank(), dtype=power.dtype, device=power.device)
    energy = power @ filters.T
    eps = torch.finfo(torch.float64).eps
    log_energy = torch.log(torch.where(energy == 0, eps, energy))
    # Stack four 10 ms frames into one 40 ms step, zero-padding the tail.
    remainder = (-log_energy.shape[1]) % STACK
    log_energy = F.pad(log_energy, (0, 0, 0, remainder))
    stacked = log_energy.reshape(log_energy.shape[0], -1, STACK * FILTERS)
    stacked = F.layer_norm(stacked, stacked.shape[-1:])
    return stacked.transpose(1, 2)


def _mel_filterbank() -> np.ndarray:
    """`python_speech_features.get_filterbanks(26, 512, 16000)`."""

    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700.0)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595.0) - 1)

    points = np.linspace(hz_to_mel(0), hz_to_mel(SAMPLE_RATE / 2), FILTERS + 2)
    bins = np.floor((FFT + 1) * mel_to_hz(points) / SAMPLE_RATE)
    bank = np.zeros([FILTERS, FFT // 2 + 1])
    for j in range(FILTERS):
        for i in range(int(bins[j]), int(bins[j + 1])):
            bank[j, i] = (i - bins[j]) / (bins[j + 1] - bins[j])
        for i in range(int(bins[j + 1]), int(bins[j + 2])):
            bank[j, i] = (bins[j + 2] - i) / (bins[j + 2] - bins[j + 1])
    return bank


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def lip_frames(mouth: Tensor, *, crop: float = 1.0, center_y: float = 0.5) -> Tensor:
    """Cached mouth crops [B, T, 3, H, W] to AV-HuBERT lips [B, 1, T, 88, 88].

    AV-HuBERT saw 96-pixel mouth regions cut from aligned faces and
    centre-cropped to 88. The cached sync view is looser, the lower face from
    the eyes to the chin. A square of `crop` times the view's width, centred at
    `center_y` of its height, is resized to 96 and centre-cropped to 88.

    The default is the whole view. `scripts/check_encoder_ports.py tune-crop`
    compared the whole view with nine tighter mouth crops on 128 genuine
    FakeAVCeleb training clips, by how often aligned audio agrees with the lips
    better than audio shifted by half a second. The whole view won, 98% of
    clips against 66% to 95%.
    """
    if mouth.ndim != 5 or mouth.shape[2] != 3:
        raise ValueError("Mouth frames must be [batch, time, 3, height, width]")
    batch, time, _, height, width = mouth.shape
    mean = mouth.new_tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    std = mouth.new_tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
    rgb = (mouth * std + mean).clamp(0, 1)
    # cv2.COLOR_BGR2GRAY weights, which AV-HuBERT's loader used.
    grey = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    side = round(crop * width)
    top = min(max(round(center_y * height - side / 2), 0), height - side)
    left = (width - side) // 2
    region = grey[:, :, top : top + side, left : left + side]
    resized = F.interpolate(
        region.reshape(batch * time, 1, side, side),
        size=(96, 96),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    offset = (96 - LIP_SIZE) // 2
    lips = resized[:, :, offset : offset + LIP_SIZE, offset : offset + LIP_SIZE]
    lips = (lips - LIP_MEAN) / LIP_STD
    return lips.reshape(batch, time, LIP_SIZE, LIP_SIZE).unsqueeze(1)


class _FairseqStub:
    """Stands in for the vocabulary objects the checkpoint pickles."""

    def __setstate__(self, state) -> None:
        self.__dict__.update(state if isinstance(state, dict) else {})


class _Unpickler(pickle.Unpickler):
    # Only fairseq's own classes are replaced. Everything else resolves
    # normally, so a checkpoint asking for anything unexpected still fails.
    def find_class(self, module: str, name: str):
        if module.startswith("fairseq."):
            return _FairseqStub
        return super().find_class(module, name)


def _rename(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """The released state dict, renamed onto this module."""
    out: dict[str, Tensor] = {}
    for key, value in state.items():
        if key.startswith(("feature_extractor_", "layer_norm.", "post_extract_proj.")):
            out[key] = value
        elif key.startswith("encoder.pos_conv.0."):
            suffix = key.removeprefix("encoder.pos_conv.0.")
            out[
                {
                    "bias": "pos_conv.conv.bias",
                    "weight_g": "pos_conv.conv.parametrizations.weight.original0",
                    "weight_v": "pos_conv.conv.parametrizations.weight.original1",
                }[suffix]
            ] = value
        elif key.startswith("encoder.layer_norm."):
            out[key.replace("encoder.layer_norm.", "encoder_layer_norm.")] = value
        elif key.startswith("encoder.layers."):
            out[key.removeprefix("encoder.")] = value
    # torch keeps query, key and value as one stacked projection.
    for index in range(sum(1 for k in out if k.endswith("self_attn.q_proj.weight"))):
        prefix = f"layers.{index}.self_attn."
        for part in ("weight", "bias"):
            out[f"{prefix}in_proj_{part}"] = torch.cat(
                [out.pop(f"{prefix}{name}_proj.{part}") for name in ("q", "k", "v")]
            )
        out[f"{prefix}out_proj.weight"] = out.pop(f"{prefix}out_proj.weight")
        out[f"{prefix}out_proj.bias"] = out.pop(f"{prefix}out_proj.bias")
    return out


def load_avhubert(path: Path) -> AVHubert:
    """The released pretraining checkpoint as an `AVHubert`, in eval mode."""
    pickle_module = types.ModuleType("avhubert_pickle")
    pickle_module.Unpickler = _Unpickler
    pickle_module.load = pickle.load
    checkpoint = torch.load(
        path, map_location="cpu", pickle_module=pickle_module, weights_only=False
    )
    model_cfg = checkpoint["cfg"]["model"]
    if model_cfg.get("modality_fuse") != "concat" or not model_cfg.get(
        "layer_norm_first"
    ):
        raise ValueError(f"{path} is not an AV-HuBERT Base pretraining checkpoint")
    layers = model_cfg.get("encoder_layers", 12)
    model = AVHubert(layers=layers)
    model.load_state_dict(_rename(checkpoint["model"]), strict=True)
    return model.eval()
