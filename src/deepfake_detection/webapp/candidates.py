"""Third-party detectors tried as the website's v2 face and voice models (branch web-models-v2).

Each takes exactly what the website's own models take from a cleaned 3-second part, so a comparison
on the same items is fair:
  - face: the part's face crops, square with a margin, ImageNet-normalized, shape [frames, 3, H, W];
  - voice: the part's waveform at 16 kHz, zero mean and unit variance, with its valid length.
Each returns the probability that the input is fake.

Effort (Yan et al., ICML 2025; CC BY-NC 4.0) is CLIP ViT-L/14 whose attention projections carry a
trained low-rank residual on top of the frozen weights. At inference the residual is folded into the
weights, so a plain CLIP vision model runs it.

AntiDeepfake (NII Yamagishi lab; CC BY-NC-SA 4.0) is a wav2vec 2.0 / XLS-R model post-trained on 30
speech anti-spoofing corpora, mean-pooled, with a two-way linear head (index 0 is fake).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
CLIP_MEAN = torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1)
CLIP_STD = torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verified(path: Path, expected: str) -> Path:
    """`path`, after checking its hash: a changed download is refused before it is read."""
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{path.name}: sha256 {actual} is not the pinned {expected}")
    return path


def fold_effort_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Effort's backbone weights as a plain CLIP vision model's: weight_main plus the residual."""
    out = {}
    for key, value in state.items():
        if not key.startswith("backbone."):
            continue
        name = key[len("backbone.") :]
        if name.endswith(".weight_main"):
            base = name[: -len(".weight_main")]
            u = state[f"backbone.{base}.U_residual"]
            s = state[f"backbone.{base}.S_residual"]
            v = state[f"backbone.{base}.V_residual"]
            out[f"{base}.weight"] = value + u @ torch.diag(s) @ v
        elif any(
            name.endswith(part)
            for part in (
                ".U_residual",
                ".S_residual",
                ".V_residual",
                ".U_r",
                ".S_r",
                ".V_r",
            )
        ):
            continue
        else:
            out[name] = value
    return out


class EffortFace(torch.nn.Module):
    def __init__(self, clip_dir: Path, checkpoint: Path) -> None:
        super().__init__()
        from transformers import CLIPVisionConfig, CLIPVisionModel

        self.backbone = CLIPVisionModel(
            CLIPVisionConfig.from_pretrained(clip_dir)
        ).vision_model
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        missing, unexpected = self.backbone.load_state_dict(
            fold_effort_state(state), strict=False
        )
        missing = [m for m in missing if "position_ids" not in m]
        if missing or unexpected:
            raise RuntimeError(
                f"Effort weights do not fit CLIP: missing {missing[:3]}, unexpected {unexpected[:3]}"
            )
        self.head = torch.nn.Linear(1024, 2)
        self.head.load_state_dict(
            {"weight": state["head.weight"], "bias": state["head.bias"]}
        )

    @torch.inference_mode()
    def forward(self, crops: torch.Tensor) -> float:
        """Mean fake probability over the part's face crops ([frames, 3, H, W], ImageNet-normalized)."""
        device = self.head.weight.device
        rgb = crops.to(device) * IMAGENET_STD.to(device) + IMAGENET_MEAN.to(device)
        rgb = F.interpolate(
            rgb.clamp(0, 1), size=(224, 224), mode="bilinear", align_corners=False
        )
        x = (rgb - CLIP_MEAN.to(device)) / CLIP_STD.to(device)
        pooled = self.backbone(pixel_values=x).pooler_output
        return float(torch.softmax(self.head(pooled), dim=1)[:, 1].mean())


def _hf_wav2vec2_key(name: str) -> str | None:
    """A fairseq wav2vec 2.0 weight name as transformers' Wav2Vec2Model names it; None to drop it."""
    import re

    if name.startswith(("final_proj.", "project_q.", "quantizer.")):
        return None  # pre-training heads, unused for features
    rules = [
        (
            r"^feature_extractor\.conv_layers\.(\d+)\.0\.",
            r"feature_extractor.conv_layers.\1.conv.",
        ),
        (
            r"^feature_extractor\.conv_layers\.(\d+)\.2\.1\.",
            r"feature_extractor.conv_layers.\1.layer_norm.",
        ),
        (r"^post_extract_proj\.", "feature_projection.projection."),
        (r"^layer_norm\.", "feature_projection.layer_norm."),
        (
            r"^encoder\.pos_conv\.0\.weight_g$",
            "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
        ),
        (
            r"^encoder\.pos_conv\.0\.weight_v$",
            "encoder.pos_conv_embed.conv.parametrizations.weight.original1",
        ),
        (r"^encoder\.pos_conv\.0\.bias$", "encoder.pos_conv_embed.conv.bias"),
        (r"\.self_attn\.", ".attention."),
        (r"\.self_attn_layer_norm\.", ".layer_norm."),
        (r"\.fc1\.", ".feed_forward.intermediate_dense."),
        (r"\.fc2\.", ".feed_forward.output_dense."),
        (r"^mask_emb$", "masked_spec_embed"),
    ]
    for pattern, replacement in rules:
        name = re.sub(pattern, replacement, name)
    return name


class AntiDeepfakeVoice(torch.nn.Module):
    def __init__(self, model_dir: Path) -> None:
        super().__init__()
        from safetensors.torch import load_file
        from transformers import Wav2Vec2Config, Wav2Vec2Model

        config = Wav2Vec2Config.from_pretrained(model_dir)
        self.ssl = Wav2Vec2Model(config)
        state = load_file(str(model_dir / "model.safetensors"))
        mapped = {}
        for key, value in state.items():
            if not key.startswith("m_ssl.model."):
                continue
            name = _hf_wav2vec2_key(key[len("m_ssl.model.") :])
            if name is not None:
                mapped[name] = value
        missing, unexpected = self.ssl.load_state_dict(mapped, strict=False)
        missing = [m for m in missing if m != "masked_spec_embed"]
        if missing or unexpected:
            raise RuntimeError(
                f"AntiDeepfake weights do not fit: missing {missing[:4]}, unexpected {unexpected[:4]}"
            )
        self.head = torch.nn.Linear(config.hidden_size, 2)
        self.head.load_state_dict(
            {"weight": state["proj_fc.weight"], "bias": state["proj_fc.bias"]}
        )

    @torch.inference_mode()
    def forward(self, wave: torch.Tensor) -> float:
        """Fake probability of a 16 kHz waveform [samples]; normalized like the model card (layer norm)."""
        device = self.head.weight.device
        x = wave.to(device).float().reshape(1, -1)
        x = F.layer_norm(x, x.shape)
        hidden = self.ssl(x).last_hidden_state.mean(dim=1)
        return float(torch.softmax(self.head(hidden), dim=1)[0, 0])


class ClipLnFace(torch.nn.Module):
    """Yermakov, Cech and Matas (2025), CLIP ViT-L/14 with LN-tuning, trained on FF++; MIT licence.

    Shipped as TorchScript (no pickled Python objects). Outputs [real, fake] logits; input is a
    CLIPProcessor image: 224 x 224, bicubic, CLIP-normalized.
    """

    def __init__(self, path: Path, device: str) -> None:
        super().__init__()
        self.device = device
        self.dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = (
            torch.jit.load(str(path), map_location=device).eval().to(self.dtype)
        )

    def logits(self, crops: torch.Tensor) -> torch.Tensor:
        """[real, fake] logits per crop; differentiable, so a heatmap can take the input gradient."""
        rgb = crops.to(self.device).float() * IMAGENET_STD.to(
            self.device
        ) + IMAGENET_MEAN.to(self.device)
        rgb = F.interpolate(
            rgb.clamp(0, 1), size=(224, 224), mode="bicubic", align_corners=False
        ).clamp(0, 1)
        x = ((rgb - CLIP_MEAN.to(self.device)) / CLIP_STD.to(self.device)).to(
            self.dtype
        )
        return self.model(x).float()

    @torch.inference_mode()
    def forward(self, crops: torch.Tensor) -> float:
        return float(torch.softmax(self.logits(crops), dim=1)[:, 1].mean())

    def saliency(self, frames: torch.Tensor) -> tuple:
        """Heatmaps (T, H, W) scaled to 0..1 and each frame's share of the evidence, like grad_cam.

        A ViT has no last convolution, so the map is the input gradient of the fake-minus-real logit
        times the input, positive part, summed over colour and smoothed: the pixels that pushed the
        score toward fake. `frames` is (1, T, 3, H, W) as the website scores it.
        """
        import numpy as np

        crops = frames[0].detach().to(self.device).float().requires_grad_(True)
        with torch.enable_grad():
            out = self.logits(crops)
            margin = (out[:, 1] - out[:, 0]).sum()
            (grad,) = torch.autograd.grad(margin, crops)
        maps = F.relu((grad * crops).sum(dim=1, keepdim=True))
        maps = F.avg_pool2d(maps, kernel_size=15, stride=1, padding=7)[:, 0]
        cams = maps.detach().cpu().numpy().astype(np.float32)
        share = cams.sum(axis=(1, 2))
        peak = cams.max()
        return (cams / peak if peak > 0 else cams), share


class ClipEmbedder:
    """CLIP ViT-L/14 image embedding of the website's face crops: L2-normalized per crop, averaged per clip.

    Crops arrive ImageNet-normalized, as the face model reads them; they are turned back into RGB and
    prepared as CLIPProcessor would (224 x 224, bicubic, CLIP-normalized).
    """

    def __init__(self, clip_dir: Path, device: str) -> None:
        from transformers import CLIPVisionModelWithProjection

        self.device = device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = (
            CLIPVisionModelWithProjection.from_pretrained(
                clip_dir, torch_dtype=self.dtype
            )
            .to(device)
            .eval()
        )

    @torch.inference_mode()
    def __call__(self, crops: torch.Tensor):
        rgb = (
            crops.to(self.device).float() * IMAGENET_STD.to(self.device)
            + IMAGENET_MEAN.to(self.device)
        ).clamp(0, 1)
        rgb = F.interpolate(
            rgb, size=(224, 224), mode="bicubic", align_corners=False
        ).clamp(0, 1)
        x = ((rgb - CLIP_MEAN.to(self.device)) / CLIP_STD.to(self.device)).to(
            self.dtype
        )
        emb = F.normalize(self.model(pixel_values=x).image_embeds.float(), dim=1).mean(
            0
        )
        return emb.cpu().numpy()


class GenProbe(torch.nn.Module):
    """The generated-video check: a linear head on the CLIP embedding of the face crops (scripts/gen_probe.py).

    The head file holds the standardization (mean, scale) and the logistic weights (coef, intercept),
    trained on HiFi-AVDF generated clips and their real twins. Returns the probability of "generated".
    """

    def __init__(self, path: Path, device: str, embed=None) -> None:
        import json

        super().__init__()
        head = json.loads(path.read_text(encoding="utf-8"))
        self.mean, self.scale = np.asarray(head["mean"]), np.asarray(head["scale"])
        self.coef, self.intercept = np.asarray(head["coef"]), float(head["intercept"])
        # Which of the generators it was trained on a clip looks most like; named only from `shown_from`.
        self.generators = head.get("generators")
        self.last_generator: dict | None = None
        self.embed = embed or ClipEmbedder(
            path.parent.parent / "pretrained" / "clip-vit-large-patch14", device
        )

    def forward(self, crops: torch.Tensor) -> float:
        embedding = np.asarray(self.embed(crops))
        z = (embedding - self.mean) / self.scale
        logit = float(z @ self.coef + self.intercept)
        self.last_generator = self._guess(embedding)
        return float(torch.sigmoid(torch.tensor(logit)))

    def _guess(self, embedding) -> dict | None:
        g = self.generators
        if not g:
            return None
        z = (embedding - np.asarray(g["mean"])) / np.asarray(g["scale"])
        logits = np.asarray(g["coef"]) @ z + np.asarray(g["intercept"])
        shares = np.exp(logits - logits.max())
        shares /= shares.sum()
        best = int(shares.argmax())
        return {
            "name": g["classes"][best],
            "share": round(float(shares[best]), 3),
            "sure": bool(shares[best] >= g["shown_from"]),
        }


class OnBatch:
    """A candidate as the scorer calls it: on the website's batch for one check.

    Face batches are (1, T, 3, H, W) crops; voice batches are (1, samples) audio with valid_samples.
    """

    def __init__(self, model: torch.nn.Module, check: str) -> None:
        self.model = model
        self.check = check

    def __call__(self, batch) -> float:
        if self.check == "face":
            return self.model(batch.values[0])
        wave = batch.values[0].reshape(-1)[: int(batch.valid_samples.reshape(-1)[0])]
        return self.model(wave)

    def saliency(self, frames: torch.Tensor):
        return self.model.saliency(frames)


LOADERS = {
    "clipln": lambda path, device: ClipLnFace(path, device),
    "effort": lambda path, device: EffortFace(
        path.parent.parent / "clip-vit-large-patch14", path
    ).to(device),
    "antideepfake": lambda path, device: AntiDeepfakeVoice(path.parent).to(device),
    "genprobe": lambda path, device: GenProbe(path, device),
}
# The batch each live check reads: the generated-video check reads the face crops.
BATCH_OF = {"face": "face", "voice": "voice", "gen": "face"}


def website_models(choice: dict, root: Path, device: str) -> dict[str, OnBatch]:
    """The models results/web-models.json puts live, each file checked against its pinned hash first.

    `choice` maps a check ("face", "voice") to {"loader", "file", "sha256"}; a check it leaves out keeps
    the study's model.
    """
    live = {}
    for check, batch in BATCH_OF.items():
        entry = choice.get(check)
        if entry is None:
            continue
        path = verified(root / entry["file"], entry["sha256"])
        live[check] = OnBatch(LOADERS[entry["loader"]](path, device).eval(), batch)
    return live
