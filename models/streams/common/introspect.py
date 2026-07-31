"""Capture what a visual stream computes internally on one clip.

The Preprocessing page walks a clip to the tensor a model receives and stops
there. This module continues that walk inside the model: it hooks the backbone's
feature stages, runs one forward pass, and returns every intermediate as plain
numpy so a page can render it.

Torch and numpy only. Nothing here imports Streamlit, matplotlib or cv2, so it
tests without a running app; the colouring and layout live in
dashboard/lib/trace_ui.py.

Two shapes of stage come back, because the backbones are not the same kind of
network. A CNN stage (Xception, EfficientNet) emits a spatial map [C, H, W] per
frame. A ViT stage (DINOv2) emits a token matrix [1 + P, D] per frame, one row
per patch plus the CLS row. `StageTrace.kind` says which, and callers must
branch: a channel grid is meaningless for tokens, and a patch grid is meaningless
for channels.

Memory: a full Xception trace at 16 frames is ~150 MB if every stage keeps every
frame at full width, which is not worth holding in session state. Each stage
therefore keeps the per-frame *summary* map for all frames (under a megabyte) and
the full activation for one frame only, chosen by `detail_frame`.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SPATIAL = "spatial"
TOKENS = "tokens"

# A prefix token is a row that is not a patch: CLS, and registers on the models
# that carry them. The count is inferred rather than hardcoded (see token_grid),
# and no timm ViT carries more than a handful.
MAX_PREFIX_TOKENS = 8


@dataclass(frozen=True)
class StageSpec:
    """One hookable point inside a backbone."""
    name: str            # module path, resolvable with backbone.get_submodule
    channels: int
    reduction: int       # input pixels per output pixel, so 32 means a 7x7 map from 224


@dataclass
class StageTrace:
    spec: StageSpec
    kind: str                  # SPATIAL or TOKENS
    shape: tuple               # the activation's shape for ONE frame
    summary: np.ndarray        # [T, h, w]: per-frame channel mean, or patch-token norm
    detail: np.ndarray         # [C, H, W], or [1 + P, D] for a ViT stage
    detail_frame: int


@dataclass
class StreamTrace:
    """Everything one clip touched on its way to an embedding."""
    stages: list[StageTrace]
    input_shape: tuple         # [1, T, 3, H, W] as handed in
    folded_shape: tuple        # [T, 3, H, W] as the backbone sees it
    frame_features: np.ndarray  # [T, D] backbone output per frame
    temporal_sequence: np.ndarray | None   # [T, H*dirs], None when mean-pooling
    clip_vector: np.ndarray    # [temporal_out] entering the projection
    embedding: np.ndarray      # [common_dim] leaving it: the fusion product
    logit: float
    prob: float
    detail_frame: int


def backbone_stages(backbone) -> list[StageSpec]:
    """The stages worth hooking, in depth order.

    timm sets `feature_info` on every backbone this project uses: a list of dicts
    naming a module, its channel count and its downsampling factor. It is the
    same list `features_only=True` would extract, read off a normally-built model
    so there is no second copy of the weights.

    A backbone without it falls back to its top-level children, which is coarse
    but never empty.
    """
    info = getattr(backbone, "feature_info", None)
    if info:
        return [StageSpec(name=d["module"], channels=int(d.get("num_chs", 0)),
                          reduction=int(d.get("reduction", 0)))
                for d in info if d.get("module")]
    return [StageSpec(name=name, channels=0, reduction=0)
            for name, _ in backbone.named_children()]


def token_grid(num_tokens: int) -> tuple[int, int]:
    """(prefix_count, grid_side) for a ViT stage output with num_tokens rows.

    The patch rows form a square, so the prefix is whatever is left over once a
    square fits. Smallest prefix wins: 257 is 1 + 16^2, not 32 + 15^2.
    """
    for prefix in range(MAX_PREFIX_TOKENS + 1):
        patches = num_tokens - prefix
        if patches <= 0:
            break
        side = int(round(patches ** 0.5))
        if side * side == patches:
            return prefix, side
    raise ValueError(f"{num_tokens} tokens do not decompose into a prefix plus a square grid")


class _StageRecorder:
    """Accumulates one stage's activations across however many chunks arrive.

    VisualStream splits the folded frames into VRAM-sized chunks, so a hook can
    fire several times per forward pass. Each call carries the next slice of
    frames, tracked by `seen`, and the detail frame is captured from whichever
    chunk happens to contain it.
    """

    def __init__(self, spec: StageSpec, detail_frame: int):
        self.spec = spec
        self.detail_frame = detail_frame
        self.seen = 0
        self.kind = None
        self.summaries: list[np.ndarray] = []
        self.detail: np.ndarray | None = None
        self.shape: tuple = ()

    def __call__(self, module, args, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        act = output.detach().float()
        n = act.shape[0]

        if act.ndim == 4:                      # [n, C, H, W]
            self.kind = SPATIAL
            summary = act.mean(dim=1)          # [n, H, W]
        elif act.ndim == 3:                    # [n, tokens, D]
            self.kind = TOKENS
            prefix, side = token_grid(act.shape[1])
            patches = act[:, prefix:, :]
            summary = patches.norm(dim=-1).reshape(n, side, side)
        else:
            raise ValueError(
                f"stage '{self.spec.name}' produced a {act.ndim}-D activation "
                f"{tuple(act.shape)}; only [N,C,H,W] and [N,tokens,D] are understood")

        self.shape = tuple(act.shape[1:])
        self.summaries.append(summary.cpu().numpy())

        local = self.detail_frame - self.seen
        if 0 <= local < n:
            self.detail = act[local].cpu().numpy()
        self.seen += n

    def result(self) -> StageTrace:
        if self.detail is None:
            raise IndexError(
                f"detail_frame {self.detail_frame} is past the {self.seen} frames "
                f"stage '{self.spec.name}' saw")
        return StageTrace(spec=self.spec, kind=self.kind, shape=self.shape,
                          summary=np.concatenate(self.summaries, axis=0),
                          detail=self.detail, detail_frame=self.detail_frame)


class _Collector:
    """Catches a module's output across chunked calls, concatenated on dim 0."""

    def __init__(self):
        self.parts: list[torch.Tensor] = []

    def __call__(self, module, args, output):
        self.parts.append(output.detach().float().cpu())

    def value(self) -> torch.Tensor:
        return torch.cat(self.parts, dim=0)


def trace_visual_stream(model, frames: torch.Tensor, detail_frame: int = 0) -> StreamTrace:
    """Run one clip through `model`, keeping every intermediate.

    frames is [1, T, 3, H, W]: the same tensor the Run path builds. Batch size is
    fixed at one because this exists to explain a single clip, and a stage
    summary averaged over several clips would explain none of them.
    """
    if frames.ndim != 5 or frames.shape[0] != 1:
        raise ValueError(f"trace needs one clip shaped [1, T, 3, H, W], got {tuple(frames.shape)}")
    num_frames = frames.shape[1]
    if not 0 <= detail_frame < num_frames:
        raise ValueError(f"detail_frame {detail_frame} outside the clip's {num_frames} frames")

    specs = backbone_stages(model.backbone)
    recorders = [_StageRecorder(spec, detail_frame) for spec in specs]
    features = _Collector()

    handles = [model.backbone.get_submodule(r.spec.name).register_forward_hook(r)
               for r in recorders]
    handles.append(model.backbone.register_forward_hook(features))
    # Gradient checkpointing has to come off, or the trace comes back empty:
    # timm runs a checkpointed backbone through a flattened functional segment,
    # so the stage modules are never called as modules and their hooks never
    # fire. It costs nothing here, since a checkpointed forward only saves memory
    # during backprop and this pass is under no_grad.
    checkpointing = getattr(model, "grad_checkpointing", False)
    if checkpointing:
        model.backbone.set_grad_checkpointing(False)
    try:
        with torch.no_grad():
            logit, embedding = model(frames)
            feats = features.value()                          # [T, D]
            temporal_sequence = None
            if model.temporal is not None:
                out, hidden = model.temporal(feats.unsqueeze(0))   # [1, T, H*dirs]
                temporal_sequence = out.squeeze(0).numpy()
                clip_vector = _final_hidden(model, hidden)
            else:
                clip_vector = feats.mean(dim=0)
    finally:
        for handle in handles:
            handle.remove()
        if checkpointing:
            model.backbone.set_grad_checkpointing(True)

    return StreamTrace(
        stages=[r.result() for r in recorders],
        input_shape=tuple(frames.shape),
        folded_shape=(num_frames,) + tuple(frames.shape[2:]),
        frame_features=feats.numpy(),
        temporal_sequence=temporal_sequence,
        clip_vector=clip_vector.numpy(),
        embedding=embedding.detach().float().cpu().squeeze(0).numpy(),
        logit=float(logit.item()),
        prob=float(torch.sigmoid(logit).item()),
        detail_frame=detail_frame,
    )


def _final_hidden(model, hidden) -> torch.Tensor:
    """The clip vector an RNN hands the projection, read off its final state.

    VisualStream.forward computes this and then throws the per-step sequence
    away, which is the part worth showing. Rebuilding the vector here from the
    same rule rather than hooking the projection's input keeps the two paths
    independently checkable: if this ever disagrees with the model's own
    embedding, the trace is wrong and should be noticed.

    `hidden` is h_n for a GRU and (h_n, c_n) for an LSTM.
    """
    h_n = hidden[0] if model.config.temporal_type.lower() == "lstm" else hidden
    num_dir = 2 if model.config.temporal_bidirectional else 1
    h_last = h_n.view(model.config.temporal_layers, num_dir, 1, model.config.temporal_hidden)[-1]
    return torch.cat([h_last[d] for d in range(num_dir)], dim=-1).squeeze(0)


# ------------------------------------------------------------------ numpy views

def normalize01(a: np.ndarray) -> np.ndarray:
    """Min-max to [0, 1]. A constant array maps to zeros rather than dividing by 0."""
    a = np.asarray(a, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def top_channels(detail: np.ndarray, k: int) -> list[int]:
    """Indices of the k channels that responded most strongly, strongest first.

    Mean rather than max: one hot pixel is usually an artifact of the
    normalisation, while a high mean means the channel fired across the map.
    """
    if detail.ndim != 3:
        raise ValueError(f"expected a [C, H, W] activation, got {detail.shape}")
    order = np.argsort(detail.mean(axis=(1, 2)))[::-1]
    return [int(i) for i in order[:k]]


def cls_similarity_map(detail: np.ndarray) -> np.ndarray:
    """Cosine similarity of each patch token to the CLS token, as a [side, side] map.

    The ViT stand-in for a channel-mean heatmap: it shows which regions the
    summary token is actually built from. DINOv2 exposes no attention weights
    through timm's fused attention, so this is measured from the tokens instead.
    """
    if detail.ndim != 2:
        raise ValueError(f"expected a [tokens, D] activation, got {detail.shape}")
    prefix, side = token_grid(detail.shape[0])
    if prefix == 0:
        raise ValueError("this stage has no prefix token to compare patches against")
    cls = detail[0]
    patches = detail[prefix:]
    norms = np.linalg.norm(patches, axis=1) * np.linalg.norm(cls)
    sims = (patches @ cls) / np.maximum(norms, 1e-12)
    return sims.reshape(side, side)


def patch_token_map(detail: np.ndarray) -> np.ndarray:
    """Per-patch token L2 norm as a [side, side] map: where the tokens carry weight."""
    if detail.ndim != 2:
        raise ValueError(f"expected a [tokens, D] activation, got {detail.shape}")
    prefix, side = token_grid(detail.shape[0])
    return np.linalg.norm(detail[prefix:], axis=1).reshape(side, side)
