"""
The reusable visual-stream template. EVERY visual stream (EfficientNet now;
Xception and DINOv2 in Stage 3) is an instance of this one class -- they differ
only by the StreamConfig passed in. That is the "reuse the proven pipeline"
principle from PROJECT_OVERVIEW.md Section 9.

Data flow for one clip (batched as [B, T, 3, H, W], T = num_frames = 16):

    frames  [B, T, 3, 224, 224]
      | flatten frames, run backbone (per-frame, in VRAM-sized chunks)
    per-frame embeddings  [B, T, F]        (F = backbone feature dim, e.g. 1280)
      | temporal model (LSTM/GRU) reads the sequence
    clip embedding  [B, H_dir]             (H_dir = hidden * num_directions)
      | projection (Linear + LayerNorm)
    stream embedding  [B, common_dim]      <-- THIS is what the feature store gets
      | temporary classifier head (dev-only, discarded at fusion)
    logit  [B]                             <-- for Stage 2 standalone metrics only

The projection output is the real product; the head exists only so Stages 2-5
can measure a stream's standalone power before fusion exists.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import torch
    import torch.nn as nn
    import timm
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Run: uv sync --extra cpu (or --extra cu130 for GPU), see README.md")
    sys.exit(1)

from models.streams.common.config import StreamConfig


class VisualStream(nn.Module):
    def __init__(self, config: StreamConfig):
        super().__init__()
        self.config = config

        # --- backbone (per-frame feature extractor) ---
        # num_classes=0 + global_pool="avg": drop the ImageNet classifier and
        # return one pooled feature vector per image instead of logits.
        try:
            self.backbone = timm.create_model(
                config.backbone_name,
                pretrained=config.pretrained,
                num_classes=0,
                global_pool="avg",
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load backbone '{config.backbone_name}' from timm "
                f"(check the name and internet for weight download): {e}"
            ) from e

        self.feature_dim = self.backbone.num_features  # e.g. 1280 for efficientnet_b0

        if config.grad_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            # Trades compute for VRAM: activations are recomputed during backward
            # instead of stored. Essential to fit 16 frames on a 6GB card.
            self.backbone.set_grad_checkpointing(True)

        # --- temporal model (frame sequence -> one clip vector) ---
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}.get(config.temporal_type.lower())
        if rnn_cls is None:
            raise ValueError(f"temporal_type must be 'lstm' or 'gru', got '{config.temporal_type}'")
        self.temporal = rnn_cls(
            input_size=self.feature_dim,
            hidden_size=config.temporal_hidden,
            num_layers=config.temporal_layers,
            batch_first=True,               # input shape [B, T, F]
            bidirectional=config.temporal_bidirectional,
        )
        num_directions = 2 if config.temporal_bidirectional else 1
        temporal_out_dim = config.temporal_hidden * num_directions

        # --- projection to the shared common_dim (the feature-store contract) ---
        # LayerNorm after the Linear keeps the scale of different streams'
        # embeddings comparable before fusion concatenates them.
        self.projection = nn.Sequential(
            nn.Linear(temporal_out_dim, config.common_dim),
            nn.LayerNorm(config.common_dim),
        )

        # --- temporary classifier head (DEV ONLY, discarded at fusion) ---
        # One logit per clip; BCEWithLogitsLoss is applied outside.
        self.temp_head = nn.Linear(config.common_dim, 1)

    # ------------------------------------------------------------------
    def _run_backbone_chunked(self, frames_flat: "torch.Tensor") -> "torch.Tensor":
        """
        frames_flat: [B*T, 3, H, W]. Run the backbone on at most
        frame_chunk_size images at a time so peak VRAM stays bounded, then
        concatenate. Gradients still flow through the concatenation.
        """
        chunk = self.config.frame_chunk_size
        if chunk is None or chunk >= frames_flat.shape[0]:
            return self.backbone(frames_flat)
        outs = []
        for i in range(0, frames_flat.shape[0], chunk):
            outs.append(self.backbone(frames_flat[i : i + chunk]))
        return torch.cat(outs, dim=0)

    def forward(self, frames: "torch.Tensor", return_embedding: bool = True):
        """
        frames: [B, T, 3, H, W].
        Returns (logit [B], embedding [B, common_dim]).
        `logit` is the dev head's output; `embedding` is the feature-store product.
        """
        B, T, C, H, W = frames.shape
        frames_flat = frames.reshape(B * T, C, H, W)

        feats_flat = self._run_backbone_chunked(frames_flat)     # [B*T, F]
        feats = feats_flat.reshape(B, T, self.feature_dim)       # [B, T, F]

        # Temporal model. We take the final layer's hidden state as the clip
        # summary. For a bidirectional RNN that's the last two directions'
        # hidden states concatenated.
        self.temporal.flatten_parameters()
        if self.config.temporal_type.lower() == "lstm":
            _, (h_n, _) = self.temporal(feats)
        else:  # gru
            _, h_n = self.temporal(feats)
        # h_n: [num_layers*num_directions, B, hidden]. Take the last layer's
        # direction(s) and concatenate them into [B, hidden*num_directions].
        num_directions = 2 if self.config.temporal_bidirectional else 1
        h_last = h_n.view(self.config.temporal_layers, num_directions, B, self.config.temporal_hidden)[-1]
        clip_vec = torch.cat([h_last[d] for d in range(num_directions)], dim=-1)  # [B, hidden*dir]

        embedding = self.projection(clip_vec)                    # [B, common_dim]
        logit = self.temp_head(embedding).squeeze(-1)            # [B]

        if return_embedding:
            return logit, embedding
        return logit

    # ------------------------------------------------------------------
    def set_backbone_trainable(self, trainable: bool):
        """Freeze/unfreeze the backbone for the two-phase training schedule."""
        for p in self.backbone.parameters():
            p.requires_grad = trainable
