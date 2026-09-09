"""The audiovisual streams: audio asks a question of video, the answer is the signal.

    mouth  [B, 50, 3, 112, 112] -> frame encoder -> [B, 50, Dv] -> [B, 50, 256]  (Key/Value)
    audio  [B, 32000]           -> wav2vec2      -> [B, Ta, Da] -> [B, Ta, 256]  (Query)
    cross-attention                              -> weights [B, Ta, 50]
                                                 -> attended [B, Ta, 256]
    mean over time, project, LayerNorm           -> embedding [B, 256]
    development head                             -> logit [B]

Why this shape rather than a classifier over frames. A visual-only model learns
what authentic video looks like, which is a property of the dataset, and the
measured consequence was 1.0000 ROC-AUC in-domain against 130 of 155 genuine
Celeb-DF videos called fake. Whether this audio matches this mouth is asked and
answered inside one clip, so there is no dataset-level appearance prior for the
model to memorise instead.

The attention weights are returned rather than consumed internally because
`diagonal_mass` over them is the stream's independent check: sound arrives at a
near-fixed offset from the articulation that produced it, so a stream that
learned the real correspondence concentrates near the diagonal, and one that
learned a shortcut does not. That check does not depend on the classification
accuracy being good, which makes it worth more than the accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from deepfake_detection.streams.config import StreamConfig
from deepfake_detection.streams.cross_attention import (
    CrossModalAttention,
    diagonal_mass,
)
from deepfake_detection.streams.encoders import FrameTokenEncoder, Wav2Vec2TokenEncoder


@dataclass(frozen=True, slots=True)
class StreamOutput:
    logit: Tensor
    embedding: Tensor
    attention: Tensor
    diagonal_mass: Tensor


def resample_tokens(tokens: Tensor, *, size: int) -> Tensor:
    """Pick `size` evenly spaced token positions from a sequence.

    Nearest-neighbour selection rather than interpolation, matching
    `branches/sync.py::_nearest_temporal_tokens`. That choice is not stylistic:
    an earlier prototype run failed outright on
    `upsample_linear1d_backward_out_cuda`, which has no deterministic
    implementation, and this pipeline runs with deterministic algorithms on.
    """
    if tokens.ndim != 3:
        raise ValueError("Tokens must be [batch, time, features]")
    if size <= 0:
        raise ValueError("Target size must be positive")
    if tokens.shape[1] == size:
        return tokens
    if tokens.shape[1] < size:
        raise ValueError(
            f"Cannot resample {tokens.shape[1]} tokens up to {size}; "
            "the encoder produced a shorter sequence than the target"
        )
    indices = (
        torch.linspace(0, tokens.shape[1] - 1, steps=size, device=tokens.device)
        .round()
        .long()
    )
    return tokens.index_select(1, indices)


class CrossModalStream(nn.Module):
    """One audiovisual stream. Lip-sync and emotion differ only in their inputs.

    `query_steps` fixes how many time steps the comparison runs over. Both
    modalities are resampled onto it, so a 50-frame video against a 99-token
    audio sequence still yields a square-ish attention map that `diagonal_mass`
    can read.
    """

    def __init__(
        self,
        *,
        config: StreamConfig,
        audio_model: str = "facebook/wav2vec2-base",
        pretrained: bool = True,
        attention_heads: int = 4,
        query_steps: int = 50,
    ) -> None:
        super().__init__()
        self.config = config
        self.query_steps = query_steps

        self.video_encoder = FrameTokenEncoder(config)
        self.audio_encoder = Wav2Vec2TokenEncoder(
            model_name=audio_model, pretrained=pretrained
        )

        dim = config.common_dim
        self.video_projection = nn.Linear(self.video_encoder.width, dim)
        self.audio_projection = nn.Linear(self.audio_encoder.width, dim)
        self.attention = CrossModalAttention(dim=dim, num_heads=attention_heads)
        self.projection = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        # Development only, exactly as VisualStream.temp_head is. The stream's
        # real product is the embedding that fusion reads; this head exists so a
        # stream can be measured on its own before fusion is trained.
        self.temp_head = nn.Linear(dim, 1)

    def forward(self, *, video: Tensor, audio: Tensor) -> StreamOutput:
        video_tokens = self.video_projection(self.video_encoder(video))
        audio_tokens = self.audio_projection(self.audio_encoder(audio))

        steps = min(self.query_steps, video_tokens.shape[1], audio_tokens.shape[1])
        video_tokens = resample_tokens(video_tokens, size=steps)
        audio_tokens = resample_tokens(audio_tokens, size=steps)

        # Audio is Query, video is Key and Value. The direction is the claim:
        # the sound is what we have, and we ask how well the mouth accounts for it.
        result = self.attention(
            query=audio_tokens, key=video_tokens, value=video_tokens
        )
        # The residual, not the attended vector. `stream_spec.py` calls this
        # stream's product a "synchronisation-mismatch vector", and the mismatch
        # is what the mouth failed to explain about the sound, so the residual is
        # the literal reading of that.
        #
        # It is also the only version that trains. Pooling the attended vector
        # alone starts almost input-independent: attention is near-uniform at
        # initialisation, so every attended step collapses to the mean video
        # token and the mean over time erases what little structure remains.
        # Measured on an untrained stream, the embedding moved by 2e-7 when the
        # audio was shifted by 320 ms, which is float noise, not a gradient.
        mismatch = audio_tokens - result.attended
        embedding = self.projection(mismatch.mean(dim=1))
        return StreamOutput(
            logit=self.temp_head(embedding).squeeze(-1),
            embedding=embedding,
            attention=result.weights,
            diagonal_mass=diagonal_mass(result.weights),
        )

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Both encoders move together, so the staged-freeze schedule in
        `training/binary.py` applies unchanged."""
        self.video_encoder.set_backbone_trainable(trainable)
        self.audio_encoder.set_backbone_trainable(trainable)


def build_lipsync_stream(
    *,
    video_backbone: str = "tf_efficientnet_b0.ns_jft_in1k",
    audio_model: str = "facebook/wav2vec2-base",
    pretrained: bool = True,
    common_dim: int = 256,
    attention_heads: int = 4,
) -> CrossModalStream:
    """Mouth crops against the audio track.

    Reads `sync_video_view` and `sync_audio_view`, which the cache already holds
    and which are both exactly 2.0 seconds from the same start, so the two
    modalities are aligned by construction with no window to reconcile.
    """
    config = StreamConfig(
        stream_name="lipsync",
        backbone_name=video_backbone,
        pretrained=pretrained,
        common_dim=common_dim,
        image_size=112,
        frame_chunk_size=25,
    )
    return CrossModalStream(
        config=config,
        audio_model=audio_model,
        pretrained=pretrained,
        attention_heads=attention_heads,
        query_steps=50,
    )


def build_emotion_stream(
    *,
    video_backbone: str = "tf_efficientnet_b0.ns_jft_in1k",
    audio_model: str = "facebook/wav2vec2-base",
    pretrained: bool = True,
    common_dim: int = 256,
    attention_heads: int = 4,
) -> CrossModalStream:
    """Facial expression against vocal affect.

    Reads `visual_view` and `audio_view`. Unlike lip-sync these two are not
    aligned by construction: the face frames span the whole clip while the audio
    is a fixed 4.0 second window, so the dataset trims the video to the leading
    frames the audio actually covers (see `STREAM_FRAME_LIMITS`).

    `stream_spec.py` names HSEmotions for the face side. EmotiEffLib, its
    maintained successor, targets timm 0.9 while this environment runs 1.0.27,
    so the same general backbone as lip-sync is used until that is resolved. The
    mechanism is the contribution; the encoder is an ablation.
    """
    config = StreamConfig(
        stream_name="emotion",
        backbone_name=video_backbone,
        pretrained=pretrained,
        common_dim=common_dim,
        image_size=224,
        frame_chunk_size=8,
    )
    return CrossModalStream(
        config=config,
        audio_model=audio_model,
        pretrained=pretrained,
        attention_heads=attention_heads,
        # The trimmed face sequence, so the comparison runs over frames the
        # audio window genuinely covers.
        query_steps=8,
    )
