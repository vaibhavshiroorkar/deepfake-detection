"""Per-modality encoders for the cross-attention streams.

Both streams need the same thing from an encoder: a sequence of tokens over
time, `[batch, time, width]`, not a single pooled vector. Pooling before the
comparison would throw away exactly the information a synchronisation signal is
made of.

On encoder choice. `dashboard/lib/stream_spec.py` names AV-HuBERT and Whisper
for lip-sync. Neither is used here yet, deliberately:

  - AV-HuBERT's released implementation needs fairseq, which pins PyTorch to
    1.08/1.13/2.0. This environment runs torch 2.12.1+cu130, and installing
    fairseq would take the working CUDA stack down with it.
  - Whisper's encoder does not accept a waveform. It takes `[B, 80, 3000]`
    log-mel and hard-rejects any other length, because 3000 frames is 30
    seconds and its positional embeddings are sized for exactly that. The
    lip-sync window is 2 seconds, so Whisper would mean 28 seconds of padding
    and 100 useful tokens out of 1500.

Wav2Vec2 takes a raw waveform of any length and is already used by
`branches/audio.py`, so it is the audio encoder here. The contribution is the
cross-modal mechanism, not the choice of encoder, and swapping one in later is
a clean ablation rather than a rewrite.
"""

from __future__ import annotations

from torch import Tensor, nn

from deepfake_detection.streams.config import StreamConfig


def _tokens(output: object) -> Tensor:
    """The `[batch, time, width]` sequence out of a transformer output object."""
    tokens = getattr(output, "last_hidden_state", output)
    if not isinstance(tokens, Tensor) or tokens.ndim != 3:
        raise ValueError("Encoder must return [batch, time, features] tokens")
    return tokens


class Wav2Vec2TokenEncoder(nn.Module):
    """Raw waveform to a token sequence, no fixed input length.

    Deliberately not `branches/audio.py`'s `AudioSpoofBranch`: that pools its
    tokens through attention into one clip vector, which is the right shape for
    a unimodal spoof classifier and the wrong shape for a comparison against
    video over time.
    """

    def __init__(self, *, model_name: str = "facebook/wav2vec2-base", pretrained: bool = True) -> None:
        super().__init__()
        from transformers import Wav2Vec2Config, Wav2Vec2Model

        self.encoder = (
            Wav2Vec2Model.from_pretrained(model_name)
            if pretrained
            else Wav2Vec2Model(Wav2Vec2Config())
        )
        self.width = self.encoder.config.hidden_size

    def forward(self, waveform: Tensor) -> Tensor:
        if waveform.ndim != 2:
            raise ValueError("Waveform must be [batch, samples]")
        return _tokens(self.encoder(waveform))

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable


class FrameTokenEncoder(nn.Module):
    """A per-frame image backbone applied across a clip, giving one token a frame.

    Frames are folded into the batch dimension and optionally processed in
    chunks, the same trick `streams/visual_stream.py` uses, because a 50-frame
    mouth sequence at batch 8 is 400 images through a backbone at once and the
    16 GB card will not hold that plus gradients.
    """

    def __init__(self, config: StreamConfig) -> None:
        super().__init__()
        from deepfake_detection.streams.visual_stream import _create_backbone

        self.config = config
        self.backbone = _create_backbone(config)
        self.width = self.backbone.num_features

    def _run_chunked(self, folded: Tensor) -> Tensor:
        chunk = self.config.frame_chunk_size
        if not chunk or chunk >= folded.shape[0]:
            return self.backbone(folded)
        import torch

        return torch.cat(
            [
                self.backbone(folded[index : index + chunk])
                for index in range(0, folded.shape[0], chunk)
            ],
            dim=0,
        )

    def forward(self, frames: Tensor) -> Tensor:
        if frames.ndim != 5:
            raise ValueError("Frames must be [batch, time, channels, height, width]")
        batch, time, channels, height, width = frames.shape
        folded = frames.reshape(batch * time, channels, height, width)
        features = self._run_chunked(folded)
        if features.ndim != 2:
            raise ValueError(
                f"Backbone must pool to [n, features], got {tuple(features.shape)}"
            )
        return features.reshape(batch, time, -1)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable
