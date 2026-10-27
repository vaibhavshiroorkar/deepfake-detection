"""Per-modality encoders for the cross-attention streams.

Both streams need the same thing from an encoder: a sequence of tokens over
time, `[batch, time, width]`, not a single pooled vector. Pooling before the
comparison would throw away exactly the information a synchronisation signal is
made of.

On encoder choice. `dashboard/lib/stream_spec.py` names AV-HuBERT and Whisper
for lip-sync, and HSEmotion for the emotion stream's face side.

  - AV-HuBERT's released implementation needs fairseq, which pins PyTorch to
    1.08/1.13/2.0, too old for this GPU. `streams/avhubert.py` restates the
    model in plain torch instead, and `AVHubertPairEncoder` below serves it.
  - HSEmotion's release is a pickle timm 1.0 cannot run, but its weights fit
    timm's `tf_efficientnet_b0`, which is what `FrameTokenEncoder` builds when
    `StreamConfig.backbone_weights` names the file (`streams/hsemotion.py`).
  - Whisper's encoder does not accept a waveform. It takes `[B, 80, 3000]`
    log-mel and hard-rejects any other length, because 3000 frames is 30
    seconds and its positional embeddings are sized for exactly that. The
    lip-sync window is 2 seconds, so Whisper would mean 28 seconds of padding
    and 100 useful tokens out of 1500. It is not used.

Wav2Vec2 takes a raw waveform of any length and is already used by
`branches/audio.py`, so it is the default audio encoder.
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

    def __init__(
        self, *, model_name: str = "facebook/wav2vec2-base", pretrained: bool = True
    ) -> None:
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


class AVHubertPairEncoder(nn.Module):
    """Frozen AV-HuBERT, run once on the lips alone and once on the audio alone.

    One model serves both modalities, so it is a pair encoder rather than two:
    two copies would double 103M frozen parameters for no gain. AV-HuBERT was
    pretrained with modality dropout, so a lips-only and an audio-only pass land
    in one feature space and their tokens can be compared frame by frame, which
    is the comparison the lip-sync stream makes.

    It stays frozen and in eval mode for the whole run. Its ResNet frontend has
    BatchNorm, and a training-mode pass would rewrite the running statistics
    even with every weight frozen.
    """

    def __init__(
        self,
        *,
        checkpoint: str = "",
        lip_crop: float = 1.0,
        lip_center_y: float = 0.5,
    ) -> None:
        super().__init__()
        from pathlib import Path

        from deepfake_detection.streams.avhubert import WIDTH, AVHubert, load_avhubert

        self.model = load_avhubert(Path(checkpoint)) if checkpoint else AVHubert()
        self.model.requires_grad_(False)
        self.lip_crop = lip_crop
        self.lip_center_y = lip_center_y
        self.width = WIDTH

    def train(self, mode: bool = True) -> AVHubertPairEncoder:
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, video: Tensor, audio: Tensor) -> tuple[Tensor, Tensor]:
        import torch

        from deepfake_detection.streams.avhubert import audio_features, lip_frames

        with torch.no_grad(), torch.autocast(video.device.type, enabled=False):
            lips = lip_frames(
                video.float(), crop=self.lip_crop, center_y=self.lip_center_y
            )
            sound = audio_features(audio.float())
            # AV-HuBERT's loader trims or zero-pads audio to the video length.
            frames = lips.shape[2]
            if sound.shape[2] >= frames:
                sound = sound[:, :, :frames]
            else:
                sound = torch.nn.functional.pad(sound, (0, frames - sound.shape[2]))
            return self.model(video=lips), self.model(audio=sound)

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Ignored: the encoder is frozen by design (see the class docstring)."""
