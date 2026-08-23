from __future__ import annotations

from torch import Tensor, nn

from .contracts import BranchOutput


def _audio_tokens(output: object) -> Tensor:
    tokens = getattr(output, "last_hidden_state", output)
    if not isinstance(tokens, Tensor) or tokens.ndim != 3:
        raise ValueError("Audio encoder must return [batch, time, features] tokens")
    return tokens


class AudioSpoofBranch(nn.Module):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        encoder_dim: int,
        projection_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projection = nn.Linear(encoder_dim, projection_dim)
        self.attention = nn.Linear(projection_dim, 1)
        self.classifier = nn.Linear(projection_dim, 1)

    def forward(self, waveform: Tensor) -> BranchOutput:
        if waveform.ndim != 2:
            raise ValueError("Audio input must have shape [batch, samples]")
        tokens = self.projection(_audio_tokens(self.encoder(waveform)))
        weights = self.attention(tokens).squeeze(-1).softmax(dim=1)
        embedding = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(embedding).squeeze(-1)
        return BranchOutput(
            logits=logits,
            embedding=embedding,
            token_count=tokens.shape[1],
        )

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable


def build_wav2vec2_audio_branch(
    *,
    model_name: str = "facebook/wav2vec2-base",
    projection_dim: int = 256,
    pretrained: bool = True,
) -> AudioSpoofBranch:
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    encoder = (
        Wav2Vec2Model.from_pretrained(model_name)
        if pretrained
        else Wav2Vec2Model(Wav2Vec2Config())
    )
    return AudioSpoofBranch(
        encoder=encoder,
        encoder_dim=encoder.config.hidden_size,
        projection_dim=projection_dim,
    )
