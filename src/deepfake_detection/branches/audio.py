from __future__ import annotations

import torch
from torch import Tensor, nn

from .contracts import BranchOutput


def _audio_tokens(output: object) -> Tensor:
    tokens = getattr(output, "last_hidden_state", output)
    if not isinstance(tokens, Tensor) or tokens.ndim != 3:
        raise ValueError("Audio encoder must return [batch, time, features] tokens")
    return tokens


def token_mask_from_valid_samples(
    valid_samples: Tensor,
    *,
    total_samples: int,
    token_count: int,
) -> Tensor:
    """Which encoder tokens carry real audio, from each clip's valid sample count.

    Every clip reaches the encoder padded to one fixed length, so the token axis
    is the same width for a 2 second clip and an 8 second one. The encoder
    downsamples by a fixed factor, so the boundary maps proportionally: a clip
    whose audio fills a quarter of the buffer fills a quarter of the tokens.

    The token holding the boundary is kept. It carries real audio mixed with
    padding, and dropping it would discard the end of every clip that does not
    land exactly on a token edge.

    Every clip keeps at least one token. A fully empty clip has no valid audio
    to attend to, and a row masked to nothing makes the softmax below produce
    NaN rather than a low score, which is a worse answer than an arbitrary one.
    """
    if valid_samples.ndim != 1:
        raise ValueError("Valid sample counts must have shape [batch]")
    if total_samples <= 0 or token_count <= 0:
        raise ValueError("Total samples and token count must be positive")
    clamped = valid_samples.clamp(min=0, max=total_samples).to(torch.float64)
    tokens = torch.ceil(clamped * token_count / total_samples).to(torch.long)
    tokens = tokens.clamp(min=1, max=token_count)
    positions = torch.arange(token_count, device=valid_samples.device)
    return positions.unsqueeze(0) < tokens.unsqueeze(1)


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

    def forward(
        self,
        waveform: Tensor,
        *,
        valid_samples: Tensor | None = None,
    ) -> BranchOutput:
        """Score one batch of padded waveforms.

        `valid_samples` is how many samples of each row are real audio. Without
        it the attention pools over the padding too, which is the padding
        shortcut named in the threat model: a short clip and a long one differ
        in how much silence they end with, and that is a cue about the clip
        rather than about the voice in it.

        It stays optional because the cached views that predate it record no
        sample count, and refusing to score them would strand every existing
        checkpoint and result. An unmasked call is the old behaviour exactly.
        """
        if waveform.ndim != 2:
            raise ValueError("Audio input must have shape [batch, samples]")
        tokens = self.projection(_audio_tokens(self.encoder(waveform)))
        scores = self.attention(tokens).squeeze(-1)
        if valid_samples is not None:
            if valid_samples.shape[0] != waveform.shape[0]:
                raise ValueError("Valid sample counts must match the batch size")
            mask = token_mask_from_valid_samples(
                valid_samples,
                total_samples=waveform.shape[1],
                token_count=tokens.shape[1],
            )
            # Masked before the softmax, not after: zeroing afterwards leaves
            # the padded steps in the denominator, so the weights no longer sum
            # to one and a heavily padded clip is scaled down as a whole.
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = scores.softmax(dim=1)
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


def build_audio_branch(
    *,
    model_name: str = "facebook/wav2vec2-base",
    projection_dim: int = 256,
    pretrained: bool = True,
) -> AudioSpoofBranch:
    """One audio branch over any HuggingFace speech encoder.

    Built through `AutoModel`, which picks the architecture the checkpoint was
    trained as. `Wav2Vec2Model.from_pretrained` would load any checkpoint into a
    Wav2Vec2 body and silently drop whatever does not fit, which is not a
    conversion but a quiet truncation: `microsoft/wavlm-base` loads that way with
    94,371,712 parameters against `WavLMModel`'s 94,381,936, so the gated
    relative position embeddings that distinguish WavLM are simply absent.

    The AUD-01 comparison asks whether WavLM beats Wav2Vec2. Running it through
    the wrong class would have answered a different question and looked like an
    answer to that one.

    For `facebook/wav2vec2-base` this is the same model as before:
    `AutoModel` resolves it to `Wav2Vec2Model`.
    """
    from transformers import AutoConfig, AutoModel

    if pretrained:
        encoder = AutoModel.from_pretrained(model_name)
    else:
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
    return AudioSpoofBranch(
        encoder=encoder,
        encoder_dim=encoder.config.hidden_size,
        projection_dim=projection_dim,
    )


def build_wav2vec2_audio_branch(
    *,
    model_name: str = "facebook/wav2vec2-base",
    projection_dim: int = 256,
    pretrained: bool = True,
) -> AudioSpoofBranch:
    """The frozen default, kept as its own name.

    Every shipped audio checkpoint was built through this call, so it stays.
    """
    return build_audio_branch(
        model_name=model_name,
        projection_dim=projection_dim,
        pretrained=pretrained,
    )
