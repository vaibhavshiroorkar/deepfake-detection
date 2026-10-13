from types import SimpleNamespace

import pytest
import torch
from torch import nn

from deepfake_detection.branches.audio import (
    AudioSpoofBranch,
    token_mask_from_valid_samples,
)
from deepfake_detection.branches.sync import (
    SynchronizationBranch,
    _nearest_temporal_tokens,
)
from deepfake_detection.branches.visual import VisualArtifactBranch


class FrameBackbone(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(3, output_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.projection(frames.mean(dim=(-1, -2)))


class AudioBackbone(nn.Module):
    def __init__(self, output_dim: int, token_count: int) -> None:
        super().__init__()
        self.projection = nn.Linear(1, output_dim)
        self.token_count = token_count

    def forward(self, waveform: torch.Tensor) -> SimpleNamespace:
        chunks = waveform.unfold(
            1,
            waveform.shape[1] // self.token_count,
            waveform.shape[1] // self.token_count,
        )
        tokens = chunks.mean(dim=-1, keepdim=True)
        return SimpleNamespace(last_hidden_state=self.projection(tokens))


def test_visual_branch_preserves_batch_and_returns_one_logit_per_clip() -> None:
    model = VisualArtifactBranch(
        backbone=FrameBackbone(12),
        backbone_dim=12,
        hidden_dim=8,
    )

    output = model(torch.randn(2, 4, 3, 8, 8))

    assert output.logits.shape == (2,)
    assert output.embedding.shape == (2, 8)


def test_audio_branch_uses_temporal_tokens_before_pooling() -> None:
    model = AudioSpoofBranch(
        encoder=AudioBackbone(output_dim=10, token_count=5),
        encoder_dim=10,
        projection_dim=8,
    )

    output = model(torch.randn(3, 100))

    assert output.logits.shape == (3,)
    assert output.embedding.shape == (3, 8)
    assert output.token_count == 5


def test_sync_branch_keeps_time_axis_and_scores_offsets() -> None:
    model = SynchronizationBranch(
        video_encoder=FrameBackbone(12),
        video_dim=12,
        audio_encoder=AudioBackbone(output_dim=10, token_count=10),
        audio_dim=10,
        projection_dim=8,
        transformer_layers=1,
        attention_heads=2,
        offset_classes=8,
    )

    output = model(
        mouth_video=torch.randn(2, 5, 3, 8, 8),
        waveform=torch.randn(2, 100),
    )

    assert output.video_tokens.shape == (2, 5, 8)
    assert output.audio_tokens.shape == (2, 5, 8)
    assert output.offset_logits.shape == (2, 8)
    assert output.aligned_similarity.shape == (2, 5)


def test_nearest_temporal_tokens_supports_deterministic_backward() -> None:
    tokens = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1)
    tokens.requires_grad_()

    with torch.backends.cudnn.flags(deterministic=True):
        resized = _nearest_temporal_tokens(tokens, size=4)
        resized.sum().backward()

    assert resized[:, 0].item() == 0
    assert resized[:, -1].item() == 5
    assert tokens.grad is not None


def test_token_mask_maps_valid_samples_onto_the_token_axis() -> None:
    mask = token_mask_from_valid_samples(
        torch.tensor([100, 50, 0, 1]),
        total_samples=100,
        token_count=10,
    )

    # A full buffer keeps every token, half a buffer keeps half.
    assert mask[0].tolist() == [True] * 10
    assert mask[1].tolist() == [True] * 5 + [False] * 5
    # An empty clip still keeps one token: a row masked to nothing makes the
    # softmax produce NaN rather than a low score.
    assert mask[2].tolist() == [True] + [False] * 9
    # The token holding the boundary is kept rather than dropped.
    assert mask[3].tolist() == [True] + [False] * 9


def test_audio_branch_ignores_padding_when_given_valid_lengths() -> None:
    torch.manual_seed(17)
    model = AudioSpoofBranch(
        encoder=AudioBackbone(output_dim=10, token_count=10),
        encoder_dim=10,
        projection_dim=8,
    ).eval()

    content = torch.randn(1, 50)
    short = torch.cat([content, torch.zeros(1, 50)], dim=1)
    # The same audio followed by different padding. Masked, the padding is
    # outside the softmax, so both must score identically.
    other = torch.cat([content, torch.randn(1, 50)], dim=1)
    valid = torch.tensor([50])

    with torch.inference_mode():
        masked_short = model(short, valid_samples=valid).logits
        masked_other = model(other, valid_samples=valid).logits
        unmasked_short = model(short).logits
        unmasked_other = model(other).logits

    assert torch.allclose(masked_short, masked_other, atol=1e-6)
    # Without the mask the padding reaches the pooling, which is the shortcut
    # the mask exists to close.
    assert not torch.allclose(unmasked_short, unmasked_other, atol=1e-6)


def test_audio_branch_rejects_a_valid_length_batch_of_the_wrong_size() -> None:
    model = AudioSpoofBranch(
        encoder=AudioBackbone(output_dim=10, token_count=5),
        encoder_dim=10,
        projection_dim=8,
    )

    try:
        model(torch.randn(3, 100), valid_samples=torch.tensor([10, 10]))
    except ValueError as error:
        assert "batch size" in str(error)
    else:
        raise AssertionError("Expected a ValueError for a mismatched batch")


def test_visual_architecture_choices_match_the_builder() -> None:
    """The CLI spells its choices out to avoid importing torch to build a parser.

    That duplication is only safe while the two agree, so this is the check that
    makes it safe. Driven through the parser rather than its internals.
    """
    import contextlib
    import io

    from deepfake_detection.branches.visual import VISUAL_ARCHITECTURES
    from deepfake_detection.cli import build_parser

    base = [
        "train",
        "visual",
        "--train-manifest",
        "t.csv",
        "--validation-manifest",
        "v.csv",
        "--cache-index",
        "i.csv",
        "--cache-root",
        "c",
        "--dataset",
        "FakeAVCeleb",
        "--checkpoint",
        "m.pt",
        "--history",
        "h.json",
        "--run-id",
        "r",
        "--split-hash",
        "s",
        "--preprocessing-hash",
        "p",
    ]
    for architecture in VISUAL_ARCHITECTURES:
        parsed = build_parser().parse_args([*base, "--architecture", architecture])
        assert parsed.architecture == architecture

    # Anything the builder cannot build must be refused by the parser, so the
    # error arrives before a training run starts rather than after it loads.
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            build_parser().parse_args([*base, "--architecture", "resnet-50"])
        except SystemExit as exit_code:
            assert exit_code.code == 2
        else:
            raise AssertionError("Expected the parser to refuse resnet-50")


def test_visual_builder_rejects_an_unknown_architecture() -> None:
    from deepfake_detection.branches.visual import build_visual_branch

    try:
        build_visual_branch(architecture="resnet-50", pretrained=False)
    except ValueError as error:
        assert "Unsupported visual architecture" in str(error)
    else:
        raise AssertionError("Expected a ValueError for an unknown architecture")


def test_audio_branch_builds_each_checkpoint_as_its_own_architecture() -> None:
    """A speech encoder must be built through the class it was trained as.

    `Wav2Vec2Model.from_pretrained` accepts any checkpoint and drops whatever
    does not fit, so WavLM loads as a Wav2Vec2 body missing its gated relative
    position embeddings. That silently answers a different question than the
    AUD-01 comparison asks. Offline: config only, no weights downloaded.
    """
    transformers = pytest.importorskip("transformers")

    wav2vec2 = transformers.AutoConfig.from_pretrained("facebook/wav2vec2-base")
    wavlm = transformers.AutoConfig.from_pretrained("microsoft/wavlm-base")

    assert type(transformers.AutoModel.from_config(wav2vec2)).__name__ == (
        "Wav2Vec2Model"
    )
    assert type(transformers.AutoModel.from_config(wavlm)).__name__ == "WavLMModel"
