"""The audiovisual stream's contract, checked without downloading weights.

Every model here is tiny and `pretrained=False`, following
`tests/test_streams_visual.py`: the suite must not pull EfficientNet or
Wav2Vec2 weights over the network to run.
"""

import pytest

pytest.importorskip("torch")
import torch

from deepfake_detection.streams.config import StreamConfig
from deepfake_detection.streams.cross_modal_stream import (
    CrossModalStream,
    resample_tokens,
)

VIDEO_STEPS = 50
AUDIO_SAMPLES = 32_000


def tiny_stream(*, heads: int = 4, dim: int = 32) -> CrossModalStream:
    config = StreamConfig(
        stream_name="lipsync",
        backbone_name="resnet18",
        pretrained=False,
        common_dim=dim,
        image_size=112,
        frame_chunk_size=25,
    )
    return CrossModalStream(
        config=config, pretrained=False, attention_heads=heads, query_steps=VIDEO_STEPS
    )


def inputs(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    video = torch.randn(batch, VIDEO_STEPS, 3, 112, 112, generator=generator)
    audio = torch.randn(batch, AUDIO_SAMPLES, generator=generator)
    return video, audio


def test_stream_returns_the_documented_shapes() -> None:
    model = tiny_stream().eval()
    video, audio = inputs()

    with torch.no_grad():
        out = model(video=video, audio=audio)

    assert out.logit.shape == (2,)
    assert out.embedding.shape == (2, 32)
    assert out.attention.shape == (2, VIDEO_STEPS, VIDEO_STEPS)
    assert out.diagonal_mass.shape == (2,)


def test_attention_rows_are_a_distribution_over_video_steps() -> None:
    """Softmax runs over the key axis, so each audio step's attention across the
    mouth sequence sums to one. If this ever normalises the other way the
    diagonal_mass reading becomes meaningless."""
    model = tiny_stream().eval()
    video, audio = inputs()

    with torch.no_grad():
        out = model(video=video, audio=audio)

    assert torch.allclose(
        out.attention.sum(dim=-1), torch.ones(2, VIDEO_STEPS), atol=1e-5
    )


def relative_change(first: torch.Tensor, second: torch.Tensor) -> float:
    """Largest change as a fraction of the tensor's own scale.

    Absolute tolerances are useless here: attention weights live at 1/50 = 0.02,
    so a 10% change is 0.002 and would pass any sane `atol`.
    """
    return float((first - second).abs().max() / first.abs().max())


def test_embedding_depends_on_the_audio() -> None:
    """The wiring check that matters. If shifting the audio leaves the embedding
    untouched, the query is disconnected and no training would fix it.

    This test caught a real design fault. Pooling the attended vector alone gave
    a change of 2e-7, float noise, because near-uniform attention at
    initialisation collapses every attended step onto the mean video token.
    Taking the residual `audio - attended` instead, which is what
    `stream_spec.py` means by "mismatch vector", moved it to 7%.
    """
    model = tiny_stream().eval()
    video, audio = inputs(batch=1)
    shifted = torch.roll(audio, shifts=5120, dims=1)  # 320 ms at 16 kHz

    with torch.no_grad():
        first = model(video=video, audio=audio)
        second = model(video=video, audio=shifted)

    assert relative_change(first.embedding, second.embedding) > 0.01


def test_embedding_depends_on_the_video_content() -> None:
    """The video path is connected and reaches the embedding.

    The second clip is scaled and offset rather than merely reseeded. An
    untrained ResNet in eval mode barely separates one field of Gaussian noise
    from another: measured token spread across time was 0.011 against the audio
    encoder's 0.587, so two reseeded noise clips move the embedding by 0.06%.
    That is a property of an untrained backbone, not of the wiring, and a
    distribution-level difference is what shows the path is live without
    pulling pretrained weights into the test suite.
    """
    model = tiny_stream().eval()
    video, audio = inputs(batch=1)
    other = (
        torch.randn(
            1, VIDEO_STEPS, 3, 112, 112, generator=torch.Generator().manual_seed(99)
        )
        * 2.0
        + 0.5
    )

    with torch.no_grad():
        first = model(video=video, audio=audio)
        second = model(video=other, audio=audio)

    assert relative_change(first.embedding, second.embedding) > 0.01


def test_frame_order_is_ignored_until_attention_sharpens() -> None:
    """Documents a real property rather than asserting a wish.

    With attention uniform at initialisation the attended vector is the mean
    over frames, and a mean cannot see order. So an untrained stream is
    order-blind by construction. Becoming order-sensitive is exactly what
    training has to achieve, which makes this the check to re-run on a trained
    checkpoint: there the assertion should invert.
    """
    model = tiny_stream().eval()
    video, audio = inputs(batch=1)

    with torch.no_grad():
        first = model(video=video, audio=audio)
        second = model(video=video.flip(dims=[1]), audio=audio)

    assert relative_change(first.embedding, second.embedding) < 1e-5


def test_untrained_attention_sits_near_chance() -> None:
    """An untrained stream must not already look synchronised, or diagonal_mass
    would be measuring the architecture rather than what was learned."""
    model = tiny_stream().eval()
    video, audio = inputs()

    with torch.no_grad():
        out = model(video=video, audio=audio)

    # Band of 1 over 50 steps is 3 cells per row, so chance is about 0.06.
    assert float(out.diagonal_mass.mean()) < 0.15


def test_freezing_both_encoders_shrinks_the_trainable_count() -> None:
    model = tiny_stream()
    before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model.set_backbone_trainable(False)
    after = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert after < before
    # The projections, attention and head stay trainable.
    assert after > 0


def test_chunking_does_not_change_the_output() -> None:
    video, audio = inputs(batch=1)
    torch.manual_seed(29)
    chunked = tiny_stream().eval()
    plain = tiny_stream().eval()
    plain.load_state_dict(chunked.state_dict())
    plain.config.frame_chunk_size = 0

    with torch.no_grad():
        first = chunked(video=video, audio=audio)
        second = plain(video=video, audio=audio)

    assert torch.allclose(first.embedding, second.embedding, atol=1e-5)


def test_resample_picks_evenly_spaced_positions() -> None:
    tokens = torch.arange(100, dtype=torch.float32).reshape(1, 100, 1)

    picked = resample_tokens(tokens, size=5).squeeze()

    assert picked.tolist() == [0.0, 25.0, 50.0, 74.0, 99.0]


def test_resample_is_a_noop_at_the_same_length() -> None:
    tokens = torch.randn(2, 40, 8)
    assert resample_tokens(tokens, size=40) is tokens


def test_resample_refuses_to_invent_tokens() -> None:
    """Upsampling would fabricate timesteps that the encoder never produced."""
    with pytest.raises(ValueError, match="shorter sequence than the target"):
        resample_tokens(torch.randn(1, 10, 4), size=50)


def test_stream_rejects_a_waveform_with_the_wrong_rank() -> None:
    model = tiny_stream().eval()
    video, _ = inputs(batch=1)

    with pytest.raises(ValueError, match=r"\[batch, samples\]"):
        model(video=video, audio=torch.randn(1, 1, AUDIO_SAMPLES))
