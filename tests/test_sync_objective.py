import torch

from deepfake_detection.branches.sync_objective import (
    OFFSET_MILLISECONDS,
    contrastive_alignment_loss,
    crop_audio_context,
)


def test_offset_classes_match_the_approved_training_offsets() -> None:
    assert OFFSET_MILLISECONDS == (-320, -160, -80, 0, 80, 160, 320)


def test_contrastive_loss_rewards_matching_pairs() -> None:
    video = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    aligned_audio = video.clone()
    mismatched_audio = video.flip(0)

    aligned_loss = contrastive_alignment_loss(video, aligned_audio, temperature=0.1)
    mismatched_loss = contrastive_alignment_loss(
        video, mismatched_audio, temperature=0.1
    )

    assert aligned_loss.item() < mismatched_loss.item()


def test_offset_crop_uses_real_context_without_padding_edges() -> None:
    context = torch.arange(10, dtype=torch.float32).unsqueeze(0)

    centered = crop_audio_context(
        context,
        output_samples=4,
        offset_ms=0,
        sample_rate=1_000,
    )
    shifted = crop_audio_context(
        context,
        output_samples=4,
        offset_ms=2,
        sample_rate=1_000,
    )

    assert centered.tolist() == [[3.0, 4.0, 5.0, 6.0]]
    assert shifted.tolist() == [[5.0, 6.0, 7.0, 8.0]]
