import torch
from torch import nn

from deepfake_detection.branches.contracts import BranchOutput
from deepfake_detection.branches.sync import SyncOutput
from deepfake_detection.data.datasets import (
    BranchBatch,
    BranchItem,
    SyncBatch,
    collate_branch_items,
)
from deepfake_detection.training.binary import BinaryTrainingConfig, fit_binary_branch
from deepfake_detection.training.losses import sync_training_loss
from deepfake_detection.training.sync import SyncTrainingConfig, fit_sync_branch
from deepfake_detection.views.contracts import QualityReport


def test_branch_collation_stacks_values_and_cue_labels() -> None:
    quality = QualityReport(1.0, True, True, False, 0.0)
    items = [
        BranchItem("a", torch.ones(2), torch.tensor(0.0), quality),
        BranchItem("b", torch.zeros(2), torch.tensor(1.0), quality),
    ]

    batch = collate_branch_items(items)

    assert batch.clip_ids == ("a", "b")
    assert batch.values.shape == (2, 2)
    assert batch.labels.tolist() == [0.0, 1.0]


def test_sync_loss_rewards_the_correct_offset_class() -> None:
    tokens = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    correct_logits = torch.zeros(2, 8)
    correct_logits[:, 3] = 5.0
    wrong_logits = torch.zeros(2, 8)
    wrong_logits[:, 0] = 5.0
    correct = SyncOutput(tokens, tokens, correct_logits, torch.ones(2, 1))
    wrong = SyncOutput(tokens, tokens, wrong_logits, torch.ones(2, 1))
    targets = torch.tensor([3, 3])

    correct_loss = sync_training_loss(correct, targets).total
    wrong_loss = sync_training_loss(wrong, targets).total

    assert correct_loss.item() < wrong_loss.item()


def test_sync_loss_uses_shifted_same_clip_pairs_for_contrastive_learning() -> None:
    video = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    audio = video.flip(0)
    output = SyncOutput(video, audio, torch.zeros(2, 8), torch.zeros(2, 1))
    shifted_targets = torch.tensor([0, 6])

    without_contrastive = sync_training_loss(
        output,
        shifted_targets,
        contrastive_weight=0.0,
    )
    with_contrastive = sync_training_loss(
        output,
        shifted_targets,
        contrastive_weight=1.0,
    )

    assert with_contrastive.contrastive.item() > 0
    assert with_contrastive.total.item() > without_contrastive.total.item()


class TinyBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(1, 1, bias=False)

    def forward(self, values: torch.Tensor) -> BranchOutput:
        logits = self.backbone(values).squeeze(-1)
        return BranchOutput(logits=logits, embedding=values, token_count=1)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable


def test_binary_training_smoke_run_updates_weights_with_large_accumulation() -> None:
    model = TinyBranch()
    nn.init.zeros_(model.backbone.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batches = tuple(
        BranchBatch((str(index),), torch.ones(1, 1), torch.ones(1))
        for index in range(3)
    )

    history = fit_binary_branch(
        model=model,
        train_batches=batches,
        validation_batches=batches,
        optimizer=optimizer,
        config=BinaryTrainingConfig(
            epochs=1,
            accumulation_steps=8,
            freeze_epochs=0,
            early_stopping_patience=2,
        ),
        device="cpu",
    )

    assert model.backbone.weight.item() > 0
    assert history.epochs[0].optimizer_steps == 1


class TinySync(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.video_encoder = nn.Linear(1, 1)
        self.audio_encoder = nn.Linear(1, 1)
        self.offset_head = nn.Linear(1, 8)

    def forward(
        self, *, mouth_video: torch.Tensor, waveform: torch.Tensor
    ) -> SyncOutput:
        batch = mouth_video.shape[0]
        summary = mouth_video.mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(-1)
        tokens = summary.unsqueeze(1)
        return SyncOutput(
            video_tokens=tokens,
            audio_tokens=tokens,
            offset_logits=self.offset_head(summary),
            aligned_similarity=torch.ones(batch, 1),
        )


def test_sync_training_smoke_run_updates_the_offset_head() -> None:
    model = TinySync()
    nn.init.zeros_(model.offset_head.weight)
    nn.init.zeros_(model.offset_head.bias)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = SyncBatch(
        clip_ids=("real-1",),
        mouth_video=torch.ones(1, 2, 3, 4, 4),
        waveform=torch.ones(1, 16),
        offset_classes=torch.tensor([3]),
    )

    history = fit_sync_branch(
        model=model,
        train_batches=(batch,),
        validation_batches=(batch,),
        optimizer=optimizer,
        config=SyncTrainingConfig(
            epochs=1,
            accumulation_steps=8,
            heads_epochs=1,
            early_stopping_patience=2,
        ),
        device="cpu",
    )

    assert history.epochs[0].optimizer_steps == 1
    assert model.offset_head.bias[3].item() > 0
