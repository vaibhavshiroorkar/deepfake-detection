"""Branch, stream and fusion training: `ddf train`."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

from deepfake_detection.cli.common import (
    _configuration_hash,
    _git_commit,
    _training_cost_metrics,
    _write_json,
    add_branch_arguments,
)
from deepfake_detection.cli.fusion import _train_fusion
from deepfake_detection.data.cache_index import read_cache_index
from deepfake_detection.data.guards import (
    reject_evaluation_only,
)
from deepfake_detection.data.manifest import load_manifest
from deepfake_detection.experiments import (
    NullRunLogger,
    runtime,
)
from deepfake_detection.experiments.training_log import (
    log_binary_training,
    log_sync_training,
)
from deepfake_detection.views.cache_store import CacheStore


def _visual_augmentation(arguments: argparse.Namespace):
    """The augmentation a visual run asked for, or None for the plain recipe."""
    from deepfake_detection.training.augment import VisualAugmentConfig

    compression = getattr(arguments, "augment_compression", 0.0)
    blend = getattr(arguments, "self_blend", 0.0)
    if compression == 0 and blend == 0:
        return None
    return VisualAugmentConfig(compression=compression, self_blend=blend)


def _binary_branch_train(arguments: argparse.Namespace) -> int:
    reject_evaluation_only(arguments.dataset, operation="training")
    started_at = time.perf_counter()
    runtime.seed_everything(arguments.seed, deterministic=True)
    runtime.require_research_cuda(arguments.device)

    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from deepfake_detection.branches.registry import build_branch
    from deepfake_detection.data.datasets import (
        CachedBranchDataset,
        collate_branch_items,
    )
    from deepfake_detection.training.augment import CodecSwap, VisualAugmenter
    from deepfake_detection.training.binary import (
        BinaryTrainingConfig,
        fit_binary_branch,
    )
    from deepfake_detection.training.checkpoints import (
        RunMetadata,
        hash_config,
        save_checkpoint,
    )

    train_records = load_manifest(
        arguments.train_manifest, dataset=arguments.dataset
    ).records
    validation_records = load_manifest(
        arguments.validation_manifest, dataset=arguments.dataset
    ).records
    index = read_cache_index(arguments.cache_index)
    cache_store = CacheStore(arguments.cache_root)
    augment_settings = _visual_augmentation(arguments)
    train_dataset = CachedBranchDataset(
        records=train_records,
        cache_index=index,
        cache_store=cache_store,
        branch=arguments.train_command,
        preprocessing_hash=arguments.preprocessing_hash,
        augment=(
            None
            if augment_settings is None
            else VisualAugmenter(augment_settings, seed=arguments.seed)
        ),
    )
    validation_dataset = CachedBranchDataset(
        records=validation_records,
        cache_index=index,
        cache_store=cache_store,
        branch=arguments.train_command,
        preprocessing_hash=arguments.preprocessing_hash,
        augment=CodecSwap()
        if getattr(arguments, "select_on_codec_swap", False)
        else None,
    )
    labels = [
        int(
            record.video_fake
            if arguments.train_command == "visual"
            else record.audio_fake
        )
        for record in train_records
    ]
    if set(labels) != {0, 1}:
        raise ValueError("Branch training requires both cue-specific classes")
    counts = {label: labels.count(label) for label in (0, 1)}
    weights = [1.0 / counts[label] for label in labels]
    generator = torch.Generator().manual_seed(arguments.seed)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=arguments.batch_size,
        sampler=sampler,
        collate_fn=collate_branch_items,
        num_workers=arguments.workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        collate_fn=collate_branch_items,
        num_workers=arguments.workers,
    )
    architecture = (
        arguments.architecture
        if arguments.train_command == "visual"
        else arguments.audio_model
    )
    model = build_branch(arguments.train_command, architecture, pretrained=True)
    model.masking_mode = (
        getattr(arguments, "masking_mode", "valid_samples")
        if arguments.train_command == "audio"
        else "legacy_none"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    config = BinaryTrainingConfig(
        epochs=arguments.epochs,
        accumulation_steps=arguments.accumulation_steps,
        freeze_epochs=arguments.freeze_epochs,
        early_stopping_patience=arguments.patience,
    )
    run_config = {
        "training": asdict(config),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": arguments.learning_rate,
            "weight_decay": arguments.weight_decay,
        },
        "data": {
            "batch_size": arguments.batch_size,
            "workers": arguments.workers,
            "sampler": "inverse_frequency_with_replacement",
            "augmentation": (
                None if augment_settings is None else asdict(augment_settings)
            ),
            "validation_codec_swap": getattr(arguments, "select_on_codec_swap", False),
        },
        "model": {
            "branch": arguments.train_command,
            "masking_mode": model.masking_mode,
            "audio_model": (
                arguments.audio_model if arguments.train_command == "audio" else None
            ),
            "pretrained": True,
        },
    }
    torch.cuda.reset_peak_memory_stats(arguments.device)
    training_started = time.perf_counter()
    history = fit_binary_branch(
        model=model,
        train_batches=train_loader,
        validation_batches=validation_loader,
        optimizer=optimizer,
        config=config,
        device=arguments.device,
    )
    training_cost = _training_cost_metrics(
        train_samples=len(train_dataset),
        completed_epochs=len(history.epochs),
        elapsed_seconds=time.perf_counter() - training_started,
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(arguments.device),
    )
    metadata = RunMetadata(
        run_id=arguments.run_id,
        branch=arguments.train_command,
        architecture=architecture,
        masking_mode=model.masking_mode,
        git_commit=_git_commit(),
        split_hash=arguments.split_hash,
        preprocessing_hash=arguments.preprocessing_hash,
        config_hash=_configuration_hash(arguments, run_config, hash_config),
        seed=arguments.seed,
        # Recorded so an out-of-fold export can verify its claim rather
        # than assert it. See `fusion/export.py::verify_out_of_fold`.
        trained_source_identities=tuple(
            sorted({record.source for record in train_records})
        ),
        selection_source_identities=tuple(
            sorted({record.source for record in validation_records})
        ),
    )
    checkpoint_hash = save_checkpoint(
        arguments.checkpoint,
        model=model,
        optimizer=optimizer,
        metadata=metadata,
        epoch=history.best_epoch,
    )
    _write_json(
        arguments.history,
        {
            "metadata": asdict(metadata),
            "config": run_config,
            "checkpoint_hash": checkpoint_hash,
            "best_epoch": history.best_epoch,
            "epochs": [asdict(epoch) for epoch in history.epochs],
            "hardware": training_cost,
        },
    )
    log_binary_training(
        getattr(arguments, "_run_logger", NullRunLogger()),
        history=history,
        configuration_hash=metadata.config_hash,
        checkpoint=arguments.checkpoint,
        history_path=arguments.history,
        elapsed_seconds=time.perf_counter() - started_at,
        samples_per_second=float(training_cost["samples_per_second"]),
        peak_gpu_memory_mib=float(training_cost["peak_gpu_memory_mib"]),
    )
    return 0


VISUAL_STREAM_PRESETS = ("dinov3", "efficientnet")


def _visual_stream_train(arguments: argparse.Namespace) -> int:
    """Train one configurable visual stream: DINOv3, EfficientNet-B0 or Xception.

    Parallel to `train visual`, which is welded to EfficientNet-B0 and returns a
    `BranchOutput`. This trains `streams/visual_stream.py`, which carries three
    backbones and projects each to a shared `common_dim` so several streams can
    be concatenated for feature-level fusion.

    `--freeze-backbone` is the option that matters. A fine-tuned backbone can
    absorb the corpus it trains on: the EfficientNet baseline reached 0.9742
    in-domain and then called 130 of 155 genuine Celeb-DF videos fake. A frozen
    self-supervised backbone never sees the corpus, so it cannot learn its
    artifacts, and only the head is fitted.
    """
    reject_evaluation_only(arguments.dataset, operation="training")
    started_at = time.perf_counter()
    runtime.seed_everything(arguments.seed, deterministic=True)
    runtime.require_research_cuda(arguments.device)

    import torch
    from torch.utils.data import DataLoader

    from deepfake_detection.data.datasets import (
        CachedBranchDataset,
        collate_branch_items,
    )
    from deepfake_detection.streams.config import (
        dinov3_config,
        efficientnet_config,
    )
    from deepfake_detection.streams.visual_stream import build_visual_stream
    from deepfake_detection.training.checkpoints import (
        RunMetadata,
        hash_config,
        save_checkpoint,
    )
    from deepfake_detection.training.streams import (
        StreamTrainingConfig,
        fit_visual_stream,
        parameter_groups,
    )

    presets = {
        "dinov3": dinov3_config,
        "efficientnet": efficientnet_config,
    }
    config = presets[arguments.backbone](
        pretrained=True,
        common_dim=arguments.common_dim,
        num_frames=arguments.num_frames,
        temporal_type=arguments.temporal,
        temporal_hidden=arguments.temporal_hidden,
        freeze_backbone=arguments.freeze_backbone,
        frame_chunk_size=arguments.frame_chunk_size,
        # Off deliberately. Peak VRAM measured 394 MiB of 16,303, so it saves
        # nothing here, and recomputing the forward during backward updates every
        # BatchNorm running statistic twice per step. The configuration measured
        # at 0.9058 had it off.
        grad_checkpointing=False,
        freeze_batchnorm_on_finetune=not arguments.live_batchnorm,
    )

    train_records = load_manifest(
        arguments.train_manifest, dataset=arguments.dataset
    ).records
    validation_records = load_manifest(
        arguments.validation_manifest, dataset=arguments.dataset
    ).records
    index = read_cache_index(arguments.cache_index)
    cache_store = CacheStore(arguments.cache_root)

    def dataset_for(records):
        return CachedBranchDataset(
            records=records,
            cache_index=index,
            cache_store=cache_store,
            branch="visual",
            preprocessing_hash=arguments.preprocessing_hash,
        )

    train_dataset = dataset_for(train_records)
    validation_dataset = dataset_for(validation_records)
    labels = [int(record.video_fake) for record in train_records]
    if set(labels) != {0, 1}:
        raise ValueError("Visual stream training needs both real and fake clips")

    # Inverse-frequency sampling, matching the branch trainers. FakeAVCeleb runs
    # roughly twenty to one fake, so uniform sampling shows a real clip about
    # once a batch.
    counts = {label: labels.count(label) for label in set(labels)}
    generator = torch.Generator()
    generator.manual_seed(arguments.seed)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=[1.0 / counts[label] for label in labels],
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=arguments.batch_size,
        sampler=sampler,
        collate_fn=collate_branch_items,
        num_workers=arguments.workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        collate_fn=collate_branch_items,
        num_workers=arguments.workers,
    )

    model = build_visual_stream(config)
    optimizer = torch.optim.AdamW(
        # A frozen backbone contributes no trainable parameters, so the group is
        # empty and only the head moves; the split still matters when it is not.
        parameter_groups(
            model,
            head_lr=arguments.learning_rate,
            encoder_lr=arguments.encoder_learning_rate,
            encoder_names=("backbone",),
        ),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    training = StreamTrainingConfig(
        epochs=arguments.epochs,
        accumulation_steps=arguments.accumulation_steps,
        freeze_epochs=arguments.epochs
        if arguments.freeze_backbone
        else arguments.freeze_epochs,
        early_stopping_patience=arguments.patience,
    )
    run_config = {
        "stream": f"visual-{arguments.backbone}",
        "dataset": arguments.dataset,
        "training": asdict(training),
        "optimizer": {
            "name": "adamw",
            "learning_rate": arguments.learning_rate,
            "encoder_learning_rate": arguments.encoder_learning_rate,
            "weight_decay": arguments.weight_decay,
        },
        "model": {
            "backbone": config.backbone_name,
            "global_pool": config.global_pool,
            "temporal": config.temporal_type,
            "temporal_hidden": config.temporal_hidden,
            "common_dim": config.common_dim,
            "frozen_backbone": arguments.freeze_backbone,
            "frozen_batchnorm": not arguments.live_batchnorm,
            "pretrained": True,
        },
    }

    torch.cuda.reset_peak_memory_stats(arguments.device)
    training_started = time.perf_counter()
    history = fit_visual_stream(
        model=model,
        train_batches=train_loader,
        validation_batches=validation_loader,
        optimizer=optimizer,
        config=training,
        device=arguments.device,
    )
    training_cost = _training_cost_metrics(
        train_samples=len(train_dataset),
        completed_epochs=len(history.epochs),
        elapsed_seconds=time.perf_counter() - training_started,
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(arguments.device),
    )
    metadata = RunMetadata(
        run_id=arguments.run_id,
        branch=f"visual-{arguments.backbone}",
        git_commit=_git_commit(),
        split_hash=arguments.split_hash,
        preprocessing_hash=arguments.preprocessing_hash,
        config_hash=_configuration_hash(arguments, run_config, hash_config),
        seed=arguments.seed,
        # Recorded so an out-of-fold export can verify its claim rather
        # than assert it. See `fusion/export.py::verify_out_of_fold`.
        trained_source_identities=tuple(
            sorted({record.source for record in train_records})
        ),
        selection_source_identities=tuple(
            sorted({record.source for record in validation_records})
        ),
    )
    checkpoint_hash = save_checkpoint(
        arguments.checkpoint,
        model=model,
        optimizer=optimizer,
        metadata=metadata,
        epoch=history.best_epoch,
    )
    best = history.epochs[history.best_epoch - 1]
    _write_json(
        arguments.history,
        {
            "metadata": asdict(metadata),
            "config": run_config,
            "checkpoint_hash": checkpoint_hash,
            "best_epoch": history.best_epoch,
            "epochs": [asdict(epoch) for epoch in history.epochs],
            "hardware": training_cost,
        },
    )
    logger = getattr(arguments, "_run_logger", NullRunLogger())
    logger.log_params({f"stream.{k}": v for k, v in run_config["model"].items()})
    logger.log_metrics(
        {
            "training.loss": best.train_loss,
            "validation.loss": best.validation_loss,
            "training.best_epoch": float(history.best_epoch),
            "training.samples_per_second": float(training_cost["samples_per_second"]),
            "training.peak_gpu_memory_mib": float(training_cost["peak_gpu_memory_mib"]),
            "training.elapsed_seconds": time.perf_counter() - started_at,
        }
    )
    logger.log_artifact(arguments.checkpoint, artifact_path="checkpoints")
    logger.log_artifact(arguments.history, artifact_path="history")
    return 0


def _stream_train(arguments: argparse.Namespace) -> int:
    """Train one audiovisual cross-attention stream.

    Parallel to `_binary_branch_train` rather than folded into it: the batch
    carries a video and an audio tensor instead of one, and the history records
    diagonal attention mass per epoch, which the binary history has no field
    for.
    """
    reject_evaluation_only(arguments.dataset, operation="training")
    started_at = time.perf_counter()
    runtime.seed_everything(arguments.seed, deterministic=True)
    runtime.require_research_cuda(arguments.device)

    import torch
    from torch.utils.data import DataLoader

    from deepfake_detection.data.datasets import (
        CachedAVPairDataset,
        collate_av_pair_items,
    )
    from deepfake_detection.streams.cross_modal_stream import (
        build_emotion_stream,
        build_lipsync_stream,
    )
    from deepfake_detection.training.checkpoints import (
        RunMetadata,
        hash_config,
        save_checkpoint,
    )
    from deepfake_detection.training.streams import (
        StreamTrainingConfig,
        fit_stream,
        parameter_groups,
    )

    train_records = load_manifest(
        arguments.train_manifest, dataset=arguments.dataset
    ).records
    validation_records = load_manifest(
        arguments.validation_manifest, dataset=arguments.dataset
    ).records
    index = read_cache_index(arguments.cache_index)
    cache_store = CacheStore(arguments.cache_root)

    def dataset_for(records):
        return CachedAVPairDataset(
            records=records,
            cache_index=index,
            cache_store=cache_store,
            stream=arguments.stream,
            preprocessing_hash=arguments.preprocessing_hash,
        )

    train_dataset = dataset_for(train_records)
    validation_dataset = dataset_for(validation_records)
    labels = [int(record.clip_fake) for record in train_records]
    if set(labels) != {0, 1}:
        raise ValueError("Stream training needs both real and fake clips")

    # Inverse-frequency sampling, matching the branch trainers. FakeAVCeleb is
    # roughly twenty to one fake, so uniform sampling would show the model a
    # real clip once a batch.
    counts = {label: labels.count(label) for label in set(labels)}
    generator = torch.Generator()
    generator.manual_seed(arguments.seed)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=[1.0 / counts[label] for label in labels],
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=arguments.batch_size,
        sampler=sampler,
        collate_fn=collate_av_pair_items,
        num_workers=arguments.workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        collate_fn=collate_av_pair_items,
        num_workers=arguments.workers,
    )

    # Each stream through its own builder. Both used the lip-sync builder
    # until the encoder became an option. For the default encoders the two
    # build the same modules and weights (the emotion clip is trimmed to 8
    # frames, so its comparison length was 8 either way), so earlier emotion
    # checkpoints load and score as before.
    builder = (
        build_lipsync_stream if arguments.stream == "lipsync" else build_emotion_stream
    )
    model = builder(
        video_backbone=arguments.video_backbone,
        audio_model=arguments.audio_model,
        pretrained=True,
        common_dim=arguments.common_dim,
        attention_heads=arguments.attention_heads,
        encoder=arguments.encoder,
    )
    optimizer = torch.optim.AdamW(
        parameter_groups(
            model,
            head_lr=arguments.learning_rate,
            encoder_lr=arguments.encoder_learning_rate,
        ),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    config = StreamTrainingConfig(
        epochs=arguments.epochs,
        accumulation_steps=arguments.accumulation_steps,
        freeze_epochs=arguments.freeze_epochs,
        early_stopping_patience=arguments.patience,
    )
    run_config = {
        "stream": arguments.stream,
        "dataset": arguments.dataset,
        "training": asdict(config),
        "optimizer": {
            "name": "adamw",
            "learning_rate": arguments.learning_rate,
            "encoder_learning_rate": arguments.encoder_learning_rate,
            "weight_decay": arguments.weight_decay,
        },
        "model": {
            "video_backbone": arguments.video_backbone,
            "audio_model": arguments.audio_model,
            "encoder": arguments.encoder,
            "common_dim": arguments.common_dim,
            "attention_heads": arguments.attention_heads,
            "temporal_position_mode": model.temporal_position_mode,
            "query_steps": model.query_steps,
            "pretrained": True,
        },
    }

    torch.cuda.reset_peak_memory_stats(arguments.device)
    training_started = time.perf_counter()
    history = fit_stream(
        model=model,
        train_batches=train_loader,
        validation_batches=validation_loader,
        optimizer=optimizer,
        config=config,
        device=arguments.device,
    )
    training_cost = _training_cost_metrics(
        train_samples=len(train_dataset),
        completed_epochs=len(history.epochs),
        elapsed_seconds=time.perf_counter() - training_started,
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(arguments.device),
    )
    metadata = RunMetadata(
        run_id=arguments.run_id,
        branch=arguments.stream,
        git_commit=_git_commit(),
        split_hash=arguments.split_hash,
        preprocessing_hash=arguments.preprocessing_hash,
        config_hash=_configuration_hash(arguments, run_config, hash_config),
        seed=arguments.seed,
        # Recorded so an out-of-fold export can verify its claim rather
        # than assert it. See `fusion/export.py::verify_out_of_fold`.
        trained_source_identities=tuple(
            sorted({record.source for record in train_records})
        ),
        selection_source_identities=tuple(
            sorted({record.source for record in validation_records})
        ),
    )
    checkpoint_hash = save_checkpoint(
        arguments.checkpoint,
        model=model,
        optimizer=optimizer,
        metadata=metadata,
        epoch=history.best_epoch,
    )
    best = history.epochs[history.best_epoch - 1]
    _write_json(
        arguments.history,
        {
            "metadata": asdict(metadata),
            "config": run_config,
            "checkpoint_hash": checkpoint_hash,
            "best_epoch": history.best_epoch,
            "epochs": [asdict(epoch) for epoch in history.epochs],
            "hardware": training_cost,
        },
    )

    logger = getattr(arguments, "_run_logger", NullRunLogger())
    logger.log_params(
        {f"stream.{key}": value for key, value in run_config["model"].items()}
    )
    logger.log_metrics(
        {
            "training.loss": best.train_loss,
            "validation.loss": best.validation_loss,
            # The stream's independent check. A falling loss beside a flat
            # diagonal mass means a shortcut was found, not synchronisation.
            "validation.diagonal_mass": best.validation_diagonal_mass,
            "training.best_epoch": float(history.best_epoch),
            "training.samples_per_second": float(training_cost["samples_per_second"]),
            "training.peak_gpu_memory_mib": float(training_cost["peak_gpu_memory_mib"]),
            "training.elapsed_seconds": time.perf_counter() - started_at,
        }
    )
    logger.log_artifact(arguments.checkpoint, artifact_path="checkpoints")
    logger.log_artifact(arguments.history, artifact_path="history")
    return 0


def _sync_branch_train(arguments: argparse.Namespace) -> int:
    reject_evaluation_only(arguments.dataset, operation="training")
    started_at = time.perf_counter()
    runtime.seed_everything(arguments.seed, deterministic=True)
    runtime.require_research_cuda(arguments.device)

    import torch
    from torch.utils.data import DataLoader

    from deepfake_detection.branches.sync import build_sync_branch
    from deepfake_detection.branches.sync_objective import OFFSET_MILLISECONDS
    from deepfake_detection.data.datasets import (
        CachedGlobalSyncDataset,
        CachedSyncDataset,
        collate_sync_items,
    )
    from deepfake_detection.training.checkpoints import (
        RunMetadata,
        hash_config,
        save_checkpoint,
    )
    from deepfake_detection.training.sync import SyncTrainingConfig, fit_sync_branch

    train_records = load_manifest(
        arguments.train_manifest, dataset=arguments.dataset
    ).records
    validation_records = load_manifest(
        arguments.validation_manifest, dataset=arguments.dataset
    ).records
    index = read_cache_index(arguments.cache_index)
    cache_store = CacheStore(arguments.cache_root)
    dataset_type = (
        CachedSyncDataset
        if arguments.label_mode == "authentic-offset"
        else CachedGlobalSyncDataset
    )
    dataset_arguments = {
        "cache_index": index,
        "cache_store": cache_store,
        "preprocessing_hash": arguments.preprocessing_hash,
    }
    train_dataset = dataset_type(
        records=train_records,
        **dataset_arguments,
        **({"sample_rate": 16_000} if dataset_type is CachedSyncDataset else {}),
    )
    validation_dataset = dataset_type(
        records=validation_records,
        **dataset_arguments,
        **({"sample_rate": 16_000} if dataset_type is CachedSyncDataset else {}),
    )
    generator = torch.Generator().manual_seed(arguments.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=arguments.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_sync_items,
        num_workers=arguments.workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        collate_fn=collate_sync_items,
        num_workers=arguments.workers,
    )
    model = build_sync_branch(audio_model_name=arguments.audio_model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    config = SyncTrainingConfig(
        epochs=arguments.epochs,
        accumulation_steps=arguments.accumulation_steps,
        heads_epochs=arguments.heads_epochs,
        early_stopping_patience=arguments.patience,
        contrastive_weight=arguments.contrastive_weight,
    )
    run_config = {
        "training": asdict(config),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": arguments.learning_rate,
            "weight_decay": arguments.weight_decay,
        },
        "data": {
            "batch_size": arguments.batch_size,
            "workers": arguments.workers,
            "offset_milliseconds": (
                list(OFFSET_MILLISECONDS)
                if arguments.label_mode == "authentic-offset"
                else []
            ),
            "mismatch": (
                "cross_identity"
                if arguments.label_mode == "authentic-offset"
                else "global_fake_label"
            ),
            "label_mode": arguments.label_mode,
        },
        "model": {
            "branch": "sync",
            "audio_model": arguments.audio_model,
            "pretrained": True,
        },
    }
    torch.cuda.reset_peak_memory_stats(arguments.device)
    training_started = time.perf_counter()
    history = fit_sync_branch(
        model=model,
        train_batches=train_loader,
        validation_batches=validation_loader,
        optimizer=optimizer,
        config=config,
        device=arguments.device,
    )
    training_cost = _training_cost_metrics(
        train_samples=len(train_dataset),
        completed_epochs=len(history.epochs),
        elapsed_seconds=time.perf_counter() - training_started,
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(arguments.device),
    )
    metadata = RunMetadata(
        run_id=arguments.run_id,
        branch="sync",
        architecture=arguments.audio_model,
        git_commit=_git_commit(),
        split_hash=arguments.split_hash,
        preprocessing_hash=arguments.preprocessing_hash,
        config_hash=_configuration_hash(arguments, run_config, hash_config),
        seed=arguments.seed,
        # Recorded so an out-of-fold export can verify its claim rather
        # than assert it. See `fusion/export.py::verify_out_of_fold`.
        trained_source_identities=tuple(
            sorted({record.source for record in train_records})
        ),
        selection_source_identities=tuple(
            sorted({record.source for record in validation_records})
        ),
    )
    checkpoint_hash = save_checkpoint(
        arguments.checkpoint,
        model=model,
        optimizer=optimizer,
        metadata=metadata,
        epoch=history.best_epoch,
    )
    _write_json(
        arguments.history,
        {
            "metadata": asdict(metadata),
            "config": run_config,
            "checkpoint_hash": checkpoint_hash,
            "best_epoch": history.best_epoch,
            "epochs": [asdict(epoch) for epoch in history.epochs],
            "hardware": training_cost,
        },
    )
    log_sync_training(
        getattr(arguments, "_run_logger", NullRunLogger()),
        history=history,
        configuration_hash=metadata.config_hash,
        checkpoint=arguments.checkpoint,
        history_path=arguments.history,
        elapsed_seconds=time.perf_counter() - started_at,
        samples_per_second=float(training_cost["samples_per_second"]),
        peak_gpu_memory_mib=float(training_cost["peak_gpu_memory_mib"]),
    )
    return 0


def register_train(commands) -> None:
    train = commands.add_parser("train")
    train_commands = train.add_subparsers(dest="train_command", required=True)
    train_fusion = train_commands.add_parser("fusion")
    train_fusion.add_argument("--feature-store", type=Path, required=True)
    train_fusion.add_argument("--output", type=Path, required=True)
    train_fusion.add_argument("--metadata", type=Path, required=True)
    train_fusion.add_argument(
        "--stream-checkpoints",
        type=Path,
        help="JSON mapping each stream to its final checkpoint path. Required for OOF deep fusion.",
    )
    train_fusion.add_argument(
        "--model",
        choices=("logistic", "mlp", "deep"),
        default="logistic",
        help=(
            "logistic and mlp read one calibrated scalar per branch; deep reads "
            "the whole embedding and trains the torch head in fusion/deep.py."
        ),
    )
    train_fusion.add_argument(
        "--branches",
        nargs="+",
        choices=(
            "visual",
            "audio",
            "sync",
            "lipsync",
            "emotion",
            "dinov3",
            "efficientnet",
        ),
        default=("visual", "audio", "sync"),
    )
    train_fusion.add_argument(
        "--drop-incomplete",
        action="store_true",
        help=(
            "Fit the scalar head on the clips every stream could read instead "
            "of refusing the store. The metadata records how many were dropped."
        ),
    )
    # Deep fusion only. Ignored by the scalar path, which fits in closed form.
    train_fusion.add_argument("--epochs", type=int, default=200)
    train_fusion.add_argument("--batch-size", type=int, default=256)
    train_fusion.add_argument("--learning-rate", type=float, default=1e-3)
    train_fusion.add_argument("--weight-decay", type=float, default=1e-4)
    train_fusion.add_argument(
        "--stream-dropout",
        type=float,
        default=0.2,
        help=(
            "Chance of zeroing a whole stream during training, so the head "
            "cannot build its decision on one stream always being present."
        ),
    )
    train_fusion.add_argument("--dropout", type=float, default=0.2)
    train_fusion.add_argument("--hidden-sizes", type=int, nargs="+", default=(128,))
    train_fusion.add_argument("--common-dim", type=int, default=256)
    train_fusion.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Held out by source identity, never by clip.",
    )
    train_fusion.add_argument("--patience", type=int, default=20)
    train_fusion.add_argument("--seed", type=int, default=17)
    train_fusion.add_argument("--device", default="cpu")
    train_fusion.set_defaults(handler=_train_fusion)
    train_visual = train_commands.add_parser("visual")
    add_branch_arguments(train_visual)
    train_visual.add_argument("--freeze-epochs", type=int, default=3)
    # Spelled out rather than imported from branches/visual.py, which pulls in
    # torch and would pay that import on every CLI invocation just to build the
    # parser. `test_visual_architecture_choices_match_the_builder` fails if the
    # two ever drift.
    train_visual.add_argument(
        "--architecture",
        choices=("convnext-tiny", "efficientnet-b0"),
        default="efficientnet-b0",
        help=(
            "Visual backbone. The default is the frozen baseline; convnext-tiny "
            "is the VIS-01 challenger, compared under the same training budget."
        ),
    )
    train_visual.add_argument(
        "--augment-compression",
        type=float,
        default=0.0,
        help=(
            "Probability of a random JPEG, H.264 or MPEG-4 round trip per clip, "
            "applied to both classes. Any value above zero also turns on flips, "
            "colour, blur and downscaling. Zero keeps the original recipe."
        ),
    )
    train_visual.add_argument(
        "--self-blend",
        type=float,
        default=0.0,
        help="Probability that a genuine clip becomes a self-blended fake.",
    )
    train_visual.add_argument(
        "--select-on-codec-swap",
        action="store_true",
        help=(
            "Choose the epoch on validation clips re-encoded with the other "
            "class's codec, so selection rewards reading faces, not encoders."
        ),
    )
    train_visual.set_defaults(handler=_binary_branch_train)
    train_audio = train_commands.add_parser("audio")
    train_audio.add_argument(
        "--masking-mode",
        choices=("legacy_none", "valid_samples"),
        default="valid_samples",
    )
    add_branch_arguments(train_audio)
    train_audio.add_argument("--freeze-epochs", type=int, default=3)
    train_audio.add_argument(
        "--audio-model",
        default="facebook/wav2vec2-base",
    )
    train_audio.set_defaults(handler=_binary_branch_train)
    train_visual_stream = train_commands.add_parser("visual-stream")
    add_branch_arguments(train_visual_stream)
    train_visual_stream.add_argument(
        "--backbone", choices=VISUAL_STREAM_PRESETS, default="efficientnet"
    )
    train_visual_stream.add_argument("--freeze-epochs", type=int, default=3)
    train_visual_stream.add_argument(
        "--freeze-backbone",
        action="store_true",
        help=(
            "Never unfreeze the backbone. Intended for DINOv3, whose "
            "self-supervised features are the reason to use it."
        ),
    )
    train_visual_stream.add_argument(
        "--live-batchnorm",
        action="store_true",
        help=(
            "Let the backbone's BatchNorm keep updating while fine-tuning. "
            "Freezing it is the default and buys in-domain accuracy: measured "
            "0.9197 validation against 0.5267. It also costs cross-corpus "
            "transfer, measured 0.5005 on DFDC against 0.6482. Pick the setting "
            "for the question being asked and report which was used."
        ),
    )
    train_visual_stream.add_argument("--common-dim", type=int, default=256)
    train_visual_stream.add_argument("--num-frames", type=int, default=16)
    train_visual_stream.add_argument(
        "--temporal", choices=("lstm", "gru", "mean"), default="lstm"
    )
    train_visual_stream.add_argument("--temporal-hidden", type=int, default=256)
    train_visual_stream.add_argument(
        "--frame-chunk-size",
        type=int,
        default=8,
        help="Frames pushed through the backbone at once, to bound VRAM.",
    )
    train_visual_stream.add_argument(
        "--encoder-learning-rate",
        type=float,
        default=5e-6,
        help="Learning rate for the pretrained backbone, kept below the head's.",
    )
    train_visual_stream.set_defaults(handler=_visual_stream_train)
    train_stream = train_commands.add_parser("stream")
    add_branch_arguments(train_stream)
    train_stream.add_argument(
        "--stream", choices=("lipsync", "emotion"), default="lipsync"
    )
    train_stream.add_argument("--freeze-epochs", type=int, default=3)
    train_stream.add_argument(
        "--video-backbone", default="tf_efficientnet_b0.ns_jft_in1k"
    )
    train_stream.add_argument("--audio-model", default="facebook/wav2vec2-base")
    train_stream.add_argument(
        "--encoder",
        choices=("default", "avhubert", "hsemotion"),
        default="default",
        help=(
            "avhubert reads lips and voice through frozen AV-HuBERT (lipsync "
            "only); hsemotion starts the face side from HSEmotion's expression "
            "weights (emotion only). Both need their weights under models/."
        ),
    )
    train_stream.add_argument("--common-dim", type=int, default=256)
    train_stream.add_argument("--attention-heads", type=int, default=4)
    train_stream.add_argument(
        "--encoder-learning-rate",
        type=float,
        default=5e-6,
        help="Learning rate for the pretrained encoders, kept well below the head's.",
    )
    train_stream.set_defaults(handler=_stream_train)
    train_sync = train_commands.add_parser("sync")
    add_branch_arguments(train_sync)
    train_sync.add_argument("--heads-epochs", type=int, default=3)
    train_sync.add_argument("--contrastive-weight", type=float, default=0.1)
    train_sync.add_argument(
        "--label-mode",
        choices=("authentic-offset", "global-fake"),
        default="authentic-offset",
    )
    train_sync.add_argument(
        "--audio-model",
        default="facebook/wav2vec2-base",
    )
    train_sync.set_defaults(handler=_sync_branch_train)
