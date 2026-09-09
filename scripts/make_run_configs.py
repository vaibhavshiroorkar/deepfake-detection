"""Generate the run configs for the pilot and full tiers.

Written rather than hand-maintained because the two tiers differ only in which
manifests they point at, and the ablation grid multiplies quickly. Hand-copying
a YAML per variant is how `runs/comparison-20260904` ended up with four files
that differ in one line each and no way to tell they were one study.

Every run gets `ablation_group`, `tier` and `dataset` tags, so MLflow can be
queried for a study instead of the study existing only in the file names.

    uv run python scripts/make_run_configs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "runs" / "full-20260904"
PROJECT = "C:/Users/vaibh/Documents/GitHub/deepfake-generalization"

EXPERIMENT = "full-20260904"
SPLIT_HASH = json.loads((HERE / "split" / "audit.json").read_text())["split_hash"]
# Recorded by the cache build; every training run must match it or the cached
# views it reads are not the ones it thinks it is reading.
PREPROCESSING_HASH = "ac79f71e7614c96610e897eb011e01129b193da4482f415d037c6cc8c17638ec"

SEEDS = (17, 29, 43)


def tracking(run_name: str, *, scope: str, group: str, tier: str) -> dict:
    return {
        "enabled": True,
        "tracking_uri": f"sqlite:///{PROJECT}/mlflow.db",
        "artifact_root": f"{PROJECT}/mlartifacts",
        "experiment_name": EXPERIMENT,
        "run_name": run_name,
        "tags": {
            "project": "deepfake-generalization",
            "environment": "local",
            "evidence_scope": scope,
            "dataset": "FakeAVCeleb",
            # The three tags that turn a pile of runs into a queryable study.
            "ablation_group": group,
            "tier": tier,
            "split_hash": SPLIT_HASH,
        },
    }


def branch_arguments(branch: str, name: str, tier: str, seed: int) -> dict:
    split = "pilot-split" if tier == "pilot" else "split"
    # The "-usable" manifests, written by `ddf manifest usable`, exclude clips
    # whose view this branch cannot read. A clip with an unstable primary face
    # track has no visual view at all, and the loader raises on it mid-epoch.
    suffix = "usable" if branch == "visual" else f"{branch}-usable"
    return {
        "train-manifest": f"runs/{EXPERIMENT}/{split}/train-{suffix}.csv",
        "validation-manifest": f"runs/{EXPERIMENT}/{split}/val-{suffix}.csv",
        "cache-index": f"runs/{EXPERIMENT}/cache-index.csv",
        "cache-root": f"runs/{EXPERIMENT}/cache",
        "dataset": "FakeAVCeleb",
        "checkpoint": f"runs/{EXPERIMENT}/checkpoints/{name}.pt",
        "history": f"runs/{EXPERIMENT}/checkpoints/{name}-history.json",
        "run-id": "configured-by-mlflow",
        "split-hash": SPLIT_HASH,
        "preprocessing-hash": PREPROCESSING_HASH,
        "device": "cuda",
        "batch-size": 8,
        "accumulation-steps": 2,
        "learning-rate": 0.0001,
        "weight-decay": 0.0001,
        "patience": 2,
        "workers": 0,
        "seed": seed,
    }


def visual(tier: str, seed: int) -> tuple[str, dict]:
    name = f"{tier}-visual-efficientnet-b0-seed{seed}"
    arguments = branch_arguments("visual", name, tier, seed)
    arguments["epochs"] = 8 if tier == "pilot" else 12
    arguments["freeze-epochs"] = 1
    return name, {
        "schema_version": 1,
        "command": ["train", "visual"],
        "arguments": arguments,
        "tracking": tracking(
            name,
            scope="development_comparison",
            group=f"{tier}-visual-seeds",
            tier=tier,
        ),
    }


def audio(tier: str, seed: int) -> tuple[str, dict]:
    name = f"{tier}-audio-wav2vec2-base-seed{seed}"
    arguments = branch_arguments("audio", name, tier, seed)
    arguments["epochs"] = 6 if tier == "pilot" else 10
    arguments["freeze-epochs"] = 1
    arguments["audio-model"] = "facebook/wav2vec2-base"
    return name, {
        "schema_version": 1,
        "command": ["train", "audio"],
        "arguments": arguments,
        "tracking": tracking(
            name,
            scope="development_comparison",
            group=f"{tier}-audio-seeds",
            tier=tier,
        ),
    }


def sync(tier: str, seed: int, *, label_mode: str = "authentic-offset") -> tuple[str, dict]:
    suffix = "" if label_mode == "authentic-offset" else f"-{label_mode}"
    name = f"{tier}-sync-temporal{suffix}-seed{seed}"
    arguments = branch_arguments("sync", name, tier, seed)
    arguments["epochs"] = 6 if tier == "pilot" else 10
    arguments["heads-epochs"] = 1
    arguments["contrastive-weight"] = 0.1
    arguments["label-mode"] = label_mode
    arguments["audio-model"] = "facebook/wav2vec2-base"
    # Sync holds two encoders and temporal tokens, so it needs a smaller batch.
    arguments["batch-size"] = 4
    arguments["accumulation-steps"] = 4
    return name, {
        "schema_version": 1,
        "command": ["train", "sync"],
        "arguments": arguments,
        "tracking": tracking(
            name,
            scope="development_comparison",
            group=f"{tier}-sync-label-mode" if suffix else f"{tier}-sync-seeds",
            tier=tier,
        ),
    }


def build(tier: str) -> list[str]:
    """Every config for one tier, written to runs/<experiment>/configs/."""
    directory = HERE / "configs"
    directory.mkdir(parents=True, exist_ok=True)
    seeds = SEEDS if tier == "pilot" else SEEDS
    written = []
    entries: list[tuple[str, dict]] = []
    for seed in seeds:
        entries.append(visual(tier, seed))
    # Audio and sync are the expensive branches, so only the first seed runs in
    # the pilot. The remaining seeds are queued in the full tier.
    audio_seeds = (17,) if tier == "pilot" else SEEDS
    sync_seeds = (17,) if tier == "pilot" else SEEDS
    for seed in audio_seeds:
        entries.append(audio(tier, seed))
    for seed in sync_seeds:
        entries.append(sync(tier, seed))
    if tier == "full":
        # The sync label-mode ablation from docs/research-design.md.
        entries.append(sync(tier, 17, label_mode="global-fake"))

    for name, document in entries:
        path = directory / f"{name}.yaml"
        path.write_text(
            yaml.safe_dump(document, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        written.append(str(path.relative_to(ROOT)).replace("\\", "/"))
    return written


def main() -> None:
    for tier in ("pilot", "full"):
        for path in build(tier):
            print(path)


if __name__ == "__main__":
    main()
