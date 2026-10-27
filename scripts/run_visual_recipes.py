"""Train and score the visual branch under recipes that target measured failures.

Each arm changes one thing against the one before it, so a difference between
neighbouring arms has one cause:

    plain        the shipped recipe, now selected on validation ROC-AUC
    compress     + random JPEG, H.264 and MPEG-4 round trips on both classes,
                   and checkpoint selection on codec-swapped validation clips
    blend        + self-blended fakes made from genuine clips
    blend-ffpp   + FaceForensics++ training clips, which bring 645 genuine
                   source videos to FakeAVCeleb's 349

Every arm trains on the same cached crops, validates on the same FakeAVCeleb
clips, and is scored on the same partitions as `program-20260906`: the
FakeAVCeleb test split, DFDC, Celeb-DF and the FaceForensics++ test split. DFDC
and Celeb-DF are never used for training, selection or thresholds.

Arms resume only when their completion receipts match the current inputs.

    uv run python scripts/run_visual_recipes.py --seeds 17
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from research_artifacts import (
    cache_payload_digests,
    completed,
    record_completion,
    sha256,
)

PROGRAM = Path("runs/program-20260906")
FFPP = Path("runs/ffpp-20260913")

ARMS = {
    "plain": [],
    "compress": ["--augment-compression", "0.8", "--select-on-codec-swap"],
    "blend": [
        "--augment-compression",
        "0.8",
        "--self-blend",
        "0.5",
        "--select-on-codec-swap",
    ],
    "blend-ffpp": [
        "--augment-compression",
        "0.8",
        "--self-blend",
        "0.5",
        "--select-on-codec-swap",
    ],
}

EVALUATIONS = (
    # Validation is scored so a threshold can be chosen without touching a test
    # partition.
    (
        "validation",
        PROGRAM / "split" / "val-usable.csv",
        "FakeAVCeleb",
        "development_validation",
    ),
    (
        "in-domain-test",
        PROGRAM / "split" / "test-usable.csv",
        "FakeAVCeleb",
        "development_test",
    ),
    (
        "dfdc",
        Path("runs/streams-20260905/dfdc-manifest.csv"),
        "DFDC",
        "generalization_celebdf",
    ),
    (
        "celebdf",
        Path("runs/full-20260904/celebdf-test-manifest.csv"),
        "Celeb-DF-v2",
        "generalization_celebdf",
    ),
    (
        "ffpp-test",
        FFPP / "split" / "test-usable.csv",
        "FaceForensics++",
        "development_comparison",
    ),
)


def _merge_csv(sources: list[Path], target: Path) -> None:
    rows: list[dict[str, str]] = []
    fields: list[str] | None = None
    for source in sources:
        with source.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = fields or list(reader.fieldnames or [])
            rows.extend(reader)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(run: Path) -> str:
    """Merged cache index and training manifest. Returns the preprocessing hash."""
    from deepfake_detection.data.cache_index import read_cache_index, write_cache_index

    hashes = {
        json.loads((source / "cache-audit.json").read_text())["preprocessing_hash"]
        for source in (PROGRAM, FFPP)
    }
    if len(hashes) != 1:
        raise SystemExit(f"The two caches disagree on preprocessing: {hashes}")
    index = run / "cache-index.csv"
    if not index.is_file():
        merged = read_cache_index(PROGRAM / "cache-index.csv")
        merged.update(read_cache_index(FFPP / "cache-index.csv"))
        write_cache_index(index, merged)
    mixed = run / "split" / "train-fav-ffpp.csv"
    if not mixed.is_file():
        _merge_csv(
            [
                PROGRAM / "split" / "train-usable.csv",
                FFPP / "split" / "train-usable.csv",
            ],
            mixed,
        )
    return hashes.pop()


def _cli(*arguments: str, log: Path) -> int:
    command = [sys.executable, "-m", "deepfake_detection.cli", *arguments]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(" ".join(command) + "\n")
        handle.flush()
        return subprocess.run(  # noqa: S603
            command, stdout=handle, stderr=subprocess.STDOUT
        ).returncode


def train(run: Path, arm: str, seed: int, preprocessing_hash: str, epochs: int) -> bool:
    name = f"{arm}-seed{seed}"
    checkpoint = run / "checkpoints" / f"{name}.pt"
    history = run / "checkpoints" / f"{name}-history.json"
    receipt = run / "checkpoints" / f"{name}-complete.json"
    # Reject legacy or interrupted checkpoints before reading their inputs.
    if checkpoint.exists() and not receipt.exists():
        raise ValueError(
            f"Unverified checkpoint has no completion receipt: {checkpoint}"
        )
    manifest = (
        run / "split" / "train-fav-ffpp.csv"
        if arm.endswith("ffpp")
        else PROGRAM / "split" / "train-usable.csv"
    )
    configuration = {
        "arm": arm,
        "seed": seed,
        "epochs": epochs,
        "preprocessing_hash": preprocessing_hash,
        "arguments": ARMS[arm],
        "train_manifest_sha256": sha256(manifest),
        "validation_manifest_sha256": sha256(PROGRAM / "split" / "val-usable.csv"),
        "cache_index_sha256": sha256(run / "cache-index.csv"),
        "cache_payloads": cache_payload_digests(
            [
                manifest,
                PROGRAM / "split" / "val-usable.csv",
                run / "cache-index.csv",
            ]
        ),
    }
    if completed(receipt, configuration, [checkpoint, history]):
        print(f"{name}: verified completed training, skipping", flush=True)
        return True
    for workers in ("2", "0"):
        code = _cli(
            "train",
            "visual",
            "--train-manifest",
            str(manifest),
            "--validation-manifest",
            str(PROGRAM / "split" / "val-usable.csv"),
            "--cache-index",
            str(run / "cache-index.csv"),
            "--cache-root",
            str(PROGRAM / "cache"),
            "--dataset",
            "FakeAVCeleb+FF++" if arm.endswith("ffpp") else "FakeAVCeleb",
            "--checkpoint",
            str(checkpoint),
            "--history",
            str(history),
            "--run-id",
            name,
            "--split-hash",
            run.name,
            "--preprocessing-hash",
            preprocessing_hash,
            "--device",
            "cuda",
            "--epochs",
            str(epochs),
            "--learning-rate",
            "0.0001",
            "--weight-decay",
            "0.0001",
            "--patience",
            "3",
            "--batch-size",
            "8",
            "--accumulation-steps",
            "2",
            "--freeze-epochs",
            "1",
            "--workers",
            workers,
            "--seed",
            str(seed),
            *ARMS[arm],
            log=run / "logs" / f"train-{name}.log",
        )
        if code == 0:
            payload = json.loads(history.read_text(encoding="utf-8"))
            if not payload.get("epochs") or not payload.get("best_epoch"):
                raise ValueError(
                    f"Training did not complete a selectable epoch: {history}"
                )
            if payload.get("checkpoint_hash") != sha256(checkpoint):
                raise ValueError(
                    f"Training history checkpoint digest mismatch: {history}"
                )
            record_completion(receipt, configuration, [checkpoint, history])
            return True
        # A worker crash is host memory, not modelling; one process always fits.
        print(f"{name}: failed with {workers} workers (exit {code})", flush=True)
    return False


def evaluate(run: Path, arm: str, seed: int, preprocessing_hash: str) -> bool:
    name = f"{arm}-seed{seed}"
    for label, manifest, dataset, scope in EVALUATIONS:
        output = run / "evaluation" / f"{name}-{label}-metrics.json"
        predictions = run / "evaluation" / f"{name}-{label}-predictions.csv"
        receipt = output.with_name(output.stem + "-complete.json")
        configuration = {
            "checkpoint_sha256": sha256(run / "checkpoints" / f"{name}.pt"),
            "cache_payloads": cache_payload_digests(
                [manifest, run / "cache-index.csv"]
            ),
            "manifest_sha256": sha256(manifest),
            "dataset": dataset,
            "scope": scope,
            "preprocessing_hash": preprocessing_hash,
        }
        if completed(receipt, configuration, [output, predictions]):
            continue
        code = _cli(
            "evaluate",
            "branch",
            "--branch",
            "visual",
            "--checkpoint",
            str(run / "checkpoints" / f"{name}.pt"),
            "--manifest",
            str(manifest),
            "--cache-index",
            str(run / "cache-index.csv"),
            "--cache-root",
            str(PROGRAM / "cache"),
            "--dataset",
            dataset,
            "--threshold",
            "0.5",
            "--output",
            str(output),
            "--predictions",
            str(predictions),
            "--evidence-scope",
            scope,
            "--preprocessing-hash",
            preprocessing_hash,
            log=run / "logs" / f"evaluate-{name}.log",
        )
        print(f"{name} {label}: exit {code}", flush=True)
        if code:
            return False
        json.loads(output.read_text(encoding="utf-8"))
        record_completion(receipt, configuration, [output, predictions])
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/retrain-20261022"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[17])
    parser.add_argument("--arms", nargs="+", choices=sorted(ARMS), default=list(ARMS))
    parser.add_argument("--epochs", type=int, default=8)
    arguments = parser.parse_args(argv)

    run = arguments.run_dir
    preprocessing_hash = prepare(run)
    failed = False
    for seed in arguments.seeds:
        for arm in arguments.arms:
            if train(run, arm, seed, preprocessing_hash, arguments.epochs):
                failed |= not evaluate(run, arm, seed, preprocessing_hash)
            else:
                failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
