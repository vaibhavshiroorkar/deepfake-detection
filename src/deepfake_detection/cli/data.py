"""Manifests, splits and the view cache: `ddf manifest`, `ddf split`, `ddf cache`."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from deepfake_detection.cli.common import _write_json, add_preprocessing_arguments
from deepfake_detection.data.cache_build import build_cache
from deepfake_detection.data.cache_index import read_cache_index, write_cache_index
from deepfake_detection.data.guards import (
    reject_evaluation_only,
)
from deepfake_detection.data.manifest import load_manifest, write_manifest
from deepfake_detection.data.protocols import (
    audit_split,
    build_method_holdout_protocol,
    build_source_split,
    identity_strict_subset,
    split_hash,
    stratified_subsample,
)
from deepfake_detection.digest import file_sha256
from deepfake_detection.training.crossfit import build_group_folds
from deepfake_detection.views.cache_store import CacheStore


def _manifest_build(arguments: argparse.Namespace) -> int:
    result = load_manifest(arguments.input, dataset=arguments.dataset)
    write_manifest(result.records, arguments.output)
    _write_json(
        arguments.audit,
        {
            "records": len(result.records),
            "quarantined_paths": [str(path) for path in result.quarantined_paths],
        },
    )
    return 0


def _manifest_from_meta(arguments: argparse.Namespace) -> int:
    """Build a manifest straight from a FakeAVCeleb meta_data.csv.

    `manifest build` validates a manifest that already exists. Producing the
    first one from a raw drop used to be an unrecorded manual step, which is why
    no script for the original 2,000-row manifest survives in the repo. The
    conversion itself is data/meta.py, already used by the dashboard; this only
    puts a command in front of it and runs the result through the same
    quarantine pass every other manifest gets.
    """
    import pandas as pd

    from deepfake_detection.data.meta import manifest_from_meta

    meta = pd.read_csv(arguments.meta)
    frame = manifest_from_meta(
        meta,
        root=arguments.root,
        data_dir=arguments.data_dir,
        require_exists=not arguments.allow_missing,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(arguments.output, index=False)

    result = load_manifest(arguments.output, dataset=arguments.dataset)
    write_manifest(result.records, arguments.output)
    _write_json(
        arguments.audit,
        {
            "meta_rows": len(meta),
            "records": len(result.records),
            "dropped_missing_media": len(meta) - len(frame),
            "quarantined_paths": [str(path) for path in result.quarantined_paths],
        },
    )
    return 0


# What each branch or stream must find in a cache entry to be usable. The
# audiovisual streams need both halves of a pair: a clip with mouth crops but no
# audio track, which is every MNW clip, can feed the sync branch's video encoder
# but has nothing for a cross-modal comparison to read.
BRANCH_VIEWS = {
    "visual": ("visual_view",),
    "audio": ("audio_view",),
    "sync": ("sync_video_view",),
    "lipsync": ("sync_video_view", "sync_audio_view"),
    "emotion": ("visual_view", "audio_view"),
}


def _partition_by_view(
    records,
    cache_index,
    cache_store,
    *,
    branch: str,
    preprocessing_hash: str | None,
):
    """Split records into those a branch can score and those it must abstain on.

    A clip is unusable when it was never cached, when its cache entry was built
    by different preprocessing, or when the view this branch reads is absent.
    The last case is the pipeline's abstention policy rather than a fault: a
    clip whose primary face track was unstable gets no visual view, because
    substituting a full-frame crop would bias the evaluation.

    Returns (usable, abstained) where abstained maps clip_id to a reason.
    """
    required = BRANCH_VIEWS[branch]
    usable = []
    abstained: dict[str, str] = {}
    for record in records:
        path = cache_index.get(record.clip_id)
        if path is None:
            abstained[record.clip_id] = "not_cached"
            continue
        try:
            views = cache_store.available_views(path)
            metadata = cache_store.load_metadata(path)
        except (OSError, ValueError, KeyError) as error:
            abstained[record.clip_id] = f"unreadable_cache: {error}"
            continue
        if (
            preprocessing_hash is not None
            and metadata.get("preprocessing_config_hash") != preprocessing_hash
        ):
            abstained[record.clip_id] = "preprocessing_hash_mismatch"
            continue
        missing = [name for name in required if name not in views]
        if missing:
            abstained[record.clip_id] = "no_" + "_and_".join(missing)
            continue
        usable.append(record)
    return usable, abstained


def _manifest_usable(arguments: argparse.Namespace) -> int:
    """Write the subset of a manifest that one branch can actually read.

    Training and evaluation both crash on a clip whose view is missing, so this
    filter used to live in an untracked one-off script pinned to a single run
    directory, a single hash and a single branch. Losing it is how a rebuilt
    cache turns into a mid-epoch exception.

    The dropped clips are not silently discarded: the audit records each one and
    its reason, so the abstention rate stays reportable rather than vanishing
    from the denominator.
    """
    records = load_manifest(arguments.manifest, dataset=arguments.dataset).records
    cache_index = read_cache_index(arguments.cache_index)
    usable, abstained = _partition_by_view(
        records,
        cache_index,
        CacheStore(arguments.cache_root),
        branch=arguments.branch,
        preprocessing_hash=arguments.preprocessing_hash,
    )
    write_manifest(usable, arguments.output)

    reasons: dict[str, int] = {}
    for reason in abstained.values():
        key = reason.split(":")[0]
        reasons[key] = reasons.get(key, 0) + 1
    labels = {record.video_fake for record in usable}
    _write_json(
        arguments.audit,
        {
            "branch": arguments.branch,
            "dataset": arguments.dataset,
            "input_rows": len(records),
            "usable_rows": len(usable),
            "abstained_rows": len(abstained),
            "abstention_rate": (len(abstained) / len(records) if records else 0.0),
            "abstained_by_reason": dict(sorted(reasons.items())),
            "abstained_clip_ids": sorted(abstained),
            "both_classes_present": labels == {True, False},
        },
    )
    if not usable:
        raise ValueError(
            f"No clip in {arguments.manifest} has a {arguments.branch} view"
        )
    return 0


def _split_build(arguments: argparse.Namespace) -> int:
    reject_evaluation_only(arguments.dataset, operation="split building")
    result = load_manifest(arguments.manifest, dataset=arguments.dataset)
    split = build_source_split(result.records, seed=arguments.seed)
    strict = identity_strict_subset(split)
    audit = audit_split(split)
    for name, records in split.items():
        write_manifest(records, arguments.output_dir / f"{name}.csv")
        write_manifest(
            strict[name], arguments.output_dir / f"{name}-identity-strict.csv"
        )
    _write_json(
        arguments.output_dir / "audit.json",
        {
            "seed": arguments.seed,
            "split_hash": split_hash(split),
            "rows": {name: len(records) for name, records in split.items()},
            "strict_rows": {name: len(records) for name, records in strict.items()},
            "source_overlaps": {
                f"{left}:{right}": sorted(values)
                for (left, right), values in audit.source_overlaps.items()
            },
            "all_identity_overlaps": {
                f"{left}:{right}": sorted(values)
                for (left, right), values in audit.all_identity_overlaps.items()
            },
            "method_counts": audit.method_counts,
        },
    )
    return 0


def _split_crossfit(arguments: argparse.Namespace) -> int:
    reject_evaluation_only(arguments.dataset, operation="cross-fitting")
    records = load_manifest(arguments.manifest, dataset=arguments.dataset).records
    fraction = getattr(arguments, "validation_fraction", 0.15)
    folds = build_group_folds(
        records,
        folds=arguments.folds,
        seed=arguments.seed,
        validation_fraction=fraction,
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, fold in enumerate(folds):
        train = tuple(records[row] for row in fold.train_indices)
        holdout = tuple(records[row] for row in fold.holdout_indices)
        validation = tuple(records[row] for row in fold.validation_indices)
        write_manifest(train, arguments.output_dir / f"fold-{index}-train.csv")
        write_manifest(holdout, arguments.output_dir / f"fold-{index}-holdout.csv")
        write_manifest(
            validation, arguments.output_dir / f"fold-{index}-validation.csv"
        )
        summaries.append(
            {
                "fold": index,
                "train_sources": len(fold.train_sources),
                "holdout_sources": len(fold.holdout_sources),
                "train_rows": len(train),
                "holdout_rows": len(holdout),
                "validation_sources": len(fold.validation_sources),
                "validation_rows": len(validation),
                "protocol_hash": split_hash(
                    {
                        "train": train,
                        "val": validation,
                        "test": holdout,
                    }
                ),
            }
        )
    _write_json(
        arguments.output_dir / "crossfit-audit.json",
        {
            "protocol": "nested-source-v2",
            "seed": arguments.seed,
            "manifest_sha256": file_sha256(arguments.manifest),
            "validation_fraction": fraction,
            "folds": summaries,
        },
    )
    return 0


def _split_method_holdout(arguments: argparse.Namespace) -> int:
    split = {
        name: load_manifest(
            arguments.split_dir / f"{name}.csv",
            dataset=arguments.dataset,
        ).records
        for name in ("train", "val", "test")
    }
    protocol = build_method_holdout_protocol(
        split,
        heldout_methods=set(arguments.methods),
    )
    for name, records in protocol.items():
        write_manifest(records, arguments.output_dir / f"{name}.csv")
    _write_json(
        arguments.output_dir / "audit.json",
        {
            "heldout_methods": sorted(arguments.methods),
            "rows": {name: len(records) for name, records in protocol.items()},
        },
    )
    return 0


def _split_subsample(arguments: argparse.Namespace) -> int:
    """Thin an existing frozen split without rebuilding or rehashing it.

    The pilot tier and the full tier have to stay comparable, so they share one
    split and one split hash. Only the number of rows fed to training changes.
    """
    rows = {}
    for name in ("train", "val", "test"):
        records = load_manifest(
            arguments.split_dir / f"{name}.csv", dataset=arguments.dataset
        ).records
        sampled = (
            stratified_subsample(
                records, seed=arguments.seed, fake_ratio=arguments.fake_ratio
            )
            if name == "train"
            else records
        )
        write_manifest(sampled, arguments.output_dir / f"{name}.csv")
        rows[name] = {
            "rows": len(sampled),
            "real": sum(1 for record in sampled if not record.video_fake),
            "fake": sum(1 for record in sampled if record.video_fake),
            "methods": len({record.method for record in sampled}),
        }
    _write_json(
        arguments.output_dir / "audit.json",
        {
            "seed": arguments.seed,
            "fake_ratio": arguments.fake_ratio,
            "source_split_dir": str(arguments.split_dir),
            "rows": rows,
        },
    )
    return 0


def _cache_build(arguments: argparse.Namespace) -> int:
    from deepfake_detection.inference.loading import build_preprocessor

    result = load_manifest(arguments.manifest, dataset=arguments.dataset)
    preprocessor = build_preprocessor(
        code_version=arguments.code_version,
        device=arguments.device,
        detector=arguments.detector,
        tracker=arguments.tracker,
        crop_mode=arguments.crop_mode,
        model_path=arguments.model_path,
        expected_model_hash=arguments.expected_model_hash,
        remove_leading_silence=not arguments.keep_leading_silence,
    )
    report = build_cache(
        records=result.records,
        dataset_root=arguments.dataset_root,
        preprocessor=preprocessor,
        cache_store=CacheStore(arguments.cache_root),
        skip_cached=arguments.skip_cached,
        shard=_parse_shard(arguments.shard),
    )
    write_cache_index(arguments.index, report.cache_index)
    _write_json(
        arguments.audit,
        {
            "succeeded": report.succeeded,
            "failed": report.failed,
            "skipped": report.skipped,
            "full_fusion_ready": report.full_fusion_ready,
            "blocker_counts": report.blocker_counts,
            "preprocessing_hash": report.preprocessing_hash,
            "failures": report.failures,
        },
    )
    return 2 if report.failed else 0


def _parse_shard(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    index, _, count = value.partition("/")
    try:
        shard = (int(index), int(count))
    except ValueError as error:
        raise ValueError(f"Shard must look like I/N, got {value!r}") from error
    return shard


def _cache_merge(arguments: argparse.Namespace) -> int:
    """Fold per-shard index and audit files into one of each.

    Sharding is what makes a 21,544-clip build finish in hours rather than a day
    and a half, but each worker can only write its own slice. Training reads one
    index, so the shards have to be reassembled before it can run.
    """
    merged_index: dict[str, Path] = {}
    for path in arguments.indexes:
        for clip_id, cache_path in read_cache_index(path).items():
            merged_index[clip_id] = cache_path

    succeeded = failed = skipped = full_fusion_ready = 0
    blocker_counts: Counter[str] = Counter()
    failures: dict[str, str] = {}
    hashes: set[str] = set()
    for path in arguments.audits:
        audit = json.loads(path.read_text(encoding="utf-8"))
        succeeded += int(audit["succeeded"])
        failed += int(audit["failed"])
        skipped += int(audit.get("skipped", 0))
        full_fusion_ready += int(audit["full_fusion_ready"])
        blocker_counts.update(audit.get("blocker_counts", {}))
        failures.update(audit.get("failures", {}))
        if audit.get("preprocessing_hash"):
            hashes.add(audit["preprocessing_hash"])
    if len(hashes) > 1:
        raise ValueError(f"Shards disagree on the preprocessing hash: {sorted(hashes)}")

    write_cache_index(arguments.index, merged_index)
    _write_json(
        arguments.audit,
        {
            "succeeded": succeeded,
            "failed": failed,
            "skipped": skipped,
            "full_fusion_ready": full_fusion_ready,
            "blocker_counts": dict(sorted(blocker_counts.items())),
            "preprocessing_hash": next(iter(hashes)) if hashes else None,
            "failures": failures,
            "shards": len(arguments.audits),
        },
    )
    return 2 if failed else 0


def register_manifest(commands) -> None:
    manifest = commands.add_parser("manifest")
    manifest_commands = manifest.add_subparsers(dest="manifest_command", required=True)
    manifest_build = manifest_commands.add_parser("build")
    manifest_build.add_argument("--input", type=Path, required=True)
    manifest_build.add_argument("--output", type=Path, required=True)
    manifest_build.add_argument("--audit", type=Path, required=True)
    manifest_build.add_argument("--dataset", required=True)
    manifest_usable = manifest_commands.add_parser("usable")
    manifest_usable.add_argument("--manifest", type=Path, required=True)
    manifest_usable.add_argument("--cache-index", type=Path, required=True)
    manifest_usable.add_argument("--cache-root", type=Path, required=True)
    manifest_usable.add_argument("--output", type=Path, required=True)
    manifest_usable.add_argument("--audit", type=Path, required=True)
    manifest_usable.add_argument("--dataset", required=True)
    manifest_usable.add_argument(
        "--branch",
        choices=("visual", "audio", "sync", "lipsync", "emotion"),
        default="visual",
    )
    manifest_usable.add_argument("--preprocessing-hash")
    manifest_usable.set_defaults(handler=_manifest_usable)
    manifest_meta = manifest_commands.add_parser("from-meta")
    manifest_meta.add_argument("--meta", type=Path, required=True)
    manifest_meta.add_argument("--root", type=Path, required=True)
    manifest_meta.add_argument("--data-dir", type=Path, default=Path("data"))
    manifest_meta.add_argument("--output", type=Path, required=True)
    manifest_meta.add_argument("--audit", type=Path, required=True)
    manifest_meta.add_argument("--dataset", required=True)
    manifest_meta.add_argument("--allow-missing", action="store_true")
    manifest_meta.set_defaults(handler=_manifest_from_meta)
    manifest_build.set_defaults(handler=_manifest_build)


def register_split(commands) -> None:
    split = commands.add_parser("split")
    split_commands = split.add_subparsers(dest="split_command", required=True)
    split_build = split_commands.add_parser("build")
    split_build.add_argument("--manifest", type=Path, required=True)
    split_build.add_argument("--output-dir", type=Path, required=True)
    split_build.add_argument("--dataset", required=True)
    split_build.add_argument("--seed", type=int, required=True)
    split_build.set_defaults(handler=_split_build)
    split_crossfit = split_commands.add_parser("crossfit")
    split_crossfit.add_argument("--manifest", type=Path, required=True)
    split_crossfit.add_argument("--output-dir", type=Path, required=True)
    split_crossfit.add_argument("--dataset", required=True)
    split_crossfit.add_argument("--folds", type=int, default=3)
    split_crossfit.add_argument("--seed", type=int, required=True)
    split_crossfit.add_argument("--validation-fraction", type=float, default=0.15)
    split_crossfit.set_defaults(handler=_split_crossfit)
    split_holdout = split_commands.add_parser("method-holdout")
    split_holdout.add_argument("--split-dir", type=Path, required=True)
    split_holdout.add_argument("--output-dir", type=Path, required=True)
    split_holdout.add_argument("--dataset", required=True)
    split_holdout.add_argument("--methods", nargs="+", required=True)
    split_subsample = split_commands.add_parser("subsample")
    split_subsample.add_argument("--split-dir", type=Path, required=True)
    split_subsample.add_argument("--output-dir", type=Path, required=True)
    split_subsample.add_argument("--dataset", required=True)
    split_subsample.add_argument("--seed", type=int, required=True)
    split_subsample.add_argument("--fake-ratio", type=float, default=3.0)
    split_subsample.set_defaults(handler=_split_subsample)
    split_holdout.set_defaults(handler=_split_method_holdout)


def register_cache(commands) -> None:
    cache = commands.add_parser("cache")
    cache_commands = cache.add_subparsers(dest="cache_command", required=True)
    cache_build = cache_commands.add_parser("build")
    cache_build.add_argument("--manifest", type=Path, required=True)
    cache_build.add_argument("--dataset-root", type=Path, required=True)
    cache_build.add_argument("--cache-root", type=Path, required=True)
    cache_build.add_argument("--index", type=Path, required=True)
    cache_build.add_argument("--audit", type=Path, required=True)
    cache_build.add_argument("--dataset", required=True)
    cache_build.add_argument("--device", default="cpu")
    cache_build.add_argument("--code-version", required=True)
    cache_build.add_argument("--keep-leading-silence", action="store_true")
    cache_build.add_argument(
        "--skip-cached",
        action="store_true",
        help="Reuse clips already present in the cache root instead of rebuilding them.",
    )
    cache_build.add_argument(
        "--shard",
        help="Process only this shard, written I/N. Shards may run concurrently.",
    )
    cache_merge = cache_commands.add_parser("merge")
    cache_merge.add_argument("--indexes", type=Path, nargs="+", required=True)
    cache_merge.add_argument("--audits", type=Path, nargs="+", required=True)
    cache_merge.add_argument("--index", type=Path, required=True)
    cache_merge.add_argument("--audit", type=Path, required=True)
    cache_merge.set_defaults(handler=_cache_merge)
    add_preprocessing_arguments(cache_build)
    cache_build.set_defaults(handler=_cache_build)
