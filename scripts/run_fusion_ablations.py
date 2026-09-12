"""Two ablations the research design specifies and that were never run.

**Late against deep.** The Design A head calibrates each branch's logit and fits
a logistic regression over the calibrated scores plus three quality features.
The Design B head projects each stream's embedding and fuses in feature space.
The comparison is cheap because both read the same feature stores, and it
answers whether the embedding carries anything the logit does not.

The late head cannot score a clip that is missing a stream: it demands every
column, which is a property of the head, not a setting. That is reported as
coverage rather than hidden by scoring the two heads on different row sets.

**Abstention against silent fallback.** The deep head can score a clip with
streams missing, by masking them to zero. So it faces a choice the late head
does not: abstain on partial coverage, or answer from what is there. Both arms
are measured on the same partition, with coverage beside the AUC, because an
abstention policy that reports 0.62 on 40 percent of the clips and one that
reports 0.59 on all of them are not comparable on AUC alone.

    uv run python scripts/run_fusion_ablations.py --run-dir runs/design-b-20260910
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class _Scored:
    score: float
    label: int
    source_identity: str


def _auc_of(items) -> float:
    from sklearn.metrics import roc_auc_score

    labels = [item.label for item in items]
    if len(set(labels)) != 2:
        return 0.5
    return float(roc_auc_score(labels, [item.score for item in items]))


def _interval(scored, samples: int, seed: int):
    from deepfake_detection.evaluation.bootstrap import cluster_bootstrap_interval

    if len({item.label for item in scored}) != 2:
        return None
    return cluster_bootstrap_interval(scored, _auc_of, samples=samples, seed=seed)


def _rows(path: Path, streams: tuple[str, ...]):
    from deepfake_detection.fusion.store import FeatureStore

    return FeatureStore(path).assemble(required_branches=streams, strict=False)


def _complete(rows, streams: tuple[str, ...]):
    return [row for row in rows if all(s in row.branch_embeddings for s in streams)]


def _deep_scores(model, rows, dims, device: str):
    import torch

    from deepfake_detection.training.fusion import as_tensors

    usable = [row for row in rows if row.branch_embeddings]
    if not usable:
        return []
    values, presence, _ = as_tensors(usable, dims, device)
    model.eval()
    with torch.inference_mode():
        output = model(values, presence)
    return [
        _Scored(float(logit), int(row.label), row.source_identity)
        for logit, row in zip(output.logit.cpu().tolist(), usable, strict=True)
    ]


def _late_scores(artifact, rows, streams: tuple[str, ...]):
    from deepfake_detection.fusion.late import FusionSample

    usable = _complete(rows, streams)
    if not usable:
        return []
    samples = [
        FusionSample(
            branch_logits={s: row.branch_logits[s] for s in streams},
            face_coverage=row.face_coverage,
            audio_clipped=row.audio_clipped,
            av_duration_delta_sec=row.av_duration_delta_sec,
        )
        for row in usable
    ]
    probabilities = artifact.predict_proba(samples)
    return [
        _Scored(float(p), int(row.label), row.source_identity)
        for p, row in zip(probabilities, usable, strict=True)
    ]


def _line(name: str, scored, total: int, samples: int, seed: int) -> dict:
    interval = _interval(scored, samples, seed)
    coverage = len(scored) / total if total else 0.0
    if interval is None:
        print(f"  {name:34} n/a          coverage {coverage:6.1%}")
        return {"auc": None, "coverage": coverage, "scored": len(scored)}
    print(
        f"  {name:34} {interval.estimate:.4f} "
        f"[{interval.lower:.4f}, {interval.upper:.4f}]  coverage {coverage:6.1%}"
    )
    return {
        "auc": interval.estimate,
        "ci": [interval.lower, interval.upper],
        "coverage": coverage,
        "scored": len(scored),
    }


def main(argv: list[str] | None = None) -> int:
    from deepfake_detection.fusion.late import LateFusion
    from deepfake_detection.fusion.store import FeatureStore
    from deepfake_detection.training.fusion import (
        fit_stream_fusion,
        stream_dimensions,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    features = arguments.run_dir / "features"
    holdout_path = features / "holdout.parquet"
    if not holdout_path.is_file():
        print(f"No holdout store at {holdout_path}. Run scripts/score_streams.py.")
        return 1

    streams = tuple(
        sorted({row.branch for row in FeatureStore(holdout_path).read() if row.available})
    )
    print(f"streams: {', '.join(streams)}\n")

    holdout = _rows(holdout_path, streams)
    dims = stream_dimensions(holdout)

    deep, history = fit_stream_fusion(
        rows=holdout,
        epochs=arguments.epochs,
        seed=arguments.seed,
        device=arguments.device,
        presence_patterns={"video": 1.0},
    )
    print(
        f"deep head: best epoch {history.best_epoch} of {len(history.epochs)}, "
        f"{history.train_clips:,} clips"
    )

    complete_holdout = _complete(holdout, streams)
    late = LateFusion(branch_names=streams)
    from deepfake_detection.fusion.late import FusionSample

    late.fit(
        [
            FusionSample(
                branch_logits={s: row.branch_logits[s] for s in streams},
                face_coverage=row.face_coverage,
                audio_clipped=row.audio_clipped,
                av_duration_delta_sec=row.av_duration_delta_sec,
            )
            for row in complete_holdout
        ],
        [int(row.label) for row in complete_holdout],
    )
    print(
        f"late head: {len(complete_holdout):,} of {len(holdout):,} holdout clips "
        "had every stream\n"
    )

    results: dict[str, dict] = {}
    for partition in ("in-domain", "dfdc"):
        path = features / f"{partition}.parquet"
        if not path.is_file():
            continue
        rows = _rows(path, streams)
        total = len(rows)
        print(f"{partition}: {total:,} clips")
        results[partition] = {
            "clips": total,
            "late_fusion": _line(
                "late fusion, complete clips only",
                _late_scores(late, rows, streams),
                total,
                arguments.bootstrap_samples,
                arguments.seed,
            ),
            "deep_abstaining": _line(
                "deep fusion, abstains on partial",
                _deep_scores(deep, _complete(rows, streams), dims, arguments.device),
                total,
                arguments.bootstrap_samples,
                arguments.seed,
            ),
            "deep_masking": _line(
                "deep fusion, answers from what ran",
                _deep_scores(deep, rows, dims, arguments.device),
                total,
                arguments.bootstrap_samples,
                arguments.seed,
            ),
        }
        print()

    output = (
        arguments.output or arguments.run_dir / "evaluation" / "fusion-ablations.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"streams": list(streams), "partitions": results}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
