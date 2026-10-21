"""Score the served fusion head, gate-fusion.pt, on the stored stream embeddings.

The web app's audio-visual model is this exact head. Until now its card quoted
the deep-fusion ablation, a head fitted on the same streams without the
deployment presence patterns, so the number was a near relative rather than the
model served. This scores the served file itself, with every stream a video can
drive present, at the video threshold stored in the checkpoint.

    uv run python scripts/score_gate_fusion.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    import torch

    from deepfake_detection.evaluation.bootstrap import bootstrap_binary_metrics
    from deepfake_detection.evaluation.metrics import EvaluationItem
    from deepfake_detection.fusion.deep_loading import load_fusion
    from deepfake_detection.fusion.store import FeatureStore
    from deepfake_detection.training.fusion import as_tensors

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/design-b-20260910"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=17)
    arguments = parser.parse_args(argv)
    run = arguments.run_dir

    loaded = load_fusion(
        run / "checkpoints" / "gate-fusion.pt", device=arguments.device
    )
    threshold = loaded.threshold_for("video") or 0.5
    streams = tuple(sorted(loaded.stream_dims))

    for partition, dataset in (("in-domain", "FakeAVCeleb"), ("dfdc", "DFDC")):
        store = FeatureStore(run / "features" / f"{partition}.parquet")
        rows = [
            row
            for row in store.assemble(required_branches=streams, strict=False)
            if any(name in row.branch_embeddings for name in streams)
        ]
        values, presence, labels = as_tensors(
            rows, loaded.stream_dims, arguments.device
        )
        loaded.model.eval()
        with torch.inference_mode():
            probabilities = torch.sigmoid(loaded.model(values, presence).logit)
        items = [
            EvaluationItem(
                label=int(row.label),
                probability=float(p),
                source_identity=row.source_identity,
                method=row.method,
            )
            for row, p in zip(rows, probabilities.cpu().tolist(), strict=True)
        ]
        intervals = bootstrap_binary_metrics(
            items,
            threshold=threshold,
            samples=arguments.bootstrap_samples,
            seed=arguments.seed,
        )
        complete = sum(
            1 for row in rows if all(name in row.branch_embeddings for name in streams)
        )
        payload = {
            "branch": "gate-fusion",
            "checkpoint": str(run / "checkpoints" / "gate-fusion.pt"),
            "dataset": dataset,
            "streams": list(streams),
            "fixed_threshold": threshold,
            "rows": len(items),
            "rows_with_every_stream": complete,
            "class_balance": {
                "fake": sum(item.label for item in items),
                "real": sum(1 - item.label for item in items),
            },
            "metrics": {name: value.estimate for name, value in intervals.items()},
            "intervals": {name: asdict(value) for name, value in intervals.items()},
            "bootstrap_samples": arguments.bootstrap_samples,
            "bootstrap_unit": "source_identity",
        }
        out = run / "evaluation" / f"gate-fusion-{partition}-metrics.json"
        out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        predictions = out.with_name(f"gate-fusion-{partition}-predictions.csv")
        with predictions.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("clip_id", "label", "probability", "source", "method"))
            for row, item in zip(rows, items, strict=True):
                writer.writerow(
                    (
                        row.clip_id,
                        item.label,
                        item.probability,
                        row.source_identity,
                        row.method,
                    )
                )
        auc = intervals["roc_auc"]
        print(
            f"{dataset}: {len(items)} clips ({complete} with every stream)  "
            f"ROC-AUC {auc.estimate:.4f} [{auc.lower:.4f}, {auc.upper:.4f}]  "
            f"EER {intervals['eer'].estimate:.4f}  "
            f"BalAcc {intervals['balanced_accuracy'].estimate:.4f}  "
            f"ECE {intervals['expected_calibration_error'].estimate:.4f}"
        )
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
