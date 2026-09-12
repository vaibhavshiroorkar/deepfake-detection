"""Fit the fusion head the dashboard serves, with a threshold per media kind.

`ddf train fusion --model deep` already fits a `StreamFusion`, and this is not a
replacement for it. It adds the two things a served head needs and an
experimental one does not.

First, it trains under the deployment presence patterns rather than with every
stream always present. The gate takes images and sound files, and a head that
only ever saw five-stream inputs has no calibrated behaviour when two arrive.

Second, it chooses a decision threshold per media kind. A video verdict and an
image verdict come from different evidence through the same head, so one
threshold cannot be right for both: an image drives the visual stream alone,
whose logits sit on a different scale than the fused five. Thresholds are chosen
on rows the head was not fitted on, by Youden's J, and stored in the checkpoint
so the server cannot pair a head with someone else's cut-off.

    uv run python scripts/fit_gate_fusion.py --run-dir runs/design-b-20260910
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _kind_streams(streams) -> dict[str, tuple[str, ...]]:
    """Which trained streams each media kind can drive, by the routing table."""
    from deepfake_detection.dashboard.lib import media_kind
    from deepfake_detection.inference.multimodal import _modality

    table = {}
    for kind in (media_kind.VIDEO, media_kind.IMAGE, media_kind.AUDIO):
        allowed = set(media_kind.runnable(kind))
        table[kind] = tuple(n for n in streams if _modality(n) in allowed)
    return table


def _probabilities(model, rows, dims, keep: tuple[str, ...], device: str):
    """Fused probabilities with every stream outside `keep` marked absent.

    Masking rather than rebuilding: this is exactly what serving an image does,
    so the threshold is chosen on the same computation it will govern.
    """
    import torch

    from deepfake_detection.training.fusion import as_tensors

    usable = [row for row in rows if any(s in row.branch_embeddings for s in keep)]
    if not usable:
        return [], []
    values, presence, labels = as_tensors(usable, dims, device)
    masked = {
        name: (flags if name in keep else torch.zeros_like(flags))
        for name, flags in presence.items()
    }
    model.eval()
    with torch.inference_mode():
        output = model(values, masked)
    return torch.sigmoid(output.logit).cpu().tolist(), labels.cpu().tolist()


def _youden_threshold(probabilities, labels) -> tuple[float, float]:
    """The cut-off maximising sensitivity plus specificity, and the J it reached.

    Ties go to the lower threshold, which is the conservative direction here: a
    detector that calls a manipulated clip real is the more expensive error.
    """
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return 0.5, 0.0
    best_cut, best_j = 0.5, -1.0
    for cut in sorted({round(p, 4) for p in probabilities} | {0.5}):
        true_positive = sum(1 for p, y in zip(probabilities, labels, strict=True)
                            if p >= cut and y == 1)
        false_positive = sum(1 for p, y in zip(probabilities, labels, strict=True)
                             if p >= cut and y == 0)
        j = true_positive / positives - false_positive / negatives
        if j > best_j:
            best_cut, best_j = float(cut), float(j)
    return best_cut, best_j


def main(argv: list[str] | None = None) -> int:
    import torch

    from deepfake_detection.fusion.store import FeatureStore
    from deepfake_detection.training.fusion import (
        DEPLOYMENT_PATTERNS,
        fit_stream_fusion,
        group_split,
        stream_dimensions,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--threshold-fraction",
        type=float,
        default=0.25,
        help="Share of the holdout rows kept back for choosing thresholds. Split "
        "by source identity, so a speaker is never on both sides.",
    )
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    store_path = arguments.run_dir / "features" / "holdout.parquet"
    if not store_path.is_file():
        print(f"No holdout store at {store_path}. Run scripts/score_streams.py first.")
        return 1

    store = FeatureStore(store_path)
    available = tuple(sorted({row.branch for row in store.read() if row.available}))
    # strict=False keeps a clip that only some streams could read. Dropping
    # those would fit the head on complete rows alone and then serve it partial
    # ones, which is the regime it most needs to have seen.
    rows = store.assemble(required_branches=available, strict=False)
    if not rows:
        print("The holdout store has no assembled rows.")
        return 1
    dims = stream_dimensions(rows)
    print(f"streams: {', '.join(sorted(dims))}")
    print(f"holdout rows: {len(rows):,}")

    fit_rows, threshold_rows = group_split(
        rows, fraction=arguments.threshold_fraction, seed=arguments.seed
    )
    print(f"fit on {len(fit_rows):,}, thresholds from {len(threshold_rows):,}\n")

    model, history = fit_stream_fusion(
        rows=fit_rows,
        epochs=arguments.epochs,
        seed=arguments.seed,
        device=arguments.device,
        presence_patterns=DEPLOYMENT_PATTERNS,
    )
    best = history.epochs[history.best_epoch - 1]
    print(
        f"best epoch {history.best_epoch} of {len(history.epochs)}  "
        f"val loss {best.validation_loss:.4f}  val AUC {best.validation_auc:.4f}\n"
    )

    thresholds: dict[str, float] = {}
    coverage: dict[str, int] = {}
    for kind, keep in _kind_streams(sorted(dims)).items():
        if not keep:
            print(f"{kind:6} no trained stream can run, no threshold")
            continue
        probabilities, labels = _probabilities(
            model, threshold_rows, dims, keep, arguments.device
        )
        if not probabilities:
            print(f"{kind:6} no row could be scored, no threshold")
            continue
        cut, j = _youden_threshold(probabilities, labels)
        thresholds[kind] = cut
        coverage[kind] = len(probabilities)
        print(
            f"{kind:6} threshold {cut:.3f}  Youden J {j:.3f}  "
            f"({len(probabilities):,} rows, streams: {', '.join(keep)})"
        )

    output = arguments.output or arguments.run_dir / "checkpoints" / "gate-fusion.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "stream_dims": history.stream_dims,
            "common_dim": 256,
            "hidden_sizes": [128],
            "split_hash": rows[0].split_hash,
            "preprocessing_hash": rows[0].preprocessing_hash,
            # Per-kind, and refusing to fall back to 0.5 silently: a kind with
            # no entry is one this head has no calibrated behaviour for, and the
            # server says so rather than guessing.
            "thresholds": thresholds,
            "threshold_rows": coverage,
        },
        output,
    )
    metadata = output.with_suffix(".json")
    metadata.write_text(
        json.dumps(
            {
                "streams": sorted(dims),
                "thresholds": thresholds,
                "threshold_rows": coverage,
                "fit_rows": len(fit_rows),
                "best_epoch": history.best_epoch,
                "validation_auc": best.validation_auc,
                "presence_patterns": DEPLOYMENT_PATTERNS,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {output} and {metadata}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
