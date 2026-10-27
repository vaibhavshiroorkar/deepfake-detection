"""A frozen encoder over face crops, with a linear probe on top.

F13 left this cell empty. A frozen DINOv3 probe over full frames sits at chance
on face manipulation, and a fine-tuned EfficientNet over face crops does not,
but those two rows change the region and the model at once. This holds the
region at the face crop the visual branch reads and swaps the fine-tuned CNN for
a frozen encoder, which is the setting the literature finds transfers best
across corpora (a frozen CLIP-family encoder with a light probe).

Two stages, so the probe can be refitted in seconds:

    extract  embed every cached crop once per partition, mean and spread over
             the 16 frames, written to one .npz per partition
    fit      logistic regression on the training partitions, C chosen on
             FakeAVCeleb validation, scored on every other partition with
             identity-clustered intervals, and on codec-swapped FakeAVCeleb
             test clips (F35)

DFDC and Celeb-DF are never used for fitting or for choosing C.

    uv run python scripts/face_crop_probe.py extract --encoder pe
    uv run python scripts/face_crop_probe.py fit --encoder pe --train fav ffpp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROGRAM = Path("runs/program-20260906")
FFPP = Path("runs/ffpp-20260913")
PARTITIONS = {
    "fav-train": (PROGRAM / "split" / "train-usable.csv", PROGRAM),
    "fav-val": (PROGRAM / "split" / "val-usable.csv", PROGRAM),
    "fav-test": (PROGRAM / "split" / "test-usable.csv", PROGRAM),
    "ffpp-train": (FFPP / "split" / "train-usable.csv", FFPP),
    "ffpp-test": (FFPP / "split" / "test-usable.csv", FFPP),
    "dfdc": (PROGRAM / "split" / "dfdc-visual.csv", PROGRAM),
    "celebdf": (Path("runs/full-20260904/celebdf-test-manifest.csv"), PROGRAM),
}
TRAIN_SETS = {"fav": "fav-train", "ffpp": "ffpp-train"}
SCORED = ("fav-test", "ffpp-test", "dfdc", "celebdf")


def _rows(manifest: Path) -> list[dict[str, str]]:
    import csv

    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def extract(arguments: argparse.Namespace) -> int:
    from deepfake_detection.data.cache_index import read_cache_index
    from deepfake_detection.inference.frozen_features import (
        embed_windows,
        load_encoder,
        normalisation,
    )
    from deepfake_detection.training.augment import CodecSwap, to_pixels
    from deepfake_detection.views.cache_store import CacheStore

    model, _ = load_encoder(arguments.encoder, device=arguments.device)
    store = CacheStore(Path("."))
    swap = CodecSwap()
    out_dir = arguments.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(name, False) for name in PARTITIONS] + [("fav-test", True)]
    for name, swapped in jobs:
        stem = f"{name}-codec-swap" if swapped else name
        target = out_dir / f"{stem}-{arguments.encoder}.npz"
        if target.is_file():
            print(f"{stem}: present", flush=True)
            continue
        manifest, run = PARTITIONS[name]
        index = read_cache_index(run / "cache-index.csv")
        vectors, labels, sources, clips = [], [], [], []
        for count, row in enumerate(_rows(manifest), start=1):
            path = index.get(row["clip_id"])
            if path is None:
                continue
            view = store.load(path, views=("visual_view",)).visual_view
            if view is None:
                continue
            label = float(row["video_fake"])
            if swapped:
                view, _ = swap(view, label)
            pixels = to_pixels(view)
            # Each frame is its own one-frame window, so embed_windows returns
            # the mean over frames and the spread across them.
            windows = pixels[:, None]
            vectors.append(
                embed_windows(model, windows, arguments.device, batch_size=32)
            )
            labels.append(int(label))
            sources.append(row["source"])
            clips.append(row["clip_id"])
            if count % 500 == 0:
                print(f"  {stem}: {count}", flush=True)
        np.savez_compressed(
            target,
            features=np.stack(vectors).astype(np.float32),
            labels=np.asarray(labels),
            sources=np.asarray(sources),
            clips=np.asarray(clips),
        )
        print(f"{stem}: {len(vectors)} clips -> {target}", flush=True)
    (out_dir / f"normalisation-{arguments.encoder}.json").write_text(
        json.dumps(normalisation(arguments.encoder), indent=2) + "\n", encoding="utf-8"
    )
    return 0


def _load(out_dir: Path, stem: str, encoder: str):
    data = np.load(out_dir / f"{stem}-{encoder}.npz")
    return data["features"], data["labels"], data["sources"]


def _interval(labels, scores, sources, samples: int, seed: int) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    from deepfake_detection.evaluation.bootstrap import cluster_bootstrap_interval
    from deepfake_detection.evaluation.metrics import EvaluationItem

    items = [
        EvaluationItem(
            label=int(y), probability=float(p), source_identity=str(s), method=""
        )
        for y, p, s in zip(labels, scores, sources, strict=True)
    ]

    def auc(chosen):
        ys = [item.label for item in chosen]
        if len(set(ys)) < 2:
            raise ValueError("one class")
        return float(roc_auc_score(ys, [item.probability for item in chosen]))

    interval = cluster_bootstrap_interval(items, auc, samples=samples, seed=seed)
    return {
        "roc_auc": interval.estimate,
        "lower": interval.lower,
        "upper": interval.upper,
        "clips": len(items),
    }


def fit(arguments: argparse.Namespace) -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out_dir, encoder = arguments.out_dir, arguments.encoder
    parts = [_load(out_dir, TRAIN_SETS[name], encoder) for name in arguments.train]
    features = np.concatenate([p[0] for p in parts])
    labels = np.concatenate([p[1] for p in parts])
    val_x, val_y, _ = _load(out_dir, "fav-val", encoder)
    swap_x, swap_y, swap_s = _load(out_dir, "fav-test-codec-swap", encoder)

    # C is chosen on FakeAVCeleb validation alone. Balanced weights, because
    # the training sets run about nine fakes to one genuine clip.
    best = None
    for c in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, max_iter=5000, class_weight="balanced"),
        ).fit(features, labels)
        score = roc_auc_score(val_y, probe.decision_function(val_x))
        print(f"C={c:<6} validation {score:.4f}", flush=True)
        if best is None or score > best[0] + 1e-4:
            best = (score, c, probe)
    _, chosen_c, probe = best

    report = {
        "encoder": encoder,
        "normalisation": json.loads(
            (out_dir / f"normalisation-{encoder}.json").read_text(encoding="utf-8")
        ),
        "train": arguments.train,
        "train_clips": int(len(labels)),
        "chosen_c": chosen_c,
        "validation_roc_auc": best[0],
        "partitions": {},
    }
    for stem in SCORED:
        x, y, s = _load(out_dir, stem, encoder)
        report["partitions"][stem] = _interval(
            y, probe.decision_function(x), s, arguments.samples, arguments.seed
        )
    report["partitions"]["fav-test-codec-swap"] = _interval(
        swap_y,
        probe.decision_function(swap_x),
        swap_s,
        arguments.samples,
        arguments.seed,
    )
    name = f"face-probe-{encoder}-{'-'.join(arguments.train)}.json"
    (out_dir / name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for stem, values in report["partitions"].items():
        print(
            f"{stem:22} {values['roc_auc']:.4f} "
            f"[{values['lower']:.4f}, {values['upper']:.4f}]  n={values['clips']}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("extract", extract), ("fit", fit)):
        command = commands.add_parser(name)
        command.add_argument("--encoder", default="pe")
        command.add_argument(
            "--out-dir", type=Path, default=Path("runs/face-probe-20261022")
        )
        command.set_defaults(handler=handler)
        if name == "extract":
            command.add_argument("--device", default="cuda")
        else:
            command.add_argument(
                "--train", nargs="+", choices=sorted(TRAIN_SETS), default=["fav"]
            )
            command.add_argument("--samples", type=int, default=1000)
            command.add_argument("--seed", type=int, default=17)
    arguments = parser.parse_args(argv)
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
