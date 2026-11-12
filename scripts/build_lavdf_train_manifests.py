"""LAV-DF train and dev windows for the audiovisual detector.

Each sampled forgery gives two rows from the same file: a window centred on the
manipulated span, labelled fake, and the widest window that misses every span,
labelled real. Same speaker, microphone and codec on both sides, so a model can
only separate them by what happens inside the window. Genuine clips add real
speakers, which FakeAVCeleb has 500 of.

Every row carries an explicit window start, which the corrected preprocessor
honours, including zero.

    uv run python scripts/build_lavdf_train_manifests.py --output-dir runs/av-detector-20261031
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from deepfake_detection.data.lavdf import manifest_from_lavdf


def windows(rows: pd.DataFrame, *, per_kind: int, genuine: int, seed: int) -> pd.DataFrame:
    forged = rows[rows["method"] == "lavdf-forged"]
    paired_files = set(rows.loc[rows["method"] == "lavdf-genuine-span", "video_path"])
    forged = forged[forged["video_path"].isin(paired_files)]
    chosen = pd.concat(
        [
            group.sample(n=min(per_kind, len(group)), random_state=seed)
            for _, group in forged.groupby("manipulation_type", sort=True)
        ]
    )
    pairs = rows[
        rows["video_path"].isin(chosen["video_path"])
        & rows["method"].isin(("lavdf-forged", "lavdf-genuine-span"))
    ]
    real = rows[rows["method"] == "real"].sample(n=genuine, random_state=seed)
    # Shuffled so a partial cache holds every kind in proportion.
    return pd.concat([pairs, real]).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/LAV-DF/LAV-DF"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-kind", type=int, default=1_000)
    parser.add_argument("--train-genuine", type=int, default=1_500)
    parser.add_argument("--dev-per-kind", type=int, default=200)
    parser.add_argument("--dev-genuine", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    arguments = parser.parse_args(argv)

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    for split, per_kind, genuine in (
        ("train", arguments.train_per_kind, arguments.train_genuine),
        ("dev", arguments.dev_per_kind, arguments.dev_genuine),
    ):
        rows = manifest_from_lavdf(arguments.root, arguments.data_dir, split=split)
        sample = windows(rows, per_kind=per_kind, genuine=genuine, seed=arguments.seed)
        sample.to_csv(arguments.output_dir / f"lavdf-{split}.csv", index=False)
        counts = sample.groupby(["method", "manipulation_type"]).size().to_dict()
        print(f"{split}: {len(sample)} rows {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
