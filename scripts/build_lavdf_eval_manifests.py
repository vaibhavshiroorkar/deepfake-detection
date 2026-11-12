"""Two LAV-DF test manifests for scoring models trained on FakeAVCeleb.

The blind manifest is the protocol DFDC is scored under: one row per clip, no
annotation used, so every view is cut where it would be for an unlabelled
upload. `sync_start_sec` stays empty on purpose. An explicit start on the fake
rows only would place the sync window by label, which is the contamination the
old LAV-DF cache carried.

The matched manifest asks a narrower question of the sync branch alone: pointed
at the manipulated span, does it score that span above an untouched span of the
same file? Both rows carry an explicit start. The visual and audio views ignore
it, so only the sync branch is scored on this one.

    uv run python scripts/build_lavdf_eval_manifests.py --output-dir runs/lavdf-eval-20261031
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from deepfake_detection.data.lavdf import manifest_from_lavdf

FAKE_KINDS = ("FakeVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-FakeAudio")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/LAV-DF/LAV-DF"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-kind", type=int, default=750)
    parser.add_argument("--matched-per-kind", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    arguments = parser.parse_args(argv)

    rows = manifest_from_lavdf(arguments.root, arguments.data_dir, split="test")
    clips = rows[rows["method"] != "lavdf-genuine-span"].copy()
    blind = pd.concat(
        [
            group.sample(n=arguments.per_kind, random_state=arguments.seed)
            for _, group in clips.groupby("manipulation_type", sort=True)
        ],
        ignore_index=True,
    )
    blind["clip_id"] = blind["clip_id"].str.removesuffix("__forged")
    blind["sync_start_sec"] = None

    stems = blind["video_path"].map(lambda path: Path(path).stem)
    chosen = set()
    for kind in FAKE_KINDS:
        pool = stems[blind["manipulation_type"] == kind]
        chosen.update(pool.sample(n=arguments.matched_per_kind, random_state=arguments.seed))
    matched = rows[rows["method"].isin(("lavdf-forged", "lavdf-genuine-span"))]
    matched = matched[matched["video_path"].map(lambda path: Path(path).stem).isin(chosen)]
    # A forgery whose spans leave no clean two seconds has no genuine partner,
    # and an unpaired row would compare different files.
    paired = matched.groupby("video_path")["method"].transform("nunique") == 2
    matched = matched[paired]

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    blind.to_csv(arguments.output_dir / "blind.csv", index=False)
    matched.to_csv(arguments.output_dir / "matched.csv", index=False)
    print(f"blind {len(blind)} rows, {blind['manipulation_type'].value_counts().to_dict()}")
    print(f"matched {len(matched)} rows from {matched['video_path'].nunique()} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
