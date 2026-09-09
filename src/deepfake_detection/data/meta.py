"""Turn a FakeAVCeleb meta_data.csv into manifest rows.

`manifest.py` owns the manifest contract that training and evaluation read.
This module is the step before it: the pandas view the dashboard uses to build
an in-memory manifest for a raw dataset drop that has never been audited, so a
freshly extracted download is browsable straight away.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .manifest import MANIPULATION_TYPES

# meta_data.csv has a trailing comma in its header, which produces an unnamed
# final column, and its ninth header "path" actually holds the filename. The
# columns are renamed positionally to reflect what they really contain.
META_COLUMNS = [
    "source",
    "target1",
    "target2",
    "method",
    "category",
    "manipulation_type",
    "race",
    "gender",
    "filename",
    "dirpath",
]

MANIFEST_COLUMNS = [
    "clip_id",
    "video_path",
    "label",
    "manipulation_type",
    "method",
    "source",
    "target1",
    "target2",
    "race",
    "gender",
]


def clip_label(clip_type: str) -> int:
    """Binary clip-level label: 1 is fake, 0 is real.

    A clip is real only if both its tracks are real, so RealVideo-FakeAudio
    counts as fake here even though every pixel is genuine. This is not the
    label a visual-only branch trains on: that branch cannot see the audio, so
    RealVideo-FakeAudio is real to it.
    """
    if clip_type not in MANIPULATION_TYPES:
        raise ValueError(
            f"Unrecognised FakeAVCeleb type {clip_type!r}. "
            f"Expected one of {sorted(MANIPULATION_TYPES)}."
        )
    return 0 if clip_type == "RealVideo-RealAudio" else 1


def resolve_video_path(root: Path, dirpath: str, filename: str) -> Path:
    """Absolute path of one meta_data.csv row's video.

    The meta dirpath looks like `FakeAVCeleb/<type>/<race>/<gender>/<id>`. The
    extracted tree drops the leading `FakeAVCeleb/` and lives under `root`, the
    directory holding meta_data.csv.
    """
    parts = Path(dirpath).parts
    if parts and parts[0] == "FakeAVCeleb":
        parts = parts[1:]
    return Path(root).joinpath(*parts, filename)


def manifest_from_meta(
    meta: pd.DataFrame,
    root: Path,
    data_dir: Path,
    require_exists: bool = True,
) -> pd.DataFrame:
    """Manifest rows for a FakeAVCeleb-style meta_data.csv.

    `root` is the directory holding meta_data.csv. `data_dir` is the repo's
    data/ directory, which video_path is written relative to, because the whole
    pipeline resolves a clip as data_dir / video_path. With require_exists, rows
    whose file is not on disk are dropped, so a partially downloaded drop still
    yields a usable manifest instead of an error.
    """
    meta = meta.copy()
    if len(meta.columns) != len(META_COLUMNS):
        raise ValueError(
            f"meta_data.csv has {len(meta.columns)} columns, expected "
            f"{len(META_COLUMNS)}: {META_COLUMNS}"
        )
    meta.columns = META_COLUMNS

    root, data_dir = Path(root), Path(data_dir)
    rows = []
    for record in meta.itertuples(index=False):
        video_path = resolve_video_path(root, record.dirpath, record.filename)
        if require_exists and not video_path.exists():
            continue
        rows.append(
            {
                "clip_id": (
                    f"{record.category}__{record.source}__{Path(record.filename).stem}"
                ),
                "video_path": str(video_path.relative_to(data_dir)),
                "label": clip_label(record.manipulation_type),
                "manipulation_type": record.manipulation_type,
                "method": record.method,
                "source": record.source,
                "target1": record.target1,
                "target2": record.target2,
                "race": record.race,
                "gender": record.gender,
            }
        )

    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    # meta_data.csv lists about 22 physical files twice with conflicting
    # `method` labels, for example wav2lip and faceswap-wav2lip for the same
    # .mp4. Dedupe on the file path so each file appears once.
    return frame.drop_duplicates(subset="video_path", keep="first").reset_index(
        drop=True
    )
