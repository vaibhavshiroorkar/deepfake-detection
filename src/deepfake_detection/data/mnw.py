"""Turn the Microsoft-Northwestern-WITNESS benchmark into manifest rows.

MNW is the locked external evaluation target. It is evaluation-only by license
and by this project's protocol: it cannot train a model, pick a threshold, or
select between candidates. `guards.py` enforces that; this module only reads it.

Three properties of the video half decide how it can be used, all verified
against the published tree rather than assumed:

  - `Deepfake_Video/` holds 120 clips, twelve generators of ten, and every file
    is named `*_no_audio_*`. There is no audio track, so only the visual branch
    can be scored on them. The audio and sync branches have nothing to read.
  - It contains no real videos. A fake-only set yields a detection rate
    (recall), never an ROC-AUC, because there are no negatives to rank against.
  - Nine of the twelve generators are absent from FakeAVCeleb, which is what
    makes this an unseen-generator test rather than a second in-domain one.

`AI_media_in_the_wild/Video/` is the opposite: both classes, expert-labelled,
and tiny (roughly a dozen clips). It is a qualitative case study, not a
benchmark, and `inconsistent` is skipped because it means the experts could not
decide, which is not a ground truth.

Schema caveat: `ClipRecord.manipulation_type` is a closed four-value enum with
no way to say "audio absent". MNW fakes are therefore recorded as
`FakeVideo-RealAudio`, which literally asserts genuine audio where there is
none. The truth is carried by `QualityReport.audio_present`, which the cache
records as False and which raises the `missing_audio` blocker.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

DATASET = "MNW"

LAB_DIRECTORY = "Deepfake_Video"
WILD_DIRECTORY = "AI_media_in_the_wild/Video"
WILD_FAKE = "likely_manipulated"
WILD_REAL = "likely_authentic"
# Expert analysis could not reach a conclusion on these, so they have no label.
WILD_SKIP = "inconsistent"

VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".webm", ".avi")

MANIFEST_COLUMNS = [
    "clip_id",
    "dataset",
    "video_path",
    "manipulation_type",
    "method",
    "source",
    "target1",
    "target2",
    "race",
    "gender",
]


def _videos(folder: Path) -> Iterator[Path]:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            yield path


def _row(
    video: Path,
    data_dir: Path,
    *,
    clip_id: str,
    method: str,
    fake: bool,
) -> dict[str, str]:
    return {
        "clip_id": clip_id,
        "dataset": DATASET,
        "video_path": str(video.relative_to(data_dir)),
        # See the module docstring: the enum cannot express "no audio track".
        "manipulation_type": (
            "FakeVideo-RealAudio" if fake else "RealVideo-RealAudio"
        ),
        "method": method,
        # Every MNW clip is its own identity. The benchmark publishes no
        # identity metadata, and assuming shared identities would fabricate a
        # grouping that the source-disjointness audit would then trust.
        "source": f"mnw-{clip_id}",
        "target1": "",
        "target2": "",
        "race": "unknown",
        "gender": "unknown",
    }


def manifest_from_mnw(
    root: Path,
    data_dir: Path,
    *,
    include_wild: bool = True,
) -> pd.DataFrame:
    """Manifest rows for an MNW checkout.

    `root` is the repository root holding `Deepfake_Video/`. `data_dir` is the
    repo's data/ directory, which video_path is written relative to.
    """
    root, data_dir = Path(root), Path(data_dir)
    rows: list[dict[str, str]] = []

    lab = root / LAB_DIRECTORY
    if lab.is_dir():
        for generator in sorted(path for path in lab.iterdir() if path.is_dir()):
            for video in _videos(generator):
                rows.append(
                    _row(
                        video,
                        data_dir,
                        clip_id=f"mnw__{generator.name}__{video.stem}",
                        method=f"mnw-{generator.name.lower()}",
                        fake=True,
                    )
                )

    wild = root / WILD_DIRECTORY
    if include_wild and wild.is_dir():
        for label_dir, fake in ((WILD_FAKE, True), (WILD_REAL, False)):
            folder = wild / label_dir
            if not folder.is_dir():
                continue
            for video in _videos(folder):
                rows.append(
                    _row(
                        video,
                        data_dir,
                        clip_id=f"mnw__wild__{label_dir}__{video.stem}",
                        method=f"mnw-wild-{label_dir.replace('_', '-')}",
                        fake=fake,
                    )
                )

    if not rows:
        raise ValueError(f"No MNW videos found under {root}")
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
