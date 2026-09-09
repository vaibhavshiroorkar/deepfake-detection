"""Turn a Celeb-DF-v2 drop into manifest rows.

Celeb-DF-v2 is this project's cross-dataset generalization target: a model is
trained on FakeAVCeleb and scored here without ever seeing a Celeb-DF frame in
training. It ships as three video directories plus an official test list, with
no metadata CSV, so the manifest is derived from filenames.

Two conventions differ from ours and are easy to get wrong:

  - `List_of_testing_videos.txt` writes `1` for REAL and `0` for FAKE, which is
    the opposite of this project's `label`. The mapping is inverted here, once,
    rather than at every call site.
  - Celeb-DF manipulates video only. Its audio is the original track, so every
    fake row is `FakeVideo-RealAudio` and the audio branch sees genuine audio.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

DATASET = "Celeb-DF-v2"

REAL_DIRECTORIES = ("Celeb-real", "YouTube-real")
FAKE_DIRECTORY = "Celeb-synthesis"
TEST_LIST = "List_of_testing_videos.txt"

# Celeb-real/id0_0000.mp4 and Celeb-synthesis/id0_id1_0000.mp4. The first id is
# the identity whose face is being replaced, which is the source under this
# project's split protocol; the second is the identity supplying the face.
REAL_NAME = re.compile(r"^(id\d+)_(\d+)$")
FAKE_NAME = re.compile(r"^(id\d+)_(id\d+)_(\d+)$")

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


def _identity(directory: str, stem: str) -> tuple[str, str]:
    """The (source, target) identity pair for one filename.

    YouTube-real clips carry no identity at all, so each one becomes its own
    source. Collapsing them into a single shared source would put every one of
    them in the same split partition and waste them.
    """
    if directory == "YouTube-real":
        return f"youtube-{stem}", ""
    fake = FAKE_NAME.match(stem)
    if fake is not None:
        return f"celeb-{fake.group(1)}", f"celeb-{fake.group(2)}"
    real = REAL_NAME.match(stem)
    if real is not None:
        return f"celeb-{real.group(1)}", ""
    raise ValueError(f"Unrecognised Celeb-DF filename {stem!r} in {directory}")


def read_test_list(root: Path) -> set[str]:
    """The official 518-clip test split, as `directory/filename.mp4` strings.

    Cross-dataset numbers are only comparable with published work if they use
    this list rather than a split we invented.
    """
    path = Path(root) / TEST_LIST
    entries: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        entries.add(parts[1].replace("\\", "/"))
    return entries


def _walk(root: Path) -> Iterator[tuple[str, Path]]:
    for directory in (*REAL_DIRECTORIES, FAKE_DIRECTORY):
        folder = Path(root) / directory
        if not folder.is_dir():
            continue
        for video in sorted(folder.glob("*.mp4")):
            yield directory, video


def manifest_from_celebdf(
    root: Path,
    data_dir: Path,
    *,
    test_only: bool = False,
) -> pd.DataFrame:
    """Manifest rows for a Celeb-DF-v2 drop.

    `root` holds the three video directories. `data_dir` is the repo's data/
    directory, which video_path is written relative to, matching how the rest of
    the pipeline resolves a clip. With test_only, only the official test list is
    emitted, which is the 518 clips the cross-dataset evaluation uses.
    """
    root, data_dir = Path(root), Path(data_dir)
    wanted = read_test_list(root) if test_only else None

    rows = []
    for directory, video in _walk(root):
        key = f"{directory}/{video.name}"
        if wanted is not None and key not in wanted:
            continue
        source, target = _identity(directory, video.stem)
        fake = directory == FAKE_DIRECTORY
        rows.append(
            {
                "clip_id": f"celebdf__{directory}__{video.stem}",
                "dataset": DATASET,
                "video_path": str(video.relative_to(data_dir)),
                # Celeb-DF swaps faces and leaves the audio track alone.
                "manipulation_type": (
                    "FakeVideo-RealAudio" if fake else "RealVideo-RealAudio"
                ),
                "method": "celebdf-synthesis" if fake else "real",
                "source": source,
                "target1": target,
                "target2": "",
                # Celeb-DF publishes no demographic metadata. build_source_split
                # stratifies on (race, gender), so one uniform stratum is the
                # honest encoding of "not supplied".
                "race": "unknown",
                "gender": "unknown",
            }
        )
    if not rows:
        raise ValueError(f"No Celeb-DF videos found under {root}")
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
