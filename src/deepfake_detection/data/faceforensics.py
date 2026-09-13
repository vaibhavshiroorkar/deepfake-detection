"""Turn a FaceForensics++ c23 drop into manifest rows.

This is the corpus nearly every published cross-dataset table trains on, and
adding it is what makes any number in this project comparable to one. Until now
the project trained on FakeAVCeleb, which almost nobody uses as a training set,
so its cross-corpus figures could only be compared against themselves.

The drop is six manipulation directories of 1,000 clips each plus 1,000
originals. Three things about it differ from the other corpora here and each one
has bitten a pipeline somewhere:

  - FF++ manipulates video only. The audio track is the original in every fake,
    so every manipulated row is `FakeVideo-RealAudio` and the audio branch must
    never be trained on these labels.
  - `DeepFakeDetection` is the Google/Jigsaw actor set, filmed separately from
    the 1,000 YouTube originals. Its filenames use actor ids rather than
    original-video numbers, so its identities do not overlap the rest of the
    corpus at all.
  - A fake is named `<target>_<source>.mp4`, where the target supplies the video
    and the source supplies the face. This project's split protocol groups on
    the identity being replaced, which is the target's, so the target number is
    written to `source`. Getting this backwards puts a clip's two halves in
    different partitions and leaks identity across the split.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

DATASET = "FaceForensics++"

REAL_DIRECTORY = "real"
FAKE_DIRECTORY = "fake"

# The six manipulation families, named as the corpus names them. Kept as a
# tuple because leave-one-method-family-out is a required ablation and needs a
# stable ordering to hold one out by.
METHODS = (
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures",
    "DeepFakeDetection",
)

# 000.mp4 for an original, 000_003.mp4 for a manipulation, and the actor set
# uses names like 01_02__outside_talking__YVGY8LOK.mp4.
REAL_NAME = re.compile(r"^(\d+)$")
FAKE_NAME = re.compile(r"^(\d+)_(\d+)$")
ACTOR_NAME = re.compile(r"^(\d+)_(\d+)__")

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


def identity(stem: str) -> tuple[str, str]:
    """The (source, target) identity pair for one FF++ filename.

    Source is the identity being replaced, which is the first number in both
    naming conventions. The actor set's names carry two numbers before a double
    underscore and the same rule applies to them.
    """
    actor = ACTOR_NAME.match(stem)
    if actor is not None:
        return f"ffpp-{actor.group(1)}", f"ffpp-{actor.group(2)}"
    fake = FAKE_NAME.match(stem)
    if fake is not None:
        return f"ffpp-{fake.group(1)}", f"ffpp-{fake.group(2)}"
    real = REAL_NAME.match(stem)
    if real is not None:
        return f"ffpp-{real.group(1)}", ""
    # An original from the actor set, named without the pair.
    return f"ffpp-{stem}", ""


def _walk(root: Path) -> Iterator[tuple[str, Path]]:
    real = Path(root) / REAL_DIRECTORY
    if real.is_dir():
        for video in sorted(real.glob("*.mp4")):
            yield "real", video
    fake_root = Path(root) / FAKE_DIRECTORY
    for method in METHODS:
        folder = fake_root / method
        if not folder.is_dir():
            continue
        for video in sorted(folder.glob("*.mp4")):
            yield method, video


def manifest_from_ffpp(
    root: Path,
    data_dir: Path,
    *,
    methods: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Manifest rows for a FaceForensics++ c23 drop.

    `root` holds `real/` and `fake/<Method>/`. `video_path` is written relative
    to `data_dir`, matching how the rest of the pipeline resolves a clip.
    `methods` restricts the manipulation families, which is how a
    leave-one-family-out training manifest is produced.
    """
    root, data_dir = Path(root), Path(data_dir)
    wanted = set(methods) if methods is not None else None

    rows = []
    for method, video in _walk(root):
        fake = method != "real"
        if fake and wanted is not None and method not in wanted:
            continue
        source, target = identity(video.stem)
        rows.append(
            {
                "clip_id": f"ffpp__{method}__{video.stem}",
                "dataset": DATASET,
                "video_path": str(video.relative_to(data_dir)).replace("\\", "/"),
                # Video only. The audio track of an FF++ fake is the original.
                "manipulation_type": (
                    "FakeVideo-RealAudio" if fake else "RealVideo-RealAudio"
                ),
                "method": f"ffpp-{method.lower()}" if fake else "real",
                "source": source,
                "target1": target,
                "target2": "",
                # FF++ publishes no demographic metadata. build_source_split
                # stratifies on (race, gender), so one uniform stratum is the
                # honest encoding of "not supplied".
                "race": "unknown",
                "gender": "unknown",
            }
        )
    if not rows:
        raise ValueError(f"No FaceForensics++ videos found under {root}")
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
