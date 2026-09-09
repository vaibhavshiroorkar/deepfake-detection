"""Turn a DFDC part into manifest rows, for cross-corpus evaluation.

DFDC is the only corpus available to this project that is both complete in
audio and independent of everything it trains on. FakeAVCeleb and LAV-DF are
both built on VoxCeleb2, so scoring one on the other changes the generator and
not the corpus. Celeb-DF-v2 and MNW carry no audio track, so neither can score
an audiovisual stream at all. DFDC was filmed for the challenge with paid
actors, and its forgeries include swapped audio.

Its metadata is thin next to LAV-DF's: a label, a split, and for each forgery
the real clip it came from. There are no manipulated-span timestamps, so no
matched pairs, and no separate video and audio flags. A DFDC "FAKE" may have
manipulated video, manipulated audio, or both, and the file does not say which.

That ambiguity is recorded rather than guessed at. Every forgery is written as
`FakeVideo-RealAudio`, which is the closest the four-value enum comes to
"manipulated, cue unknown", and the cue-specific `audio_fake` label it implies
is not trustworthy here. Use DFDC to score whole-clip detection, not to make a
claim about which modality a branch caught.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

DATASET = "DFDC"

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


def _metadata_files(root: Path) -> Iterator[Path]:
    """Every label file under the checkout, one per downloaded part.

    Matches `metadata*.json` rather than the exact name because each DFDC part
    ships its own `metadata.json`, and a checkout that flattens parts into one
    directory has to rename them apart. Globbing the exact name in that layout
    silently finds one part's labels and leaves the rest of the videos
    unlabelled, which reads as a much smaller dataset rather than as an error.
    """
    yield from sorted(Path(root).rglob("metadata*.json"))


def manifest_from_dfdc(root: Path, data_dir: Path) -> pd.DataFrame:
    """Manifest rows for a DFDC checkout.

    `root` holds the extracted videos and their `metadata.json`. `data_dir` is
    the repo's data/ directory, which video_path is written relative to.
    """
    root, data_dir = Path(root), Path(data_dir)
    videos = {path.name: path for path in root.rglob("*.mp4")}

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for metadata_path in _metadata_files(root):
        entries = json.loads(metadata_path.read_text(encoding="utf-8"))
        for name, entry in entries.items():
            video = videos.get(name)
            if video is None or name in seen:
                continue
            seen.add(name)
            fake = str(entry.get("label", "")).upper() == "FAKE"
            # A forgery groups with the clip it was made from, so the two never
            # land on opposite sides of a split.
            origin = entry.get("original") or name
            rows.append(
                {
                    "clip_id": f"dfdc__{Path(name).stem}",
                    "dataset": DATASET,
                    "video_path": str(video.relative_to(data_dir)),
                    "manipulation_type": (
                        "FakeVideo-RealAudio" if fake else "RealVideo-RealAudio"
                    ),
                    "method": "dfdc-swap" if fake else "real",
                    "source": f"dfdc-{Path(origin).stem}",
                    "target1": "",
                    "target2": "",
                    "race": "unknown",
                    "gender": "unknown",
                }
            )
    if not rows:
        raise ValueError(f"No DFDC videos with metadata found under {root}")
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
