"""
Pure functions for turning FakeAVCeleb's meta_data.csv into our manifest.

Kept separate from audit_dataset.py (which does the file walking and CSV
writing) so this logic can be tested without touching the 21,544-clip dataset.
The dashboard also calls manifest_from_meta() to build an in-memory manifest for
a raw dataset drop that has not been audited yet (dashboard/lib/datasets.py).
"""
from pathlib import Path

import pandas as pd


# The four categories FakeAVCeleb ships, as they appear in meta_data.csv's
# `type` column. Anything outside this set is a bug, not a new category.
CLIP_TYPES = (
    "RealVideo-RealAudio",
    "RealVideo-FakeAudio",
    "FakeVideo-RealAudio",
    "FakeVideo-FakeAudio",
)


def clip_label(clip_type: str) -> int:
    """
    Binary CLIP-LEVEL label: 1 = fake, 0 = real.

    A clip is real only if both its tracks are real, so RealVideo-FakeAudio
    counts as fake here even though every pixel is genuine.

    Note this is NOT the label a visual-only stream trains on -- that stream
    cannot see the audio, so RealVideo-FakeAudio is "real" to it. See
    docs/PROJECT_OVERVIEW.md section 6.
    """
    if clip_type not in CLIP_TYPES:
        raise ValueError(
            f"Unrecognised FakeAVCeleb type {clip_type!r}. "
            f"Expected one of {CLIP_TYPES}."
        )
    return 0 if clip_type == "RealVideo-RealAudio" else 1


# meta_data.csv has a trailing comma in its header, producing an unnamed final
# column; and its 9th header "path" actually holds the filename. We rename
# positionally to reflect the real content.
META_COLUMNS = ["source", "target1", "target2", "method", "category",
                "manipulation_type", "race", "gender", "filename", "dirpath"]

MANIFEST_COLUMNS = ["clip_id", "video_path", "label", "manipulation_type",
                    "method", "source", "target1", "target2", "race", "gender"]


def resolve_video_path(root: Path, dirpath: str, filename: str) -> Path:
    """
    Absolute path of one meta_data.csv row's video.

    meta dirpath looks like 'FakeAVCeleb/<type>/<race>/<gender>/<id>'. The real
    extracted tree drops the leading 'FakeAVCeleb/' and lives under `root` (the
    directory holding meta_data.csv). Append the filename to get the actual file.
    """
    parts = Path(dirpath).parts
    if parts and parts[0] == "FakeAVCeleb":
        parts = parts[1:]
    return Path(root).joinpath(*parts, filename)


def manifest_from_meta(meta: pd.DataFrame, root: Path, data_dir: Path,
                       require_exists: bool = True) -> pd.DataFrame:
    """
    Manifest rows for a FakeAVCeleb-style meta_data.csv.

    `root` is the directory holding meta_data.csv; `data_dir` is the repo's
    data/ directory, which video_path is written relative to (the whole pipeline
    resolves clips as data_dir / video_path). With require_exists, rows whose
    file is not on disk are dropped -- a partially downloaded drop still yields a
    usable manifest instead of an error.
    """
    meta = meta.copy()
    if len(meta.columns) != len(META_COLUMNS):
        raise ValueError(
            f"meta_data.csv has {len(meta.columns)} columns, expected "
            f"{len(META_COLUMNS)}: {META_COLUMNS}"
        )
    meta.columns = META_COLUMNS  # positional rename (see note above)

    root, data_dir = Path(root), Path(data_dir)
    rows = []
    for r in meta.itertuples(index=False):
        vp = resolve_video_path(root, r.dirpath, r.filename)
        if require_exists and not vp.exists():
            continue
        rows.append({
            "clip_id": f"{r.category}__{r.source}__{Path(r.filename).stem}",
            "video_path": str(vp.relative_to(data_dir)),
            "label": clip_label(r.manipulation_type),
            "manipulation_type": r.manipulation_type,
            "method": r.method,
            "source": r.source,
            "target1": r.target1,
            "target2": r.target2,
            "race": r.race,
            "gender": r.gender,
        })

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    # meta_data.csv lists ~22 physical files twice with conflicting `method`
    # labels (e.g. 'wav2lip' vs 'faceswap-wav2lip' for the same .mp4). Dedupe
    # on the actual file path, keeping the first, so each file appears once.
    return df.drop_duplicates(subset="video_path", keep="first").reset_index(drop=True)
