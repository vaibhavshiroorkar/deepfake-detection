"""Discovery of whatever datasets happen to sit under data/, nothing hardcoded.

The dashboard used to carry a fixed registry of dataset names and manifest
paths, so a new drop only appeared after editing code and a stale entry showed
as "not available" forever. Instead we scan data/ on every refresh and infer:

  - a RAW dataset is a directory holding a FakeAVCeleb-style meta_data.csv. Its
    manifest is built in memory (preprocessing.manifest.manifest_from_meta), so
    a freshly extracted drop is usable before audit_dataset.py has ever run.
  - a MANIFEST is any CSV under data/ carrying clip_id/video_path/label. It is
    attached to the dataset its video_path column points into, which is how the
    pipeline's flat data/train.csv finds its way onto data/<drop>/, not by
    filename convention.

Pure and Streamlit-free so it can be unit-tested against a tmp_path tree; the
widgets and caching live in selectors.py.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from preprocessing.manifest import manifest_from_meta

META_NAME = "meta_data.csv"
# The columns that make a CSV a clip manifest rather than some other table.
MANIFEST_COLUMNS = {"clip_id", "video_path", "label"}
# Split name for the in-memory manifest derived straight from meta_data.csv.
RAW_SPLIT = "all (raw)"
# Pipeline splits first, in pipeline order; anything else sorts after them.
_SPLIT_ORDER = ["train", "val", "test", "full_manifest"]
_MAX_DEPTH = 3


@dataclass
class Dataset:
    """One dataset found under data/: its clip root and the manifests for it."""
    name: str
    root: Path
    manifests: dict[str, Path] = field(default_factory=dict)
    meta_csv: Path | None = None

    @property
    def splits(self) -> list[str]:
        def rank(s):
            return (_SPLIT_ORDER.index(s) if s in _SPLIT_ORDER else len(_SPLIT_ORDER), s)
        named = sorted(self.manifests, key=rank)
        return named + ([RAW_SPLIT] if self.meta_csv is not None else [])


def discover(data_dir: Path) -> dict[str, Dataset]:
    """Datasets under data/, keyed by name (their path relative to data/)."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return {}

    metas, csvs = _scan(data_dir)
    found: dict[Path, Dataset] = {
        meta.parent: Dataset(name=_name(meta.parent, data_dir), root=meta.parent,
                             meta_csv=meta)
        for meta in metas
    }
    for csv in csvs:
        if not _is_manifest(csv):
            continue
        root = _clip_root(csv, data_dir, found)
        ds = found.get(root)
        if ds is None:
            ds = found[root] = Dataset(name=_name(root, data_dir), root=root)
        ds.manifests[csv.stem] = csv

    return {ds.name: ds for ds in sorted(found.values(), key=lambda d: d.name)}


def load_split(ds: Dataset, split: str, data_dir: Path) -> pd.DataFrame:
    """The manifest frame for one split of a discovered dataset."""
    if split in ds.manifests:
        return pd.read_csv(ds.manifests[split])
    if split == RAW_SPLIT and ds.meta_csv is not None:
        return manifest_from_meta(pd.read_csv(ds.meta_csv), ds.root, data_dir)
    raise KeyError(f"{ds.name!r} has no split {split!r}")


# ------------------------------------------------------------------ internals

def _scan(data_dir: Path) -> tuple[list[Path], list[Path]]:
    """(meta_data.csv paths, other CSV paths) within _MAX_DEPTH of data/.

    A directory that holds a meta_data.csv is not descended into: it is a raw
    drop, and walking FakeAVCeleb's ~500 identity folders buys nothing.
    """
    metas, csvs, stack = [], [], [(data_dir, 0)]
    while stack:
        directory, depth = stack.pop()
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            continue
        files = [e for e in entries if e.is_file() and e.suffix.lower() == ".csv"]
        meta = next((f for f in files if f.name.lower() == META_NAME), None)
        if meta is not None:
            metas.append(meta)
        csvs.extend(f for f in files if f is not meta)
        if meta is None and depth + 1 < _MAX_DEPTH:
            stack.extend((e, depth + 1) for e in entries
                         if e.is_dir() and not e.name.startswith("."))
    return metas, csvs


def _is_manifest(csv: Path) -> bool:
    try:
        head = pd.read_csv(csv, nrows=1)
    except Exception:
        return False
    return MANIFEST_COLUMNS.issubset(head.columns)


def _clip_root(csv: Path, data_dir: Path, known: dict[Path, Dataset]) -> Path:
    """Which dataset a manifest belongs to: where its clips actually live.

    video_path is stored relative to data/, so the nearest enclosing raw drop
    owns the manifest. Falls back to the CSV's own directory for a manifest that
    points nowhere known (e.g. an external eval set with no meta_data.csv).
    """
    try:
        video_path = str(pd.read_csv(csv, nrows=1)["video_path"].iloc[0])
    except Exception:
        return csv.parent
    for parent in (data_dir / video_path).parents:
        if parent in known:
            return parent
    return csv.parent


def _name(root: Path, data_dir: Path) -> str:
    try:
        rel = root.relative_to(data_dir)
    except ValueError:
        return root.name
    return data_dir.name if rel == Path(".") else rel.as_posix()
