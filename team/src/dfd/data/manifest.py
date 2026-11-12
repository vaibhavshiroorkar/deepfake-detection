"""Dataset manifest: schema, writer, reader and validation.

A manifest is a UTF-8 CSV with a header and one row per sample (one image, one video clip or one audio
clip). It is the single description of a dataset that the split, leakage-check and evaluation code read.

Schema (column order is fixed)
------------------------------
sample_id       Stable, unique across all datasets, for example ``ffpp:Deepfakes:123_456`` or
                ``asvspoof2019_la:LA_T_1138215``. Built as ``<dataset>:<local id>``.
dataset         Builder key, for example ``ffpp``, ``asvspoof2019_la``, ``forensynths_progan_val``.
family          Source family for the cross-dataset rule (PROTOCOL section 1). Datasets that share
                real sources or identities share a family (Celeb-DF v2 and Celeb-DF++ are one family).
                A family used in training must never be a ``test_x``.
path            File path relative to the repository ``data/`` folder, always with forward slashes.
modality        ``video``, ``audio`` or ``image``.
label           0 real, 1 fake.
method          Generator or manipulation method; the literal ``real`` for every label 0 row.
group_id        The unit that may sit on one side of a split only (``<dataset>:<local group>``).
group_basis     How the group was formed: ``source_video``, ``identity``, ``class`` or ``generator``.
identity_known  ``true`` when the group reflects person identity, ``false`` when the group falls back to
                the source video, class or generator (D-012, PROTOCOL section 1).
official_split  The dataset's own split name (``train``, ``val``, ``dev``, ``test``, ``eval``) or empty.
notes           Free text, never used by the code except where a builder documents it (for example
                ``ids=id0|id16`` for Celeb-DF fakes).

Group ids are namespaced by dataset, so equal local names in two datasets never collide. Overlap of
identities between two datasets is a separate check (``dfd.data.leakage.identity_overlap``).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence

COLUMNS = (
    "sample_id",
    "dataset",
    "family",
    "path",
    "modality",
    "label",
    "method",
    "group_id",
    "group_basis",
    "identity_known",
    "official_split",
    "notes",
)

MODALITIES = ("video", "audio", "image")
GROUP_BASES = ("source_video", "identity", "class", "generator")
REAL_METHOD = "real"


class ManifestError(ValueError):
    """Raised when a manifest or a record violates the schema."""


@dataclass(frozen=True)
class Record:
    sample_id: str
    dataset: str
    family: str
    path: str
    modality: str
    label: int
    method: str
    group_id: str
    group_basis: str
    identity_known: bool
    official_split: str = ""
    notes: str = ""

    def to_row(self) -> list[str]:
        return [
            self.sample_id,
            self.dataset,
            self.family,
            self.path,
            self.modality,
            str(int(self.label)),
            self.method,
            self.group_id,
            self.group_basis,
            "true" if self.identity_known else "false",
            self.official_split,
            self.notes,
        ]


def _parse_bool(text: str) -> bool:
    t = text.strip().lower()
    if t == "true":
        return True
    if t == "false":
        return False
    raise ManifestError(f"identity_known must be 'true' or 'false', got {text!r}")


def validate_record(r: Record) -> list[str]:
    """Return a list of problems with one record (empty when it is valid)."""
    problems: list[str] = []
    for name in ("sample_id", "dataset", "family", "path", "method", "group_id"):
        if not getattr(r, name):
            problems.append(f"{r.sample_id or '<no id>'}: empty {name}")
    if not r.sample_id.startswith(r.dataset + ":"):
        problems.append(f"{r.sample_id}: sample_id must start with '<dataset>:' ({r.dataset}:)")
    if r.group_id and not r.group_id.startswith(r.dataset + ":"):
        problems.append(f"{r.sample_id}: group_id {r.group_id!r} must start with '<dataset>:'")
    if r.modality not in MODALITIES:
        problems.append(f"{r.sample_id}: modality {r.modality!r} not in {MODALITIES}")
    if r.label not in (0, 1):
        problems.append(f"{r.sample_id}: label must be 0 or 1, got {r.label!r}")
    if r.group_basis not in GROUP_BASES:
        problems.append(f"{r.sample_id}: group_basis {r.group_basis!r} not in {GROUP_BASES}")
    if r.label == 0 and r.method != REAL_METHOD:
        problems.append(f"{r.sample_id}: label 0 requires method 'real', got {r.method!r}")
    if r.label == 1 and r.method == REAL_METHOD:
        problems.append(f"{r.sample_id}: label 1 cannot have method 'real'")
    if r.identity_known and r.group_basis != "identity":
        problems.append(f"{r.sample_id}: identity_known is true but group_basis is {r.group_basis!r}")
    p = r.path
    if "\\" in p or p.startswith("/") or (len(p) > 1 and p[1] == ":") or ".." in p.split("/"):
        problems.append(f"{r.sample_id}: path must be relative to data/ with forward slashes, got {p!r}")
    return problems


def validate_records(records: Sequence[Record]) -> list[str]:
    """Validate every record and the uniqueness of sample_id. Returns all problems found."""
    problems: list[str] = []
    seen: set[str] = set()
    dup: set[str] = set()
    for r in records:
        problems.extend(validate_record(r))
        if r.sample_id in seen:
            dup.add(r.sample_id)
        seen.add(r.sample_id)
    for d in sorted(dup):
        problems.append(f"duplicate sample_id {d}")
    return problems


def write_manifest(path: str | Path, records: Iterable[Record]) -> int:
    """Validate and write records as UTF-8 CSV (LF line ends). Returns the row count."""
    recs = list(records)
    problems = validate_records(recs)
    if problems:
        shown = "; ".join(problems[:5])
        raise ManifestError(f"{len(problems)} manifest problem(s), first: {shown}")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(COLUMNS)
        for r in recs:
            w.writerow(r.to_row())
    return len(recs)


def read_manifest(path: str | Path, validate: bool = True) -> list[Record]:
    """Read a manifest CSV. Raises ManifestError on a wrong header, bad value or (if validate) a rule break."""
    p = Path(path)
    with open(p, "r", encoding="utf-8", newline="") as fh:
        rd = csv.reader(fh)
        try:
            header = next(rd)
        except StopIteration:
            raise ManifestError(f"{p}: empty file") from None
        if tuple(header) != COLUMNS:
            raise ManifestError(f"{p}: header {header} != {list(COLUMNS)}")
        out: list[Record] = []
        for n, row in enumerate(rd, start=2):
            if len(row) != len(COLUMNS):
                raise ManifestError(f"{p}:{n}: expected {len(COLUMNS)} fields, got {len(row)}")
            d = dict(zip(COLUMNS, row))
            try:
                label = int(d["label"])
            except ValueError:
                raise ManifestError(f"{p}:{n}: label {d['label']!r} is not an integer") from None
            out.append(
                Record(
                    sample_id=d["sample_id"],
                    dataset=d["dataset"],
                    family=d["family"],
                    path=d["path"],
                    modality=d["modality"],
                    label=label,
                    method=d["method"],
                    group_id=d["group_id"],
                    group_basis=d["group_basis"],
                    identity_known=_parse_bool(d["identity_known"]),
                    official_split=d["official_split"],
                    notes=d["notes"],
                )
            )
    if validate:
        problems = validate_records(out)
        if problems:
            raise ManifestError(f"{p}: {len(problems)} problem(s), first: {problems[0]}")
    return out


def rel_posix(path: Path, data_root: Path) -> str:
    """Path of ``path`` relative to ``data_root`` with forward slashes."""
    return path.relative_to(data_root).as_posix()


def default_data_root() -> Path:
    """The repository data/ folder (three levels above this file's package: src/dfd/data)."""
    return Path(__file__).resolve().parents[3] / "data"


assert [f.name for f in fields(Record)] == list(COLUMNS)
