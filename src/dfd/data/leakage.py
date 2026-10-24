"""Leakage checks over one or more manifests and split files (PROTOCOL A6, D-012, CLAUDE.md rule 3).

Failures (the CLI exits 1 on any of them):
  1. GROUP_OVERLAP        a group_id has samples in more than one split.
  2. FAMILY_LEAK          a source family appears in train or val and also in test_x. ``merge_families`` lets the
                          caller treat several family names as one (for example lavdf and fakeavceleb).
  3. DUPLICATE_SAMPLE_ID  the same sample_id appears twice in the manifests or in the split files.
  4. UNKNOWN_SAMPLE       a split row names a sample that no manifest contains.
  5. DUPLICATE_CONTENT    (only with hashing on) two files with the same SHA-256 sit in different splits. Equal
                          files inside one split are a warning unless ``strict_duplicates`` is set.
  6. IDENTITY_OVERLAP     two families listed in POSSIBLE_SHARED_FAMILIES are used one in train/val and one in
                          test_x, and they share an identity (idNNNNN). If no identity can be read from the
                          rows, only a warning is given (the check could not be made).
Warnings: manifest rows with no split (ignored by the checks), same-split duplicate content.

Exit codes of the CLI: 0 clean, 1 leakage found, 2 input error.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .hashing import HashCache, sha256_of  # noqa: F401  (sha256_of re-exported)
from .manifest import Record, default_data_root

TRAIN_LIKE = ("train", "val")
# Families that may share real sources or identities although they are different datasets (D-021, DATA_LOG.md).
POSSIBLE_SHARED_FAMILIES = (frozenset({"lavdf", "fakeavceleb"}),)
VOX_ID_RE = re.compile(r"id\d{5}")


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def text(self) -> str:
        lines = ["LEAKAGE CHECK REPORT"]
        lines += [f"  info: {m}" for m in self.info]
        lines += [f"  WARNING: {m}" for m in self.warnings]
        lines += [f"  FAIL: {m}" for m in self.errors]
        lines.append("RESULT: " + ("PASS (no leakage found)" if self.ok else f"FAIL ({len(self.errors)} problem(s))"))
        return "\n".join(lines)


def identity_set(records: Iterable[Record], pattern: re.Pattern[str] = VOX_ID_RE) -> set[str]:
    """Identities (default VoxCeleb2 ids idNNNNN) read from group_id, path and notes of the records."""
    out: set[str] = set()
    for r in records:
        for text in (r.group_id, r.path, r.notes):
            out.update(pattern.findall(text))
    return out


def identity_overlap(a: Iterable[Record], b: Iterable[Record],
                     pattern: re.Pattern[str] = VOX_ID_RE) -> dict:
    """Compare the identity sets of two record collections (for example LAV-DF against FakeAVCeleb)."""
    sa, sb = identity_set(a, pattern), identity_set(b, pattern)
    both = sorted(sa & sb)
    return {"a_identities": len(sa), "b_identities": len(sb), "shared": both, "shared_count": len(both)}


def check(
    records: Sequence[Record],
    splits: Mapping[str, str],
    *,
    split_row_count: int | None = None,
    merge_families: Sequence[Iterable[str]] = (),
    data_root: Path | None = None,
    hash_files: bool = False,
    strict_duplicates: bool = False,
    cache: HashCache | None = None,
) -> Report:
    """Run all checks. ``records`` may come from several manifests; ``splits`` maps sample_id to split."""
    rep = Report()
    # 3. duplicate sample ids in the manifests
    seen: dict[str, Record] = {}
    dup_set: set[str] = set()
    for r in records:
        if r.sample_id in seen:
            dup_set.add(r.sample_id)
        else:
            seen[r.sample_id] = r
    dups = sorted(dup_set)
    if dups:
        rep.errors.append(f"DUPLICATE_SAMPLE_ID: {len(dups)} id(s) appear more than once, first: {dups[:3]}")
    if split_row_count is not None and split_row_count != len(splits):
        rep.errors.append(
            f"DUPLICATE_SAMPLE_ID: split files hold {split_row_count} rows but {len(splits)} distinct ids"
        )
    by_id = seen
    # 4. unknown samples
    unknown = sorted(s for s in splits if s not in by_id)
    if unknown:
        rep.errors.append(f"UNKNOWN_SAMPLE: {len(unknown)} split row(s) not in any manifest, first: {unknown[:3]}")
    unassigned = sum(1 for s in by_id if s not in splits)
    if unassigned:
        rep.warnings.append(f"{unassigned} manifest row(s) have no split and are ignored by the checks")
    assigned = [(by_id[s], sp) for s, sp in splits.items() if s in by_id]
    rep.info.append(f"{len(by_id)} manifest rows, {len(assigned)} assigned to a split")

    # 1. group overlap
    group_splits: dict[str, set[str]] = {}
    for r, sp in assigned:
        group_splits.setdefault(r.group_id, set()).add(sp)
    bad_groups = {g: sorted(s) for g, s in group_splits.items() if len(s) > 1}
    if bad_groups:
        first = sorted(bad_groups.items())[:3]
        rep.errors.append(f"GROUP_OVERLAP: {len(bad_groups)} group(s) in more than one split, first: {first}")
    rep.info.append(f"{len(group_splits)} groups checked")

    # 2. family leak
    merge = {}
    for fam_set in merge_families:
        fam_set = set(fam_set)
        name = "+".join(sorted(fam_set))
        for f in fam_set:
            merge[f] = name

    def fam(r: Record) -> str:
        return merge.get(r.family, r.family)

    fam_splits: dict[str, set[str]] = {}
    for r, sp in assigned:
        fam_splits.setdefault(fam(r), set()).add(sp)
    for f, sps in sorted(fam_splits.items()):
        if "test_x" in sps and (sps & set(TRAIN_LIKE)):
            rep.errors.append(
                f"FAMILY_LEAK: family {f!r} is in test_x and also in {sorted(sps & set(TRAIN_LIKE))}"
            )
    rep.info.append("families: " + ", ".join(f"{f}={sorted(s)}" for f, s in sorted(fam_splits.items())))

    # 6. possible shared families (identity based)
    for pair in POSSIBLE_SHARED_FAMILIES:
        a, b = sorted(pair)
        for train_fam, x_fam in ((a, b), (b, a)):
            tr = [r for r, sp in assigned if r.family == train_fam and sp in TRAIN_LIKE]
            tx = [r for r, sp in assigned if r.family == x_fam and sp == "test_x"]
            if not tr or not tx:
                continue
            res = identity_overlap(tr, tx)
            if res["a_identities"] == 0 or res["b_identities"] == 0:
                rep.warnings.append(
                    f"possible shared family {train_fam}/{x_fam}: no identities readable, overlap NOT checked"
                )
            elif res["shared_count"]:
                rep.errors.append(
                    f"IDENTITY_OVERLAP: {train_fam} (train/val) and {x_fam} (test_x) share "
                    f"{res['shared_count']} identities, first: {res['shared'][:3]}"
                )
            else:
                rep.info.append(f"identity overlap {train_fam} vs {x_fam}: none ({res['a_identities']} vs {res['b_identities']} ids)")

    # 5. duplicate content
    if hash_files:
        root = data_root or default_data_root()
        cache = cache or HashCache(None)
        by_hash: dict[str, list[tuple[str, str]]] = {}
        missing = 0
        for r, sp in assigned:
            f = root / r.path
            if not f.is_file():
                missing += 1
                continue
            by_hash.setdefault(cache.sha256(f, r.path), []).append((r.sample_id, sp))
        if missing:
            rep.errors.append(f"DUPLICATE_CONTENT: could not hash {missing} file(s) (missing), check is incomplete")
        cross = {h: v for h, v in by_hash.items() if len({sp for _, sp in v}) > 1}
        same = {h: v for h, v in by_hash.items() if len(v) > 1 and h not in cross}
        if cross:
            first = next(iter(cross.values()))[:3]
            rep.errors.append(f"DUPLICATE_CONTENT: {len(cross)} content hash(es) in different splits, first: {first}")
        if same:
            msg = f"{len(same)} content hash(es) duplicated inside one split, first: {next(iter(same.values()))[:3]}"
            if strict_duplicates:
                rep.errors.append("DUPLICATE_CONTENT: " + msg)
            else:
                rep.warnings.append(msg)
        rep.info.append(f"hashed {sum(len(v) for v in by_hash.values())} files")
    else:
        rep.info.append("file content hashing: off (use --hash-files)")
    return rep


def load_inputs(manifests: Sequence[Path], split_files: Sequence[Path]) -> tuple[list[Record], dict[str, str], int]:
    from .manifest import read_manifest
    from .splits import SplitError, read_split

    recs: list[Record] = []
    for m in manifests:
        recs.extend(read_manifest(m, validate=False))
    splits: dict[str, str] = {}
    total_rows = 0
    for s in split_files:
        part = read_split(s)
        total_rows += len(part)
        for k, v in part.items():
            splits[k] = v
    # A sample_id repeated across split files shows up as total_rows != len(splits), which check() reports.
    return recs, splits, total_rows
