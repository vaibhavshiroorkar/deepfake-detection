"""Identity-separated development / sealed split of FakeAVCeleb (D-070, Option B).

Rule (fixed on 2026-10-26 before any split was computed):
  1. Identities (VoxCeleb2 ids) are grouped by their race x gender cell (read from the manifest notes ``cell=``).
  2. Inside a cell the identities are ordered by SHA-256 of ``<SALT>:<id>`` and assigned alternately to half 0 and half 1, so each cell holds
     the same number (rounded down/up by one) of identities on both sides. Half 0 is the development side (``dev_x``), half 1 the sealed
     side (``test_x``).
  3. A clip is kept only when its source identity and every target identity (``ids=`` in the notes) lie in the same half. All other clips are
     excluded from both sides and listed with the reason. Excluded clips are not placed in dev_x.
  4. ``verify`` re-derives the identity sets of both sides and fails when they intersect.

    python -m dfd.data.fakeavceleb_split --manifest data/_manifests/fakeavceleb.manifest.csv --out-dir data/_splits
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from .manifest import Record, default_data_root, read_manifest
from .splits import write_split

SALT = "FakeAVCeleb-E015-2026-10-26"
DEV, SEALED = "dev_x", "test_x"


def note_field(notes: str, key: str) -> str:
    for part in notes.split(";"):
        if part.startswith(key + "="):
            return part[len(key) + 1:]
    return ""


def clip_identities(r: Record) -> list[str]:
    return [x for x in note_field(r.notes, "ids").split("|") if x]


def half_of(identities_by_cell: dict[str, list[str]], salt: str = SALT) -> dict[str, int]:
    out: dict[str, int] = {}
    for cell, ids in identities_by_cell.items():
        ordered = sorted(set(ids), key=lambda i: hashlib.sha256(f"{salt}:{i}".encode()).hexdigest())
        for k, i in enumerate(ordered):
            out[i] = k % 2
    return out


def identity_cells(records: Sequence[Record]) -> dict[str, list[str]]:
    cells: dict[str, set[str]] = defaultdict(set)
    for r in records:
        src = clip_identities(r)[0]
        cells[note_field(r.notes, "cell")].add(src)
    return {c: sorted(v) for c, v in cells.items()}


def assign(records: Sequence[Record], salt: str = SALT) -> tuple[dict[str, str], list[dict], dict[str, int]]:
    half = half_of(identity_cells(records), salt)
    kept: dict[str, str] = {}
    excluded: list[dict] = []
    for r in records:
        ids = clip_identities(r)
        sides = {half[i] for i in ids if i in half}
        unknown = [i for i in ids if i not in half]
        if unknown:
            # a target that is not a source identity cannot be placed; exclude it explicitly
            excluded.append({"sample_id": r.sample_id, "reason": "identity without a half: " + "|".join(unknown), "ids": "|".join(ids)})
        elif len(sides) > 1:
            excluded.append({"sample_id": r.sample_id, "reason": "source and target identities on different sides", "ids": "|".join(ids)})
        else:
            kept[r.sample_id] = SEALED if sides == {1} else DEV
    return kept, excluded, half


def verify(records: Sequence[Record], kept: dict[str, str]) -> dict:
    """Identity sets per side; ok only when they do not intersect (any role) and every kept clip has all identities on its own side."""
    ids: dict[str, set[str]] = {DEV: set(), SEALED: set()}
    stems: dict[str, set[str]] = {DEV: set(), SEALED: set()}
    for r in records:
        side = kept.get(r.sample_id)
        if side:
            ids[side].update(clip_identities(r))
            stems[side].add(Path(r.path).name.split("_")[0].replace(".mp4", "") + "|" + clip_identities(r)[0])
    shared = sorted(ids[DEV] & ids[SEALED])
    shared_stems = sorted(stems[DEV] & stems[SEALED])
    return {"identities_dev": len(ids[DEV]), "identities_sealed": len(ids[SEALED]), "shared_identities": shared,
            "shared_derivative_families": shared_stems, "ok": not shared and not shared_stems}


def summarise(records: Sequence[Record], kept: dict[str, str]) -> dict:
    by = {r.sample_id: r for r in records}
    out: dict = {}
    for side in (DEV, SEALED):
        rs = [by[s] for s, v in kept.items() if v == side]
        out[side] = {"clips": len(rs), "real_label": sum(r.label == 0 for r in rs), "fake_label": sum(r.label == 1 for r in rs),
                     "categories": dict(Counter(note_field(r.notes, "category") for r in rs)),
                     "methods": dict(Counter(r.method for r in rs)), "cells": dict(Counter(note_field(r.notes, "cell") for r in rs))}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <data-root>/_splits")
    args = ap.parse_args(argv)
    out_dir = args.out_dir or (default_data_root() / "_splits")
    records = read_manifest(args.manifest)
    kept, excluded, half = assign(records)
    ver = verify(records, kept)
    if not ver["ok"]:
        print("identity separation FAILED:", ver, file=sys.stderr)
        return 1
    params = {"rule": "D-070 option B", "salt": SALT, "dev": DEV, "sealed": SEALED, "excluded": len(excluded)}
    kept_records = [r for r in records if r.sample_id in kept]
    split_id, csv_path = write_split(kept_records, kept, out_dir, "fakeavceleb", params, [args.manifest])
    exc_path = out_dir / f"{split_id}.exclusions.csv"
    with open(exc_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample_id", "reason", "ids"])
        w.writeheader()
        w.writerows(excluded)
    report = {"split_id": split_id, "records": len(records), "kept": len(kept), "excluded": len(excluded),
              "excluded_reasons": dict(Counter(e["reason"].split(":")[0] for e in excluded)), "verify": ver,
              "summary": summarise(records, kept), "halves": {"half0_dev": sum(1 for v in half.values() if v == 0),
                                                              "half1_sealed": sum(1 for v in half.values() if v == 1)}}
    (out_dir / f"{split_id}.report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("split_id", "kept", "excluded", "excluded_reasons", "verify", "halves")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
