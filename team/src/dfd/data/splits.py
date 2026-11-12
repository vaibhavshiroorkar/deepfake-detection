"""Deterministic, group-aware splits (PROTOCOL section 1; D-012).

Split names: ``train``, ``val``, ``test_in``, ``test_x``, ``dev_x`` (D-035: a development slice of a cross-dataset source,
not a final split).

Modes
-----
official   Use the manifest's ``official_split`` column mapped by OFFICIAL_MAP
           (train->train, val/dev->val, test/eval->test_in). Every row must have one. A group that sits in two
           official splits is an error (a group is never split).
group      Seeded split of whole groups into the parts given by ``fractions`` (for example
           ``{"train": 0.7, "val": 0.15, "test_in": 0.15}``). Stratified: groups are first ordered by their
           class composition so both classes are spread over the parts, the part with the largest remaining
           sample deficit (per class) takes each group, and a repair step moves whole groups so that each
           non-empty part holds both classes whenever the data allows it.
dev_slice  D-035: sort the unit names, draw numpy default_rng(20261024).permutation(n_units), the first k units go to
           ``dev_x`` (every sample of a drawn unit), all other units stay ``test_x``. The unit is the group
           (``--dev-unit group``) or, for the CNN subset, the generator folder (``--dev-unit generator``: the
           second part of the group id, so every class group of a generator moves together). With
           ``--dev-identity-only`` only groups with identity_known true are eligible (Celeb-DF: the 59 identity
           groups; the 300 YouTube-real videos have no identity and stay test_x).
fixed      All rows get one named split (for example ``test_x`` for a dataset never used in training, or
           ``test_in`` for the ProGAN test set).
auto       ``official`` when every row has an official split, ``group`` when none has, error when mixed.

Output: data/_splits/<split_id>.csv (sample_id, split; rows sorted by sample_id, LF line ends) and
<split_id>.json (parameters, seed, counts, manifest hashes, SHA-256 of the CSV). The split id is
``<name>-<first 12 hex digits of the CSV SHA-256>``, so identical content gives the identical id.

CLI: python -m dfd.data.splits --manifest data/_manifests/x.manifest.csv --name x --mode auto --seed 1 \
         --fractions train=0.7,val=0.15,test_in=0.15
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from .manifest import Record, default_data_root, read_manifest

SPLIT_NAMES = ("train", "val", "test_in", "test_x", "dev_x")
DEV_SLICE_SEED = 20261024  # D-035
OFFICIAL_MAP = {"train": "train", "val": "val", "dev": "val", "test": "test_in", "eval": "test_in"}


class SplitError(ValueError):
    pass


def _groups(records: Sequence[Record]) -> dict[str, list[Record]]:
    g: dict[str, list[Record]] = {}
    for r in records:
        g.setdefault(r.group_id, []).append(r)
    return g


def assign_official(records: Sequence[Record]) -> dict[str, str]:
    out: dict[str, str] = {}
    for r in records:
        if r.official_split not in OFFICIAL_MAP:
            raise SplitError(f"{r.sample_id}: official_split {r.official_split!r} not in {sorted(OFFICIAL_MAP)}")
        out[r.sample_id] = OFFICIAL_MAP[r.official_split]
    for gid, rows in _groups(records).items():
        parts = {out[r.sample_id] for r in rows}
        if len(parts) > 1:
            raise SplitError(f"group {gid} spans official splits {sorted(parts)}; a group is never split")
    return out


def assign_fixed(records: Sequence[Record], split: str) -> dict[str, str]:
    if split not in SPLIT_NAMES:
        raise SplitError(f"unknown split {split!r}")
    return {r.sample_id: split for r in records}


def assign_groups(records: Sequence[Record], fractions: dict[str, float], seed: int) -> dict[str, str]:
    """Seeded group-level split, stratified by class, never splitting a group."""
    parts = [p for p in SPLIT_NAMES if fractions.get(p, 0) > 0]
    if not parts or any(p not in SPLIT_NAMES for p in fractions):
        raise SplitError(f"fractions must use {SPLIT_NAMES} with positive values: {fractions}")
    total_f = sum(fractions[p] for p in parts)
    frac = {p: fractions[p] / total_f for p in parts}
    groups = _groups(records)
    # per-group class counts [n_real, n_fake]
    cnt = {g: [sum(r.label == 0 for r in rs), sum(r.label == 1 for r in rs)] for g, rs in groups.items()}
    totals = [sum(c[0] for c in cnt.values()), sum(c[1] for c in cnt.values())]
    target = {p: [frac[p] * totals[0], frac[p] * totals[1]] for p in parts}
    cur = {p: [0, 0] for p in parts}
    rng = random.Random(seed)
    ids = sorted(groups)
    rng.shuffle(ids)

    # Stratum by class composition (mixed, fake-only, real-only); the seeded shuffle order is kept inside a stratum.
    def stratum(g: str) -> int:
        r, f = cnt[g]
        return 0 if (r and f) else (1 if f else 2)

    ids.sort(key=stratum)  # stable sort
    where: dict[str, str] = {}

    def deficit(p: str, g: str) -> float:
        d = 0.0
        for k in (0, 1):
            if cnt[g][k] and target[p][k] > 0:
                d += cnt[g][k] * (target[p][k] - cur[p][k]) / target[p][k]
        return d

    order_tiebreak = {p: i for i, p in enumerate(parts)}
    for g in ids:
        best = max(parts, key=lambda p: (deficit(p, g), -order_tiebreak[p]))
        where[g] = best
        cur[best][0] += cnt[g][0]
        cur[best][1] += cnt[g][1]

    # Repair: each part should hold both classes when the data allows it.
    for k in (0, 1):
        for p in parts:
            if totals[k] == 0 or cur[p][k] > 0:
                continue
            donors = sorted(
                (g for g in ids if cnt[g][k] and where[g] != p),
                key=lambda g: (-cnt[g][k], g),
            )
            for g in donors:
                d = where[g]
                other = 1 - k
                if cur[d][k] - cnt[g][k] <= 0:
                    continue
                if cur[d][other] - cnt[g][other] <= 0 and cur[d][other] > 0 and cnt[g][other] > 0:
                    continue
                for j in (0, 1):
                    cur[d][j] -= cnt[g][j]
                    cur[p][j] += cnt[g][j]
                where[g] = p
                break
    return {r.sample_id: where[r.group_id] for r in records}


def unit_of(r: Record, unit: str) -> str:
    if unit == "group":
        return r.group_id
    if unit == "generator":  # group ids look like <dataset>:<generator>[:<class>]
        return r.group_id.split(":")[1]
    raise SplitError(f"unknown dev slice unit {unit!r}")


def draw_dev_units(units: Sequence[str], k: int, seed: int = DEV_SLICE_SEED) -> list[str]:
    """D-035 rule: sort the names, permute with numpy default_rng(seed), take the first k."""
    import numpy as np

    names = sorted(set(units))
    if not 0 < k < len(names):
        raise SplitError(f"k must satisfy 0 < k < {len(names)} units, got {k}")
    perm = np.random.default_rng(seed).permutation(len(names))
    return [names[i] for i in perm[:k]]


def merge_units_by_real_content(records: Sequence[Record], unit: str, data_root: Path,
                                cache: "HashCache | None" = None) -> dict[str, str]:
    """D-035 Amendment 1: units whose REAL (label 0) files share a content hash form one unit.

    Returns {base unit name: merged unit name}; a merged name is the sorted member names joined by '+'.
    Units with no shared real content map to themselves. Nothing is hard-coded about names."""
    from .builders.ffpp import UnionFind
    from .hashing import HashCache

    cache = cache or HashCache(None)
    uf = UnionFind()
    first_unit_of_hash: dict[str, str] = {}
    for r in sorted(records, key=lambda x: x.sample_id):
        u = unit_of(r, unit)
        uf.add(u)
        if r.label != 0:
            continue
        h = cache.sha256(data_root / r.path, r.path)
        if h in first_unit_of_hash:
            uf.union(first_unit_of_hash[h], u)
        else:
            first_unit_of_hash[h] = u
    members: dict[str, list[str]] = {}
    for u in list(uf.parent):
        members.setdefault(uf.find(u), []).append(u)
    out: dict[str, str] = {}
    for mem in members.values():
        name = "+".join(sorted(mem))
        for u in mem:
            out[u] = name
    return out


def assign_dev_slice(records: Sequence[Record], k: int, unit: str = "group", seed: int = DEV_SLICE_SEED,
                     identity_only: bool = False,
                     unit_map: Mapping[str, str] | None = None) -> tuple[dict[str, str], list[str]]:
    """Units are drawn from all records, or (identity_only) only from records with identity_known true;
    records outside the eligible set (for example Celeb-DF YouTube-real videos, which carry no identity)
    always stay test_x. ``unit_map`` renames units before the draw (see merge_units_by_real_content)."""
    def u_of(r: Record) -> str:
        u = unit_of(r, unit)
        return unit_map.get(u, u) if unit_map else u

    eligible = [r for r in records if r.identity_known or not identity_only]
    drawn = set(draw_dev_units([u_of(r) for r in eligible], k, seed))
    ok = {r.sample_id for r in eligible}
    asg = {r.sample_id: ("dev_x" if r.sample_id in ok and u_of(r) in drawn else "test_x") for r in records}
    return asg, sorted(drawn)


def assign(records: Sequence[Record], mode: str = "auto", seed: int = 0,
           fractions: dict[str, float] | None = None, fixed: str | None = None,
           dev_k: int | None = None, dev_unit: str = "group", dev_identity_only: bool = False,
           dev_unit_map: Mapping[str, str] | None = None) -> dict[str, str]:
    if mode == "dev_slice":
        if dev_k is None:
            raise SplitError("mode dev_slice needs --dev-k")
        return assign_dev_slice(records, dev_k, dev_unit, identity_only=dev_identity_only, unit_map=dev_unit_map)[0]
    if mode == "fixed":
        if fixed is None:
            raise SplitError("mode fixed needs --fixed <split>")
        return assign_fixed(records, fixed)
    if mode == "auto":
        have = sum(bool(r.official_split) for r in records)
        if have == len(records) and records:
            mode = "official"
        elif have == 0:
            mode = "group"
        else:
            raise SplitError(f"mixed: {have} of {len(records)} rows have an official split; choose a mode")
    if mode == "official":
        return assign_official(records)
    if mode == "group":
        if not fractions:
            raise SplitError("mode group needs fractions")
        return assign_groups(records, fractions, seed)
    raise SplitError(f"unknown mode {mode!r}")


def split_csv_bytes(assignment: dict[str, str]) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["sample_id", "split"])
    for sid in sorted(assignment):
        w.writerow([sid, assignment[sid]])
    return buf.getvalue().encode("utf-8")


def read_split(path: str | Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd, None)
        if header != ["sample_id", "split"]:
            raise SplitError(f"{path}: header {header} != ['sample_id', 'split']")
        for n, row in enumerate(rd, start=2):
            if len(row) != 2 or row[1] not in SPLIT_NAMES:
                raise SplitError(f"{path}:{n}: bad row {row}")
            if row[0] in out:
                raise SplitError(f"{path}:{n}: duplicate sample_id {row[0]}")
            out[row[0]] = row[1]
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_split(records: Sequence[Record], assignment: dict[str, str], out_dir: Path, name: str,
                params: dict, manifest_paths: Sequence[Path] = ()) -> tuple[str, Path]:
    """Write <split_id>.csv and <split_id>.json into out_dir. Returns (split_id, csv path)."""
    data = split_csv_bytes(assignment)
    digest = hashlib.sha256(data).hexdigest()
    split_id = f"{name}-{digest[:12]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{split_id}.csv"
    csv_path.write_bytes(data)
    by_id = {r.sample_id: r for r in records}
    counts: dict[str, dict] = {}
    for sid, sp in assignment.items():
        r = by_id[sid]
        c = counts.setdefault(sp, {"samples": 0, "real": 0, "fake": 0, "groups": set()})
        c["samples"] += 1
        c["real" if r.label == 0 else "fake"] += 1
        c["groups"].add(r.group_id)
    for c in counts.values():
        c["groups"] = len(c["groups"])
    sidecar = {
        "split_id": split_id,
        "name": name,
        "csv_sha256": digest,
        "csv_file": csv_path.name,
        "rows": len(assignment),
        "counts": {k: counts[k] for k in sorted(counts)},
        "parameters": params,
        "manifests": [{"file": p.name, "sha256": sha256_file(p)} for p in manifest_paths],
    }
    (out_dir / f"{split_id}.json").write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")
    return split_id, csv_path


def _parse_fractions(text: str) -> dict[str, float]:
    out = {}
    for part in text.split(","):
        k, v = part.split("=")
        out[k.strip()] = float(v)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--name", required=True, help="prefix of the split id")
    ap.add_argument("--mode", default="auto", choices=["auto", "official", "group", "fixed", "dev_slice"])
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--fractions", type=_parse_fractions, default=None, help="e.g. train=0.8,val=0.2")
    ap.add_argument("--fixed", default=None, choices=SPLIT_NAMES, help="for mode fixed")
    ap.add_argument("--dev-k", type=int, default=None, help="mode dev_slice: number of units drawn into dev_x")
    ap.add_argument("--dev-identity-only", action="store_true", help="draw only among groups with identity_known true; others stay test_x")
    ap.add_argument("--dev-unit", default="group", choices=["group", "generator"])
    ap.add_argument("--dev-merge-real-content", action="store_true",
                    help="D-035 Amendment 1: units whose real files share content hashes are one draw unit")
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <data>/_splits")
    args = ap.parse_args(argv)
    out_dir = args.out_dir or (default_data_root() / "_splits")
    unit_map = None
    try:
        recs = read_manifest(args.manifest)
        if args.dev_merge_real_content:
            from .hashing import HashCache

            root = (args.data_root or default_data_root()).resolve()
            cache = HashCache(root / "_manifests" / "content_hash_cache.json")
            unit_map = merge_units_by_real_content(recs, args.dev_unit, root, cache)
            cache.save()
        asg = assign(recs, args.mode, args.seed, args.fractions, args.fixed, args.dev_k, args.dev_unit,
                     args.dev_identity_only, unit_map)
    except (SplitError, ValueError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    params = {"mode": args.mode, "seed": args.seed, "fractions": args.fractions, "fixed": args.fixed}
    if args.mode == "dev_slice":
        _, drawn = assign_dev_slice(recs, args.dev_k, args.dev_unit, identity_only=args.dev_identity_only,
                                    unit_map=unit_map)
        all_units = {(unit_map or {}).get(unit_of(r, args.dev_unit), unit_of(r, args.dev_unit)) for r in recs
                     if r.identity_known or not args.dev_identity_only}
        params.update({"dev_k": args.dev_k, "dev_unit": args.dev_unit, "dev_seed": DEV_SLICE_SEED,
                       "dev_identity_only": args.dev_identity_only,
                       "dev_merge_real_content": args.dev_merge_real_content,
                       "units_total": len(all_units), "units": sorted(all_units), "dev_units": drawn})
    sid, p = write_split(recs, asg, out_dir, args.name, params, [args.manifest])
    c = Counter(asg.values())
    print(f"split_id {sid}: {dict(c)} -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
