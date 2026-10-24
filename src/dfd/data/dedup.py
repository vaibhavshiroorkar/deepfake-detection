"""Remove from dev_x and test_x every sample whose file content equals a development sample (D-026, D-035, D-012).

Rule, in two steps.
 1. A ``dev_x`` sample is dropped when its SHA-256 equals that of a ``train``, ``val`` or ``test_in`` sample
    (this extends D-026 to dev_x: a development slice must not hold training content).
 2. A ``test_x`` sample is dropped when its SHA-256 equals that of a ``train``, ``val``, ``test_in`` or a
    remaining ``dev_x`` sample (D-026 and D-035).
Duplicates inside one split are kept (the leakage check reports them as warnings). The train, val and
test_in sides are never changed. The samples may come from several manifests.

Output. For every input split file that loses rows, a new split file (new id, derived from the new content)
plus its sidecar JSON, with ``dedup_of`` (old split id), ``original_parameters``, the number dropped and the
exclusions file name; split files without exclusions are left as they are. One exclusions file
``<out_dir>/<name>.exclusions.csv`` lists, for each dropped sample: sample_id, split it was dropped from,
dataset, group_id, sha256, the kept sample with the same content, and the reason.
``<name>.exclusions.json`` holds the counts.

CLI: python -m dfd.data.dedup --manifest M1 M2 ... --split S1 S2 ... --name images [--cache FILE]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from .hashing import HashCache
from .manifest import ManifestError, Record, default_data_root
from .splits import SplitError, read_split, write_split

DEV_SPLITS = ("train", "val", "test_in", "dev_x")
FINAL_OR_TRAIN = ("train", "val", "test_in")
REASON_DEV = "dev_x content equals a train, val or test_in sample (D-026 extended, D-035)"
REASON = "content equals a train, val, test_in or dev_x sample (D-026, D-035)"


def find_exclusions(records: Sequence[Record], assignment: Mapping[str, str], data_root: Path,
                    cache: HashCache | None = None) -> list[dict]:
    """Return one dict per dev_x or test_x sample to drop (see the module docstring)."""
    cache = cache or HashCache(None)
    by_id = {r.sample_id: r for r in records}
    rows = [(by_id[s], sp) for s, sp in sorted(assignment.items()) if s in by_id]
    hashes: dict[str, str] = {}
    for r, sp in rows:
        if sp in DEV_SPLITS or sp == "test_x":
            f = data_root / r.path
            if not f.is_file():
                raise FileNotFoundError(f"cannot hash {r.path}: file missing")
            hashes[r.sample_id] = cache.sha256(f, r.path)
    base_hash: dict[str, str] = {}
    for r, sp in rows:
        if sp in FINAL_OR_TRAIN:
            base_hash.setdefault(hashes[r.sample_id], r.sample_id)
    out: list[dict] = []
    dropped: set[str] = set()
    for r, sp in rows:
        if sp == "dev_x" and hashes[r.sample_id] in base_hash:
            out.append({"sample_id": r.sample_id, "split": "dev_x", "dataset": r.dataset, "group_id": r.group_id,
                        "sha256": hashes[r.sample_id], "matches": base_hash[hashes[r.sample_id]],
                        "reason": REASON_DEV})
            dropped.add(r.sample_id)
    kept_hash = dict(base_hash)
    for r, sp in rows:
        if sp == "dev_x" and r.sample_id not in dropped:
            kept_hash.setdefault(hashes[r.sample_id], r.sample_id)
    for r, sp in rows:
        if sp == "test_x" and hashes[r.sample_id] in kept_hash:
            out.append({"sample_id": r.sample_id, "split": "test_x", "dataset": r.dataset, "group_id": r.group_id,
                        "sha256": hashes[r.sample_id], "matches": kept_hash[hashes[r.sample_id]],
                        "reason": REASON})
    return out


def apply_exclusions(assignment: Mapping[str, str], excluded_ids: set[str]) -> dict[str, str]:
    return {s: sp for s, sp in assignment.items() if s not in excluded_ids}


def _split_name(csv_path: Path) -> str:
    side = csv_path.with_suffix(".json")
    if side.exists():
        try:
            return json.loads(side.read_text(encoding="utf-8"))["name"]
        except (KeyError, ValueError):
            pass
    return csv_path.stem.rsplit("-", 1)[0]


def main(argv: list[str] | None = None) -> int:
    from .manifest import read_manifest

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="+", required=True, type=Path)
    ap.add_argument("--split", nargs="+", required=True, type=Path)
    ap.add_argument("--name", required=True, help="prefix of the exclusions file")
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <data>/_splits")
    ap.add_argument("--cache", type=Path, default=None, help="hash cache file (default: <data>/_manifests/content_hash_cache.json)")
    args = ap.parse_args(argv)
    root = (args.data_root or default_data_root()).resolve()
    out_dir = args.out_dir or (root / "_splits")
    cache = HashCache(args.cache or (root / "_manifests" / "content_hash_cache.json"))
    try:
        recs: list[Record] = []
        for m in args.manifest:
            recs.extend(read_manifest(m))
        per_file = {s: read_split(s) for s in args.split}
        combined: dict[str, str] = {}
        for a in per_file.values():
            for k, v in a.items():
                if k in combined:
                    raise SplitError(f"sample_id {k} appears in more than one split file")
                combined[k] = v
        excl = find_exclusions(recs, combined, root, cache)
    except (ManifestError, SplitError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    finally:
        cache.save()
    out_dir.mkdir(parents=True, exist_ok=True)
    ex_csv = out_dir / f"{args.name}.exclusions.csv"
    cols = ["sample_id", "split", "dataset", "group_id", "sha256", "matches", "reason"]
    with open(ex_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(excl)
    ex_ids = {e["sample_id"] for e in excl}
    new_ids: dict[str, str] = {}
    kept: dict[str, int] = {}
    for sp_file, asg in per_file.items():
        drop_here = ex_ids & set(asg)
        if not drop_here:
            continue
        old_name = _split_name(sp_file)
        new_asg = apply_exclusions(asg, drop_here)
        params = {"dedup_of": sp_file.stem, "dropped": len(drop_here), "exclusions_file": ex_csv.name,
                  "rule": REASON}
        old_side = sp_file.with_suffix(".json")
        if old_side.exists():
            try:
                params["original_parameters"] = json.loads(old_side.read_text(encoding="utf-8"))["parameters"]
            except (KeyError, ValueError):
                pass
        sid, _ = write_split(recs, new_asg, out_dir, old_name, params, args.manifest)
        new_ids[sp_file.stem] = sid
        kept[sid] = len(new_asg)
    summary = {
        "name": args.name,
        "excluded_total": len(excl),
        "excluded_by_split": dict(Counter(e["split"] for e in excl)),
        "excluded_by_dataset": dict(Counter(e["dataset"] for e in excl)),
        "excluded_by_group": dict(sorted(Counter(e["group_id"] for e in excl).items())),
        "distinct_hashes": len({e["sha256"] for e in excl}),
        "rule": [REASON_DEV, REASON],
        "new_split_ids": new_ids,
        "rows_kept_in_new_splits": kept,
        "hash_cache_hits": cache.hits,
        "hash_cache_misses": cache.misses,
    }
    (out_dir / f"{args.name}.exclusions.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
