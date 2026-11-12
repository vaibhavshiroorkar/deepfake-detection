"""Keep the identities of the Celeb-DF v2 development slice out of the Celeb-DF++ test_x (D-035, D-020, user rule 2026-10-25).

Celeb-DF++ is built on the same 59 celebrities as Celeb-DF v2 (one family, D-012). The 15 identities of the Celeb-DF v2 `dev_x`
slice are used to rank candidates, so no Celeb-DF++ sample that shows one of them may be in test_x. A sample is excluded when its
TARGET identity (the group, as in Celeb-DF v2) or its celebrity SOURCE identity (face-swap and reenactment names carry one; VoxCeleb
speaker ids of the talking-face names have five digits and are not celebrities of this family) is one of the 15.
Excluded samples are not placed in dev_x either (the user's rule is to exclude them); they are listed with the reason and counted.

    python -m dfd.data.identity_exclusion --manifest data/_manifests/celebdf_pp.manifest.csv \
        --reference-manifest data/_manifests/celebdf_v2.manifest.csv --reference-split data/_splits/celebdf_v2-<hash>.csv --name celebdf_pp
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

from .hashing import HashCache
from .manifest import Record, default_data_root, read_manifest
from .splits import read_split, write_split

CELEB_ID = re.compile(r"^id\d{1,3}$")


def identity_of_group(group_id: str) -> str | None:
    last = group_id.split(":")[-1]
    return last if CELEB_ID.match(last) else None


def dev_identities(ref_records: Sequence[Record], ref_assignment: dict[str, str]) -> set[str]:
    return {i for r in ref_records if ref_assignment.get(r.sample_id) == "dev_x" and (i := identity_of_group(r.group_id))}


def celeb_identities(r: Record) -> tuple[set[str], set[str]]:
    """(target identities, celebrity source identities) of a record."""
    target = {i for i in [identity_of_group(r.group_id)] if i}
    source = set()
    if r.notes.startswith("ids="):
        for tok in r.notes[4:].split("|"):
            if CELEB_ID.match(tok) and tok not in target:
                source.add(tok)
    return target, source


def exclude(records: Sequence[Record], dev_ids: set[str]) -> tuple[dict[str, str], list[dict]]:
    """Assignment (every kept sample test_x) and the exclusion rows."""
    asg, rows = {}, []
    for r in records:
        t, s = celeb_identities(r)
        hit_t, hit_s = sorted(t & dev_ids), sorted(s & dev_ids)
        if hit_t or hit_s:
            rows.append({"sample_id": r.sample_id, "split": "test_x", "dataset": r.dataset, "group_id": r.group_id,
                         "method": r.method, "label": r.label, "matched_target": "|".join(hit_t), "matched_source": "|".join(hit_s),
                         "reason": "identity of the Celeb-DF v2 dev_x slice (15 identities) must not be in test_x"})
        else:
            asg[r.sample_id] = "test_x"
    return asg, rows


def drop_duplicates(records: Sequence[Record], assignment: dict[str, str], ref_records: Sequence[Record], ref_assignment: dict[str, str],
                    data_root: Path, cache: HashCache | None = None) -> list[dict]:
    """Samples whose file content (SHA-256) equals a dev_x or test_x sample of the reference dataset: one source video is counted
    once in the family, so the copy of the dataset that is not the reference is dropped (the reference keeps its group id)."""
    cache = cache or HashCache(None)
    ref_hash: dict[str, str] = {}
    for r in ref_records:
        if ref_assignment.get(r.sample_id) in ("dev_x", "test_x"):
            ref_hash.setdefault(cache.sha256(data_root / r.path, r.path), r.sample_id)
    rows = []
    for r in records:
        if r.sample_id in assignment:
            h = cache.sha256(data_root / r.path, r.path)
            if h in ref_hash:
                rows.append({"sample_id": r.sample_id, "split": "test_x", "dataset": r.dataset, "group_id": r.group_id, "method": r.method,
                             "label": r.label, "matched_target": "", "matched_source": "", "sha256": h, "matches": ref_hash[h],
                             "reason": "same file content as a sample of the reference dataset: one source video is counted once in the family"})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--reference-manifest", required=True, type=Path)
    ap.add_argument("--reference-split", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/_splits"))
    ap.add_argument("--data-root", type=Path, default=None)
    a = ap.parse_args(argv)
    recs = read_manifest(a.manifest)
    dev = dev_identities(read_manifest(a.reference_manifest), read_split(a.reference_split))
    if not dev:
        print("the reference split has no dev_x identities", file=sys.stderr)
        return 2
    ref_recs, ref_asg = read_manifest(a.reference_manifest), read_split(a.reference_split)
    asg, rows = exclude(recs, dev)
    n_identity = len(rows)
    dups = drop_duplicates(recs, asg, ref_recs, ref_asg, a.data_root or default_data_root(), HashCache((a.data_root or default_data_root()) / "_manifests" / "content_hash_cache.json"))
    for d in dups:
        asg.pop(d["sample_id"], None)
    rows += dups
    kept = [r for r in recs if r.sample_id in asg]
    by = Counter((x["method"], x["label"]) for x in rows)
    params = {"mode": "fixed test_x with identity exclusion", "dev_identities": sorted(dev), "reference_split": a.reference_split.stem,
              "excluded": len(rows), "excluded_by_identity": n_identity, "excluded_duplicate_content": len(dups),
              "excluded_by_target": sum(1 for x in rows if x["matched_target"]),
              "excluded_only_by_source": sum(1 for x in rows if not x["matched_target"] and x["matched_source"]),
              "exclusions_file": f"{a.name}.exclusions.csv"}
    split_id, csv_path = write_split(kept, asg, a.out_dir, a.name, params, [a.manifest])
    ex_csv = a.out_dir / f"{a.name}.exclusions.csv"
    with open(ex_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, ["sample_id", "split", "dataset", "group_id", "method", "label", "matched_target", "matched_source", "sha256", "matches", "reason"],
                           restval="")
        w.writeheader()
        w.writerows(rows)
    summary = {"split_id": split_id, "rows_in_manifest": len(recs), "kept_test_x": len(kept), **params,
               "excluded_real": sum(1 for x in rows if x["label"] == 0), "excluded_fake": sum(1 for x in rows if x["label"] == 1),
               "kept_real": sum(1 for r in kept if r.label == 0), "kept_fake": sum(1 for r in kept if r.label == 1),
               "kept_groups": len({r.group_id for r in kept}), "excluded_by_method": {f"{m}:{l}": n for (m, l), n in sorted(by.items())}}
    (a.out_dir / f"{a.name}.exclusions.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
