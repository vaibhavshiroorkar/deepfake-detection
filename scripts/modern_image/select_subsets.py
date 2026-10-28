"""Choose the Community Forensics shards to download, by fixed rules (W1-I, E-028 and E-029). Reads only the metadata index.

Usage: python scripts/modern_image/select_subsets.py
Writes data/cf_eval/_meta/subset.json and data/cf_small/_meta/subset.json (shards, LFS SHA-256 from the Hub, expected rows, rule text).

Eval rule (E-028, test only): a cell is (model_name, label, real_source). Fake cells and non-RAISE real cells get a target of
TARGET images each; the RAISE reals (about 20 MB per TIFF image, 8,000 images in 8 per-generator pairings) are pooled into one cell
"RAISE" with target RAISE_TARGET, filled from at least two different pairings. Shards are added greedily by useful rows per byte
(useful = sum over cells of min(remaining need, rows of that cell in the shard)), ties broken by shard name, until every cell is at
target or has no remaining rows. Every row of a downloaded shard is kept (NSFW-flagged rows are dropped at extraction; Eval has none).

Small rule (E-029, training and validation only): fake shards = LatDiff shards 0 and 64 (first and last of the 66 Systematic
shards) plus the Manual shards 66 to 83 of at most 0.65 GB; real shards = COCO 117 and 118, VISION and COCO 116, FFHQ (train split) 94.
The measured download rate on 2026-10-27 (about 3 to 4 MB/s in total) sets the size: Eval targets 200 per cell and 120 pooled RAISE
reals, Small about 11 GB (decided autonomously - needs user review). NSFW-flagged rows are dropped; at most CAP images per (model_name, real_source) enter training.
"""
import csv, json, collections, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import _env

TARGET, RAISE_TARGET, CAP = 200, 120, 1000
C = collections.Counter


def eval_subset():
    rows = list(csv.DictReader(open(_env.CF_EVAL / "_meta" / "index.csv", encoding="utf-8")))
    js = json.load(open(_env.CF_EVAL / "_meta" / "shards.json"))
    size = {k: v["size"] for k, v in js["shards"].items()}

    def cell(r):
        if r["label"] == "0" and r["real_source"] == "RAISE":
            return ("RAISE", "0", "RAISE")
        return (r["model_name"], r["label"], r["real_source"])
    sh = collections.defaultdict(C)
    raise_pair = collections.defaultdict(C)
    for r in rows:
        sh[r["shard"]][cell(r)] += 1
        if cell(r)[0] == "RAISE":
            raise_pair[r["shard"]][r["model_name"]] += 1
    avail = C()
    for v in sh.values():
        avail.update(v)
    need = {k: min(RAISE_TARGET if k[0] == "RAISE" else TARGET, n) for k, n in avail.items()}
    chosen, raise_from = [], set()
    while True:
        best, bs = None, 0.0
        for s in sorted(sh):
            if s in chosen:
                continue
            u = 0
            for k, n in sh[s].items():
                if k[0] == "RAISE" and raise_pair[s] and len(raise_from) == 1 and set(raise_pair[s]) <= raise_from \
                        and need[k] <= RAISE_TARGET // 2:
                    continue  # second half of the RAISE target must come from another pairing
                u += min(need[k], n)
            sc = u / size[s]
            if u and sc > bs:
                best, bs = s, sc
        if best is None:
            break
        chosen.append(best)
        for k, n in sh[best].items():
            need[k] = max(0, need[k] - n)
        raise_from |= set(raise_pair[best])
    got = C()
    for s in chosen:
        got.update(sh[s])
    return {"repo": _env.EVAL_REPO, "revision": _env.EVAL_REV, "rule": __doc__.split("Small rule")[0].split("Eval rule")[1],
            "target": TARGET, "raise_target": RAISE_TARGET,
            "shards": [{"file": s, "sha256": js["shards"][s]["sha256"], "size": size[s], "rows": js["shards"][s]["rows"]} for s in sorted(chosen)],
            "total_bytes": sum(size[s] for s in chosen),
            "cells": {"|".join(k): {"selected": got.get(k, 0), "available": avail[k]} for k in sorted(avail)},
            "raise_pairings_used": sorted(raise_from)}


def small_subset():
    js = json.load(open(_env.CF_SMALL / "_meta" / "shards.json"))
    size = {k: v["size"] for k, v in js["shards"].items()}
    ids = [0, 64] + [i for i in range(66, 84) if size[f"data/HFCF_small_{i}.parquet"] <= 0.65e9] + [94, 116, 117, 118]
    files = [f"data/HFCF_small_{i}.parquet" for i in ids]
    return {"repo": _env.SMALL_REPO, "revision": _env.SMALL_REV, "cap_per_generator": CAP, "rule": "Small rule" + __doc__.split("Small rule")[1],
            "shards": [{"file": s, "sha256": js["shards"][s]["sha256"], "size": size[s], "rows": js["shards"][s]["rows"]} for s in files],
            "total_bytes": sum(size[s] for s in files)}


if __name__ == "__main__":
    e = eval_subset()
    json.dump(e, open(_env.CF_EVAL / "_meta" / "subset.json", "w"), indent=1)
    print("eval shards", len(e["shards"]), "GB", round(e["total_bytes"] / 1e9, 2), "raise pairings", e["raise_pairings_used"])
    for k, v in e["cells"].items():
        print("  ", k, v)
    s = small_subset()
    json.dump(s, open(_env.CF_SMALL / "_meta" / "subset.json", "w"), indent=1)
    print("small shards", len(s["shards"]), "GB", round(s["total_bytes"] / 1e9, 2), [x["file"].split("_")[-1] for x in s["shards"]])
