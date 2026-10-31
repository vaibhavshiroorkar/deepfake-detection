"""Frozen sample manifests and evaluation groups for the post-freeze evaluations E-033 (LAV-DF) and E-034 (Community Forensics-Eval).

Protocol: docs/POST_FREEZE_EVALUATION_PROTOCOL.md and the E-033 / E-034 blocks of docs/EXPERIMENTS.md.
Reads metadata only (no media, no score). Writes new files only, under data/_manifests/posteval/; refuses to overwrite a frozen manifest.

    python scripts/posteval/build_manifests.py e033
    python scripts/posteval/build_manifests.py e034-select     # writes the shard list to download (before any download)
    python scripts/posteval/build_manifests.py e034            # after the download: manifest and groups of the downloaded rows
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = DATA / "_manifests" / "posteval"
SEED = 20261031
WINDOW_S = 64600 / 16000  # the speech head scores the first 64,600 samples at 16 kHz (after trimming)

# E-033 nested group sizes, counted in source groups (a real original and every fake made from it)
E033_SIZES = [("G1", "Small", 60), ("G2", "Medium", 300), ("G3", "Large", 1200), ("G4", "Very large", 3000)]
# E-034 nested group sizes, counted in images per class (stratified by label)
E034_SIZES = [("G1", "Small", 100), ("G2", "Medium", 400), ("G3", "Large", 1200)]
E034_BUDGET = 5.0e9  # bytes of shards to download


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def write_once(path: Path, text: str) -> str:
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return sha256_file(path)
        raise SystemExit(f"refused: {path} exists with different content (frozen manifests are never overwritten)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return sha256_file(path)


def csv_text(cols, rows) -> str:
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({c: r[c] for c in cols})
    return buf.getvalue()


def group_entry(gid, title, kind, desc, rows, label_key, overlap, unit):
    y = [int(r[label_key]) for r in rows]
    return {"id": gid, "title": title, "kind": kind, "description": desc, "overlap": overlap,
            "n_samples": len(rows), "n_fake": sum(y), "n_real": len(y) - sum(y),
            "n_groups": len({r["group_id"] for r in rows}), "group_unit": unit,
            "sample_ids": sorted(r["sample_id"] for r in rows)}


# ---------------------------------------------------------------- E-033 LAV-DF
def e033() -> None:
    meta = json.load(open(DATA / "lavdf" / "LAV-DF" / "metadata.min.json", encoding="utf-8"))
    byf = {m["file"]: m for m in meta}
    src = lambda m: m["original"] or m["file"]  # noqa: E731 - the real original groups its fakes
    test = [m for m in meta if m["split"] == "test"]
    sealed = sorted(r["sample_id"].split(":", 1)[1] for r in csv.DictReader(open(DATA / "_splits" / "lavdf_subset-e9d12decaf1f.csv"))
                    if r["split"] == "test_in")
    sealed_set = set(sealed)
    sealed_src = {src(byf[f]) for f in sealed}
    rows, excl = [], []
    for m in sorted(test, key=lambda m: m["file"]):
        if m["file"] in sealed_set:
            excl.append({"file": m["file"], "reason": "in the sealed LAV-DF subset (lavdf_subset-e9d12decaf1f test_in, ledger I-5 and I-6)"})
            continue
        if src(m) in sealed_src:
            excl.append({"file": m["file"], "reason": "shares its source video with a sealed sample"})
            continue
        starts = sorted(a for a, _ in m["fake_periods"])
        first = starts[0] if starts else ""
        cell = {(False, False): "real", (True, False): "video_fake", (False, True): "audio_fake", (True, True): "both_fake"}[(m["modify_video"], m["modify_audio"])]
        rows.append({"sample_id": "lavdf:" + m["file"], "path": "lavdf/LAV-DF/" + m["file"], "group_id": "lavdf:" + src(m), "cell": cell,
                     "label_visual": int(m["modify_video"]), "label_audio": int(m["modify_audio"]),
                     "first_fake_start_s": first, "duration_s": m["duration"],
                     "audio_fake_in_window": int(bool(m["modify_audio"]) and first != "" and first < WINDOW_S)})
    groups = sorted({r["group_id"] for r in rows})
    order = groups[:]
    random.Random(SEED).shuffle(order)
    rank = {g: i for i, g in enumerate(order)}
    for r in rows:
        r["group_rank"] = rank[r["group_id"]]
    cols = ["sample_id", "path", "group_id", "group_rank", "cell", "label_visual", "label_audio", "first_fake_start_s", "duration_s", "audio_fake_in_window"]
    man_hash = write_once(OUT / "e033_lavdf.manifest.csv", csv_text(cols, rows))
    ex_hash = write_once(OUT / "e033_lavdf.exclusions.csv", csv_text(["file", "reason"], excl))

    def nested(k):
        return [r for r in rows if r["group_rank"] < k]

    datasets = []
    for head, label_key, modality, ds_key, ds_title in (("T-V", "label_visual", "visual", "lavdf_unread_tv", "LAV-DF test, unread part: face-video head (T-V)"),
                                                        ("T-A", "label_audio", "audio", "lavdf_unread_ta", "LAV-DF test, unread part: speech head (T-A)")):
        gs = []
        for gid, title, k in E033_SIZES:
            sub = nested(k)
            gs.append(group_entry(gid, f"{title} sample", "general",
                                  f"The first {k} source groups of the seeded order (seed {SEED}); every real original comes with all fakes made from it.",
                                  sub, label_key, "nested: contains every smaller group", "source video"))
        g3 = nested(dict((g, k) for g, _, k in E033_SIZES)["G3"])
        if head == "T-V":
            sub = [r for r in g3 if r["cell"] != "real"]
            desc = ("Diagnostic: the Large group without the real originals. Positives: video modified (video_fake, both_fake); negatives: audio_fake "
                    "(video untouched but re-produced by the dataset pipeline). Tests whether the head responds to the visual edit or to the production pipeline.")
            title = "Diagnostic: produced videos only"
        else:
            sub = [r for r in g3 if r["label_audio"] == 0 or r["audio_fake_in_window"] == 1]
            desc = (f"Diagnostic: the Large group with audio fakes kept only when the first fake segment starts before {WINDOW_S:.4f} s, the length of the "
                    "window the speech head scores (start measured on the untrimmed track, so the overlap is approximate). Negatives: every clip with "
                    "untouched audio. Tests how much of the error comes from edits outside the scored window.")
            title = "Diagnostic: audio edit inside the scored window"
        gs.append(group_entry("G5", title, "diagnostic", desc, sub, label_key, "subset of G3", "source video"))
        full = group_entry("FULL", "Full eligible pool", "full", "Every eligible clip (explicit action only; not a default).", rows, label_key,
                           "contains every group", "source video")
        datasets.append({"dataset": ds_key, "title": ds_title, "experiment": "E-033", "modality": modality, "head": head,
                         "label_column": label_key, "split": "post_x", "groups": gs, "full": full})
    payload = {"experiment": "E-033", "seed": SEED, "manifest": "e033_lavdf.manifest.csv", "manifest_sha256": man_hash,
               "exclusions": "e033_lavdf.exclusions.csv", "exclusions_sha256": ex_hash,
               "source": "data/lavdf/LAV-DF/metadata.min.json (official split field == test)",
               "counts": {"test_split": len(test), "excluded": len(excl), "eligible": len(rows), "source_groups": len(groups),
                          "cells": dict(Counter(r["cell"] for r in rows)), "exclusion_reasons": dict(Counter(e["reason"] for e in excl))},
               "datasets": datasets}
    h = write_once(OUT / "e033_lavdf.groups.json", json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps(payload["counts"], indent=1))
    for d in datasets:
        for g in d["groups"] + [d["full"]]:
            print(d["head"], g["id"], g["n_samples"], "fake", g["n_fake"], "real", g["n_real"], "groups", g["n_groups"])
    print("groups.json sha256", h)


# ---------------------------------------------------------------- E-034 Community Forensics-Eval
def _cf_index():
    base = DATA / "cf_eval" / "_meta"
    shards = json.load(open(base / "shards.json"))["shards"]
    used = {s["file"] for s in json.load(open(base / "subset.json"))["shards"]}
    rows = list(csv.DictReader(open(base / "index.csv", encoding="utf-8")))
    by = defaultdict(list)
    for r in rows:
        by[r["shard"]].append(r)
    return shards, used, by


def e034_select() -> None:
    shards, used, by = _cf_index()
    pool = sorted(k for k in shards if k not in used and not any(r["real_source"] == "RAISE" for r in by[k]))
    order = pool[:]
    random.Random(SEED).shuffle(order)
    sel, total = [], 0
    for k in order:
        if total >= E034_BUDGET:
            break
        sel.append(k)
        total += shards[k]["size"]
    payload = {"experiment": "E-034", "repo": "OwensLab/CommunityForensics-Eval", "revision": "7d4a74a88d2cac93b513c0853bf92c260eaceea0",
               "rule": (f"Pool: the shards of the Eval set that are NOT in the E-028 subset (data/cf_eval/_meta/subset.json) and hold no RAISE-paired row "
                        f"(RAISE reals are about 20 MB TIFF files each and were pooled in E-028). Sorted by name, shuffled with random.Random({SEED}), taken "
                        f"in that order until at least {E034_BUDGET / 1e9:.1f} GB. Chosen from metadata only, before any image was downloaded."),
               "pool_shards": len(pool), "shards": [{"file": k, **shards[k]} for k in sel], "total_bytes": total,
               "expected_rows": dict(Counter(r["label"] for k in sel for r in by[k]))}
    h = write_once(OUT / "e034_cfeval.shards.json", json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(len(sel), "shards", round(total / 1e9, 2), "GB", payload["expected_rows"], "sha256", h)


def e034() -> None:
    shards, used, by = _cf_index()
    sel = json.load(open(OUT / "e034_cfeval.shards.json"))
    log = DATA / "cf_eval_e034" / "_meta" / "download_log.jsonl"
    ok = {json.loads(l)["file"] for l in open(log) if l.strip() and json.loads(l)["match"]} if log.exists() else set()
    missing = [s["file"] for s in sel["shards"] if s["file"] not in ok]
    if missing:
        raise SystemExit(f"refused: {len(missing)} selected shards are not downloaded and verified")
    rows = []
    for s in sel["shards"]:
        for r in by[s["file"]]:
            assert r["shard"] not in used
            rows.append({"sample_id": f"cfeval:{Path(r['shard']).stem}:{r['row']}", "shard": r["shard"], "row": r["row"], "group_id": f"cfeval:{Path(r['shard']).stem}:{r['row']}",
                         "label": int(r["label"]), "generator": r["model_name"] if r["label"] == "1" else "", "real_source": r["real_source"] if r["label"] == "0" else "",
                         "architecture": r["architecture"], "format": r["format"], "resolution": r["resolution"], "image_name": r["image_name"]})
    rows.sort(key=lambda r: r["sample_id"])
    rng = random.Random(SEED)
    for lab in (0, 1):
        ids = [r for r in rows if r["label"] == lab]
        rng.shuffle(ids)
        for i, r in enumerate(ids):
            r["class_rank"] = i
    cols = ["sample_id", "shard", "row", "group_id", "label", "generator", "real_source", "architecture", "format", "resolution", "image_name", "class_rank"]
    man_hash = write_once(OUT / "e034_cfeval.manifest.csv", csv_text(cols, rows))
    gs = []
    for gid, title, k in E034_SIZES:
        sub = [r for r in rows if r["class_rank"] < k]
        gs.append(group_entry(gid, f"{title} sample", "general",
                              f"The first {k} real and the first {k} generated images of the seeded order (seed {SEED}, separate order per class).",
                              sub, "label", "nested: contains every smaller group", "image"))
    gs.append(group_entry("G4", "Very large sample", "general", "Every downloaded image (classes not balanced; the share is shown).",
                          rows, "label", "contains every group", "image"))
    png_real = [r for r in rows if r["label"] == 0 and r["format"] == "PNG"]
    png_fake = sorted([r for r in rows if r["label"] == 1 and r["format"] == "PNG"], key=lambda r: r["class_rank"])[: len(png_real)]
    gs.append(group_entry("G5", "Diagnostic: PNG only, format-matched", "diagnostic",
                          "Every real PNG image and the same number of generated PNG images (first in the seeded order). A file-format cue cannot separate "
                          "this group (E-028 found that format alone separates the full set with AUC 0.93).", png_real + png_fake, "label",
                          "overlaps G1 to G4", "image"))
    datasets = [{"dataset": "cfeval_unread_ti", "title": "Community Forensics-Eval, unread shards: generated-image head (T-I)", "experiment": "E-034",
                 "modality": "image", "head": "T-I", "label_column": "label", "split": "post_x", "groups": gs, "full": None}]
    payload = {"experiment": "E-034", "seed": SEED, "manifest": "e034_cfeval.manifest.csv", "manifest_sha256": man_hash,
               "shards": "e034_cfeval.shards.json", "shards_sha256": sha256_file(OUT / "e034_cfeval.shards.json"),
               "counts": {"rows": len(rows), "by_label": dict(Counter(r["label"] for r in rows)),
                          "generators": dict(Counter(r["generator"] for r in rows if r["label"] == 1)),
                          "real_sources": dict(Counter(r["real_source"] for r in rows if r["label"] == 0)),
                          "format_by_label": {f"{l}:{f}": n for (l, f), n in Counter((r["label"], r["format"]) for r in rows).items()}},
               "datasets": datasets}
    h = write_once(OUT / "e034_cfeval.groups.json", json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps(payload["counts"], indent=1))
    for g in gs:
        print(g["id"], g["n_samples"], "fake", g["n_fake"], "real", g["n_real"])
    print("groups.json sha256", h)


if __name__ == "__main__":
    {"e033": e033, "e034-select": e034_select, "e034": e034}[sys.argv[1]]()
