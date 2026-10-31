"""Post-freeze evaluations E-033 and E-034: duplicate checks, Lab score files and descriptors, results (protocol: docs/EXPERIMENTS.md).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/posteval/evaluate.py dupcheck e033|e034
    PYTHONPATH=src .venv/Scripts/python.exe scripts/posteval/evaluate.py export e033|e034
    PYTHONPATH=src .venv/Scripts/python.exe scripts/posteval/evaluate.py results e033|e034

`export` writes the score files (split post_x, dataset keys of the groups file, temperature-scaled scores of the frozen checkpoint) and one
`*.lab.json` descriptor per head under data/_evallab/imports/posteval/, so the Evaluation Lab offers the predeclared groups.
`results` computes every group through the Lab itself (dfd.platform.evallab, synchronous runner, records in data/_evallab/runs), so the
documents and the Lab show the same numbers, plus the predeclared secondary analyses; it refuses to overwrite results.json.
Nothing under data/_final, ledger/, checkpoints/ or src/ is written; no threshold is chosen here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
POST = DATA / "_manifests" / "posteval"
RES = DATA / "_results" / "posteval"
IMPORTS = DATA / "_evallab" / "imports" / "posteval"
CKPT = ROOT / "checkpoints" / "I-5-freeze-candidate"
SEED = 20261031

CFG = {
    "e033": {"exp": "E-033", "groups": POST / "e033_lavdf.groups.json", "manifest": POST / "e033_lavdf.manifest.csv",
             "heads": {"lavdf_unread_tv": ("T-V", "visual", "label_visual", "E-033_TV"),
                       "lavdf_unread_ta": ("T-A", "audio", "label_audio", "E-033_TA")}},
    "e034": {"exp": "E-034", "groups": POST / "e034_cfeval.groups.json", "manifest": POST / "e034_cfeval.manifest.csv",
             "heads": {"cfeval_unread_ti": ("T-I", "image", "label", "E-034_TI")}},
}
SCORE_KIND = ("temperature-scaled score of the frozen checkpoint (temperature fitted in-domain on the training dataset's validation split); "
              "on this dataset it is a ranking score, not a calibrated probability")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def read_csv(p: Path) -> list[dict]:
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_new_json(p: Path, obj) -> None:
    if p.exists():
        sys.exit(f"refused: {p} exists; results are never overwritten")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def meta_threshold(head: str) -> dict:
    meta = json.load(open(CKPT / "metadata.json"))
    t = meta["tasks"][head]
    return {"value": float(t["threshold"]["value"]), "temperature": float(t["calibration"]["temperature"]),
            "provenance": (f"frozen checkpoint {meta.get('checkpoint_id', 'I-5-freeze-candidate')}, task {head}: maximum balanced accuracy "
                           f"on the validation split of its training dataset (D-024), read from metadata.json; not tuned on this data")}


# ------------------------------------------------------------------ duplicate checks
def dupcheck(key: str) -> None:
    out = RES / CFG[key]["exp"] / "duplicates.json"
    if key == "e033":
        pool = read_csv(CFG[key]["manifest"])
        sealed = [r["sample_id"].split(":", 1)[1] for r in read_csv(DATA / "_splits" / "lavdf_subset-e9d12decaf1f.csv")]
        head = [r["path"] for r in read_csv(DATA / "_manifests" / "lavdf_headtrain.manifest.csv")]
        ref = {"lavdf/LAV-DF/" + f for f in sealed} | set(head)
        ref_hash = {}
        for k, rel in enumerate(sorted(ref)):
            ref_hash[sha256_file(DATA / rel)] = rel
        pool_hash, dup_ref, dup_within = {}, [], []
        for r in pool:
            h = sha256_file(DATA / r["path"])
            if h in ref_hash:
                dup_ref.append({"sample_id": r["sample_id"], "matches": ref_hash[h]})
            if h in pool_hash:
                dup_within.append({"sample_id": r["sample_id"], "matches": pool_hash[h]})
            pool_hash.setdefault(h, r["sample_id"])
        res = {"experiment": "E-033", "pool_files": len(pool), "reference_files": len(ref),
               "reference": "sealed lavdf_subset (all three splits: train, val, test_in) and lavdf_headtrain",
               "exact_duplicates_of_reference": dup_ref, "exact_duplicates_within_pool": dup_within,
               "excluded": sorted({d["sample_id"] for d in dup_ref}),
               "near_duplicates": "not checked (no perceptual hashing); fakes share a group with their source by construction"}
    else:
        sys.path.insert(0, str(ROOT / "scripts" / "posteval"))
        from score_e034 import iter_shard
        pred = read_csv(RES / "E-034" / "predictions.csv")
        ref = {}
        e028 = json.load(open(DATA / "cf_eval" / "_meta" / "subset.json"))["shards"]
        for s in e028:
            for row, d in iter_shard(DATA / "cf_eval" / "shards" / s["file"]):
                ref[hashlib.sha256(d["image_data"]).hexdigest()] = f"E-028 {s['file']}:{row}"
        n028 = len(ref)
        for m in ("forensynths_progan_val", "forensynths_progan_test"):
            for r in read_csv(DATA / "_manifests" / f"{m}.manifest.csv"):
                ref[sha256_file(DATA / r["path"])] = r["sample_id"]
        seen, dup_ref, dup_within = {}, [], []
        for r in pred:
            h = r["sha256"]
            if h in ref:
                dup_ref.append({"sample_id": r["sample_id"], "matches": ref[h]})
            if h in seen:
                dup_within.append({"sample_id": r["sample_id"], "matches": seen[h]})
            seen.setdefault(h, r["sample_id"])
        idx = read_csv(DATA / "cf_eval" / "_meta" / "index.csv")
        e028_files = {s["file"] for s in e028}
        e028_real_names = {r["image_name"] for r in idx if r["shard"] in e028_files and r["label"] == "0"}
        man = read_csv(CFG[key]["manifest"])
        name_overlap = sorted(r["sample_id"] for r in man if r["label"] == "0" and r["image_name"] in e028_real_names)
        res = {"experiment": "E-034", "pool_images": len(pred), "reference_images": {"e028": n028, "progan_val_and_test": len(ref) - n028},
               "exact_duplicates_of_reference": dup_ref, "exact_duplicates_within_pool": dup_within,
               "excluded": sorted({d["sample_id"] for d in dup_ref} | {d["sample_id"] for d in dup_within}),
               "real_image_name_overlap_with_e028_reals": name_overlap,
               "near_duplicates": "not checked (no perceptual hashing)"}
    write_new_json(out, res)
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in res.items()}, indent=1))


# ------------------------------------------------------------------ export to the lab
def export(key: str) -> None:
    c = CFG[key]
    exp = c["exp"]
    dup = json.load(open(RES / exp / "duplicates.json"))
    excluded = set(dup["excluded"])
    man = {r["sample_id"]: r for r in read_csv(c["manifest"])}
    pred = {r["sample_id"]: r for r in read_csv(RES / exp / "predictions.csv")}
    missing = sorted(set(man) - set(pred))
    if missing:
        sys.exit(f"refused: {len(missing)} manifest samples have no prediction row")
    gsha = sha256_file(c["groups"])
    prereg = json.load(open(RES / exp / "prereg.json"))
    IMPORTS.mkdir(parents=True, exist_ok=True)
    summary = {}
    for ds, (head, modality, label_key, stem) in c["heads"].items():
        thr = meta_threshold(head)
        rows, unscored = [], Counter()
        for sid, m in man.items():
            p = pred[sid]
            if sid in excluded:
                unscored["exact duplicate (excluded)"] += 1
                continue
            if key == "e034":
                ok, score, sha = p["ok"] == "1", p["score"], p["sha256"]
                why = p["error"] or "not scored"
            else:
                col = "visual" if head == "T-V" else "audio"
                ok, score, sha = p[f"{col}_available"] == "1" and p[f"{col}_score"] != "", p[f"{col}_score"], ""
                why = (p["error"] or ("no face found" if head == "T-V" else "no audio") if not ok else "")
            if not ok:
                unscored[why[:60]] += 1
                continue
            rows.append({"run_id": f"{exp}-{head.replace('-', '')}", "iteration": exp, "seed": 0, "modality": modality, "dataset": ds,
                         "split": "post_x", "sample_id": sid, "group_id": m["group_id"], "label": int(m[label_key]),
                         "score": float(score), **({"sha256": sha} if sha else {}),
                         **({"generator": m["generator"] or "(real)", "source": m["real_source"] or "(generated)"} if key == "e034" else {})})
        cols = ["run_id", "iteration", "seed", "modality", "dataset", "split", "sample_id", "group_id", "label", "score"]
        cols += [k for k in ("sha256", "generator", "source") if k in rows[0]]
        sp = IMPORTS / f"{stem}.csv"
        if sp.exists():
            sys.exit(f"refused: {sp} exists")
        with open(sp, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        doc = json.load(open(c["groups"]))
        dsd = next(d for d in doc["datasets"] if d["dataset"] == ds)
        caveats = {
            "e033": ["LAV-DF behaviour under these heads was already seen on LAV-DF dev and on the sealed 2,000-clip subset: this is a larger "
                     "replication on unread clips of a seen distribution, not an independent dataset-level test.",
                     "Identities are not in the LAV-DF metadata: the bootstrap resamples source videos, so intervals understate the "
                     "uncertainty from the 31 test identities.",
                     "LAV-DF fakes are short word-level edits made by one pipeline; the speech head scores only about 4 s."],
            "e034": ["Same evaluation set as E-028 (other shards): a replication on unread images of a seen distribution.",
                     "12 of the 21 generators (RAISE-paired generators excluded by download cost); reals and fakes differ in file "
                     "format and size, which the diagnostic group controls only for format.",
                     "Images are the bootstrap units; rows of one shard may be correlated, so intervals may be optimistic."],
        }[key]
        desc = {"schema": "evallab-groups/1", "dataset": ds, "title": dsd["title"], "experiment": exp, "iteration": exp,
                "modality": modality, "head": head, "split": "post_x", "run_ids": [rows[0]["run_id"]], "score_file": sp.name,
                "groups_file": c["groups"].relative_to(ROOT).as_posix(), "groups_sha256": gsha, "bootstrap_seed": SEED,
                "threshold": {"value": thr["value"], "provenance": thr["provenance"]}, "score_kind": SCORE_KIND,
                "eligibility": ("eligible: never used to train, tune, calibrate or select the frozen checkpoint; samples disjoint from every "
                                "earlier read of this dataset (see docs/EXPERIMENTS.md " + exp + ")"),
                "caveats": caveats,
                "provenance": {"experiment": exp, "prereg_sha256": prereg["block_sha256"], "prereg_written_at": prereg["written_at"],
                               "manifest": c["manifest"].relative_to(ROOT).as_posix(), "manifest_sha256": sha256_file(c["manifest"]),
                               "predictions_sha256": sha256_file(RES / exp / "predictions.csv"), "score_file_sha256": sha256_file(sp),
                               "checkpoint_id": "I-5-freeze-candidate-20261025-1608",
                               "checkpoint_metadata_sha256": sha256_file(CKPT / "metadata.json"), "temperature": thr["temperature"]}}
        lp = IMPORTS / f"{stem}.lab.json"
        if lp.exists():
            sys.exit(f"refused: {lp} exists")
        lp.write_text(json.dumps(desc, indent=1) + "\n", encoding="utf-8")
        summary[ds] = {"rows": len(rows), "unscored": dict(unscored), "score_file_sha256": desc["provenance"]["score_file_sha256"]}
    print(json.dumps(summary, indent=1))


# ------------------------------------------------------------------ results through the lab
def _ci_auc(y, s, groups):
    from dfd.eval.bootstrap import cluster_bootstrap
    r = cluster_bootstrap(np.asarray(y), np.asarray(s, float), np.asarray(groups), metric="auc", n_boot=2000, seed=SEED)
    return {"estimate": r.estimate, "low": r.low, "high": r.high}


def results(key: str) -> None:
    from dfd.platform.evallab import EvalLab
    c = CFG[key]
    exp = c["exp"]
    lab = EvalLab(synchronous=True)
    listing = {d["dataset"]: d for d in lab.groups.listing()["datasets"]}
    out = {"experiment": exp, "seed": SEED, "n_boot": 2000, "datasets": {}}
    for ds in c["heads"]:
        if ds not in listing:
            sys.exit(f"refused: the lab does not list {ds}")
        d = listing[ds]
        gids = [g["id"] for g in d["groups"]] + (["FULL"] if d["full"] else [])
        res = {}
        for gid in gids:
            ent, _ = lab.groups.resolve(ds, gid)
            dd = ent["desc"]
            rec = lab.runner.submit({"score_file_id": ent["score_file_id"], "iteration": dd["iteration"], "modality": dd["modality"],
                                     "dataset": ds, "split": "post_x", "run_ids": dd["run_ids"], "threshold_source": "fixed",
                                     "threshold": dd["threshold"]["value"], "threshold_provenance": dd["threshold"]["provenance"],
                                     "purpose": "post_freeze_evaluation", "eval_group": gid, "n_boot": 2000, "seed": SEED,
                                     "label": f"{exp} {ds} {gid} (results for docs)"})
            if rec["state"] != "done":
                sys.exit(f"{ds} {gid}: {rec['state']} {rec.get('error')}")
            res[gid] = {"lab_run_id": rec["id"], "config_sha256": rec["config_sha256"], "data": {k: rec["data"][k] for k in
                        ("n_samples", "n_fake", "n_real", "n_groups", "class_balance", "group_missing_scores", "warnings")},
                        "metrics": {k: {q: v[q] for q in ("estimate", "low", "high")} for k, v in rec["metrics"].items()},
                        "confusion": {k: rec["confusion"][k] for k in ("tp", "fp", "tn", "fn")}, "threshold": rec["threshold"]["value"]}
            print(ds, gid, res[gid]["data"]["n_samples"], "AUROC", round(res[gid]["metrics"]["auc"]["estimate"], 4), flush=True)
        out["datasets"][ds] = {"groups": res, "threshold": d["threshold"], "score_file": d["score_file"]}
    out["secondary"] = secondary(key)
    out["hypotheses"] = verdicts(key, out)
    write_new_json(RES / exp / "results.json", out)
    print(json.dumps(out["hypotheses"], indent=1))
    print(json.dumps(out["secondary"], indent=1)[:3000])


def _scores(stem):
    return read_csv(IMPORTS / f"{stem}.csv")


def secondary(key: str) -> dict:
    if key == "e033":
        man = {r["sample_id"]: r for r in read_csv(CFG[key]["manifest"])}
        out = {}
        for ds, (head, _, _, stem) in CFG[key]["heads"].items():
            rows = _scores(stem)
            y = [int(man[r["sample_id"]]["label_visual"] == "1" or man[r["sample_id"]]["label_audio"] == "1") for r in rows]
            out[f"{ds}_FULL_any_modality_auroc"] = {**_ci_auc(y, [float(r["score"]) for r in rows], [r["group_id"] for r in rows]),
                                                   "n_fake": sum(y), "n_real": len(y) - sum(y),
                                                   "label": "modify_video OR modify_audio (single head, not fusion)"}
        return out
    from sklearn.metrics import average_precision_score
    rows = _scores("E-034_TI")
    g4 = set(next(g for g in json.load(open(CFG[key]["groups"]))["datasets"][0]["groups"] if g["id"] == "G4")["sample_ids"])
    rows = [r for r in rows if r["sample_id"] in g4]
    thr = meta_threshold("T-I")["value"]
    reals = [float(r["score"]) for r in rows if r["label"] == "0"]
    per = {}
    by = defaultdict(list)
    for r in rows:
        if r["label"] == "1":
            by[r["generator"]].append(float(r["score"]))
    for gname, fs in sorted(by.items()):
        y = [1] * len(fs) + [0] * len(reals)
        s = fs + reals
        pred = [int(v >= thr) for v in s]
        per[gname] = {"n_fake": len(fs), "n_real": len(reals), "ap": float(average_precision_score(y, s)),
                      "accuracy": float(np.mean([p == t for p, t in zip(pred, y)])),
                      "balanced_accuracy": float(0.5 * (np.mean([v >= thr for v in fs]) + np.mean([v < thr for v in reals])))}
    return {"per_generator_vs_all_g4_reals": per, "mean_ap": float(np.mean([v["ap"] for v in per.values()])),
            "mean_accuracy": float(np.mean([v["accuracy"] for v in per.values()])),
            "mean_balanced_accuracy": float(np.mean([v["balanced_accuracy"] for v in per.values()])), "n_generators": len(per),
            "note": "each generator's fakes against all G4 reals at the frozen threshold; the published Community Forensics values pair "
                    "each generator with its own reals and do not state their accuracy threshold (Tier B at best)"}


def verdicts(key: str, out: dict) -> dict:
    def line(ci, above=True):
        lo, hi = ci["low"], ci["high"]
        if above:
            return "SUPPORTED" if lo > 0.5 else ("CONTRADICTED" if hi < 0.5 else "NOT SUPPORTED")
        return "SUPPORTED" if hi < 0.5 else ("CONTRADICTED" if lo > 0.5 else "NOT SUPPORTED")
    if key == "e033":
        tv, ta = out["datasets"]["lavdf_unread_tv"]["groups"], out["datasets"]["lavdf_unread_ta"]["groups"]
        a = lambda g, gid: g[gid]["metrics"]["auc"]["estimate"]  # noqa: E731
        return {"H-33a (T-V G4 AUROC upper bound below 0.50)": line(tv["G4"]["metrics"]["auc"], above=False),
                "H-33b (T-A G4 AUROC lower bound above 0.50)": line(ta["G4"]["metrics"]["auc"]),
                "H-33c (T-A G5 above G3, descriptive)": "OBSERVED" if a(ta, "G5") > a(ta, "G3") else "NOT OBSERVED",
                "H-33d (T-V G5 closer to 0.50 than G3, descriptive)": ("OBSERVED" if abs(a(tv, "G5") - 0.5) < abs(a(tv, "G3") - 0.5)
                                                                       else "NOT OBSERVED")}
    ti = out["datasets"]["cfeval_unread_ti"]["groups"]
    fnr = ti["G4"]["metrics"]["fnr"]
    return {"H-34a (T-I G4 AUROC lower bound above 0.50)": line(ti["G4"]["metrics"]["auc"]),
            "H-34b (T-I G4 FNR lower bound above 0.50)": "SUPPORTED" if fnr["low"] > 0.5 else "NOT SUPPORTED",
            "H-34c (G5 PNG-only AUROC below G4, descriptive)": ("OBSERVED" if ti["G5"]["metrics"]["auc"]["estimate"] < ti["G4"]["metrics"]["auc"]["estimate"]
                                                                else "NOT OBSERVED")}


if __name__ == "__main__":
    cmd, key = sys.argv[1], sys.argv[2]
    if not (RES / CFG[key]["exp"] / "prereg.json").exists():
        sys.exit("refused: preregistration record missing")
    {"dupcheck": dupcheck, "export": export, "results": results}[cmd](key)
