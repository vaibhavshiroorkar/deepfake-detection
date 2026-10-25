"""E-019 v2 re-audit on canonical-v2 copies (see docs/EXPERIMENTS.md, E-019 v2 addendum). Reuses scripts/canonical/reaudit.py
(and through it the E-018 functions) by import; neither is modified. DF26: descriptive only, no classifier fitted.

Run: PYTHONPATH=src CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/canonical/reaudit_v2.py
"""
import csv
import json
import platform
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import reaudit as ra  # noqa: E402

e18 = ra.e18
ROOT = ra.ROOT
CAN = ROOT / "data" / "_canonical_v2"
OUT = ROOT / "data" / "_results" / "canonical_v2"
ra.OUT = OUT  # write_csv in reaudit.py reads this module global
SET_S = ["bit_rate", "file_size"]
SET_H_NUM = [c for c in e18.NUM if c not in SET_S]
SET_H_CAT = list(e18.CAT)
TAGS = ["color_space", "color_primaries", "color_transfer", "color_range"]


def tags(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=color_space,color_primaries,color_transfer,color_range", "-of", "json", str(path)],
                       capture_output=True, text=True)
    s = json.loads(r.stdout)["streams"][0]
    return tuple(s.get(t, "missing") for t in TAGS)


def tag_check(files_by_type, n=60):
    rnd = random.Random(20261026)
    out = {}
    for k, fl in files_by_type.items():
        smp = rnd.sample(fl, min(n, len(fl)))
        with ThreadPoolExecutor(max_workers=8) as ex:
            ts = list(ex.map(tags, smp))
        cnt = {}
        for t in ts:
            cnt[str(t)] = cnt.get(str(t), 0) + 1
        out[k] = dict(n=len(smp), tag_order=TAGS, combos=cnt,
                      all_bt709_tv=all(t == ("bt709", "bt709", "bt709", "tv") for t in ts))
    return out


def accept_H(res):
    hi = {n: res["A"][n]["ci95"][1] for n in res["A"]}
    bad = [s for s in res["single_feature"] if s["auc_direction_free"] >= 0.60]
    return dict(upper_bounds=hi, a_pass=all(v < 0.60 for v in hi.values()),
                features_at_or_above_0_60=bad, b_pass=len(bad) == 0,
                constant_features=res["constant_features"],
                PASS=all(v < 0.60 for v in hi.values()) and len(bad) == 0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = dict(protocol="canonical-v2", env=dict(os=platform.platform(), python=sys.version.split()[0],
                   numpy=np.__version__, opencv=cv2.__version__, ffprobe=e18.run(["ffprobe", "-version"]).splitlines()[0]),
                   set_H=dict(numeric=SET_H_NUM, categorical=SET_H_CAT), set_S=SET_S)
    GENS, REAL = e18.GENS, e18.REAL
    files = sorted(p for p in (CAN / "deepaction").glob("*/*/*.mp4") if p.parent.parent.name in GENS + [REAL])
    with ThreadPoolExecutor(max_workers=8) as ex:
        fs = list(ex.map(ra.feats, files))
    rows = [dict(source=f.parent.parent.name, prompt_id=int(f.parent.name), variant=f.stem, **d) for f, d in zip(files, fs)]
    ra.write_csv("deepaction_canonical_v2_features.csv", rows, ["source", "prompt_id", "variant"] + e18.CAT + e18.NUM + ra.PIX)
    da = dict(n=len(rows), n_real=sum(r["source"] == REAL for r in rows), n_generated=sum(r["source"] != REAL for r in rows))
    da["frame_count_by_source"] = {s: ra.counts([int(r["n_frames_decoded"]) for r in rows if r["source"] == s]) for s in [REAL] + GENS}
    rng = np.random.default_rng(e18.SEED)
    da["set_H"] = ra.audit(rows, SET_H_NUM, SET_H_CAT, "H", rng)
    da["set_H"]["acceptance"] = accept_H(da["set_H"])
    da["set_S"] = ra.audit(rows, SET_S, [], "S", rng)
    da["set_H_plus_S"] = ra.audit(rows, SET_H_NUM + SET_S, SET_H_CAT, "HS", rng)
    da["residual_pixel"] = ra.audit(rows, ra.PIX, [], "pixel", rng)
    da["per_source_pixel"] = ra.per_source_summary(rows, ra.PIX)
    da["per_source_S"] = ra.per_source_summary(rows, SET_S)
    da["colour_tags_by_source"] = {c: {s: ra.counts([str(r[c]) for r in rows if r["source"] == s]) for s in [REAL] + GENS}
                                   for c in TAGS}
    results["deepaction"] = da
    # DF26 descriptive only
    vid = {}
    with open(ROOT / "data" / "df26" / "metadata" / "videos.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            vid[r["video_path"]] = r
    base = CAN / "df26"
    dfiles = sorted(base.rglob("*.mp4"))
    with ThreadPoolExecutor(max_workers=8) as ex:
        dfs = list(ex.map(ra.feats, dfiles))
    drows, missing = [], 0
    for f, d in zip(dfiles, dfs):
        rel = f.relative_to(base).as_posix()
        m = vid.get(rel)
        if m is None:
            missing += 1
            continue
        drows.append(dict(source=m["generator"], label=m["label"], path=rel, **d))
    ra.write_csv("df26_canonical_v2_features.csv", drows, ["source", "label", "path"] + e18.CAT + e18.NUM + ra.PIX)
    cols = ["fps", "nb_frames", "file_size", "bit_rate", "gop_mean_i_dist", "width", "height", "duration"] + ra.PIX
    labels = sorted(set(r["label"] for r in drows))
    dd = dict(n=len(drows), n_not_in_metadata=missing, by_label={}, by_generator=ra.per_source_summary(drows, cols),
              categorical_by_label={})
    for lab in labels:
        dd["by_label"][lab] = ra.per_source_summary([dict(r, source=lab) for r in drows if r["label"] == lab], cols)[lab]
    for c in e18.CAT:
        dd["categorical_by_label"][c] = {lab: ra.counts([str(r[c]) for r in drows if r["label"] == lab]) for lab in labels}
    results["df26_descriptive_only"] = dd
    # ffprobe verification of the four colour tags
    da_real = [f for f in files if f.parent.parent.name == REAL]
    da_gen = [f for f in files if f.parent.parent.name != REAL]
    real_lab = {r["path"] for r in drows if r["label"] == "real"}
    df_real = [f for f in dfiles if f.relative_to(base).as_posix() in real_lab]
    df_fake = [f for f in dfiles if f.relative_to(base).as_posix() not in real_lab]
    results["colour_tag_verification"] = tag_check(dict(deepaction_real=da_real, deepaction_generated=da_gen,
                                                        df26_real=df_real, df26_fake=df_fake))
    (OUT / "e019_v2_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()
