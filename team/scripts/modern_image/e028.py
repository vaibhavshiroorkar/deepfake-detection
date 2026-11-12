"""E-028: the frozen T-I detector on the CommunityForensics-Eval subset I7-S (preregistered in docs/EXPERIMENTS.md; data/_results/modern/e028/prereg.json).

Usage (repo root, PYTHONPATH=src): python scripts/modern_image/e028.py [--partial]
Reads the per-shard cache of extract.py; runs the parity check (200 rows rescored one at a time by RealAdapter._score_whole_image), then the
metrics, the face analysis (F1 to F3), the format and size controls (C1 to C3) and the error tables. Nothing is fitted or chosen.
Writes data/_results/modern/e028/{parity.json, e028_results.json, e028_cells.csv}.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
import _env  # noqa: E402
import evalcore as ev  # noqa: E402

OUT = _env.RES / "e028"


def parity(E, n=200):
    import cv2
    from dfd.infer.real_adapter import RealAdapter
    rng = np.random.default_rng(lib.SEED)
    pick = sorted(rng.choice(len(E["ti"]), size=min(n, len(E["ti"])), replace=False).tolist())
    adapter = RealAdapter(lib.CKPT, device="cuda")
    shards = lib.verified_shards("eval")
    want = {}
    for i in pick:
        want.setdefault(E["shard"][i], {})[int(E["row"][i])] = i
    diffs = []
    for sh, rows in want.items():
        for row, d in lib.iter_shard(shards[sh], columns=("image_data",)):
            if row in rows:
                bgr = cv2.imdecode(np.frombuffer(d["image_data"], np.uint8), cv2.IMREAD_COLOR)
                p = adapter._score_whole_image("T-I", bgr)
                diffs.append(abs(p - E["ti"][rows[row]]))
    rec = {"n": len(diffs), "max_abs_diff": float(np.max(diffs)), "median_abs_diff": float(np.median(diffs)), "tolerance": 0.02,
           "pass": bool(np.max(diffs) <= 0.02), "rule": "200 rows, numpy default_rng(20261027), RealAdapter._score_whole_image one at a time"}
    json.dump(rec, open(OUT / "parity.json", "w"), indent=1)
    return rec


def reencode_control(E, score_name, s_orig, re_scores, arch_of, sets, thr, n_boot=lib.N_BOOT):
    """C2: per architecture and pooled, AUC (and FNR at thr) of the sampled PNG fakes before and after JPEG-90 re-encoding against the
    same reals; paired bootstrap (reals and sampled fakes resampled separately, the same draws for both versions)."""
    key_to_i = {k: i for i, k in enumerate(E["key"])}
    samp = np.array([key_to_i[f"{r['shard']}#{r['row']}"] for r in E["re"]])
    s_re = np.asarray(re_scores, float)
    y = E["label"]
    out = {"n_sampled": int(len(samp)), "median_score_change": lib.r4(np.median(s_re - s_orig[samp])),
           "share_score_lower_after": lib.r4(np.mean(s_re < s_orig[samp])),
           "fnr_before": lib.r4(np.mean(s_orig[samp] < thr)), "fnr_after": lib.r4(np.mean(s_re < thr)), "by_set": {}}
    rng = np.random.default_rng(lib.SEED + 1)
    groups = {"pooled": None, **{f"arch:{a}": a for a in sorted(set(arch_of.values()))}}
    for name, a in groups.items():
        real = np.flatnonzero(sets[name if a else "pooled"] & (y == 0))
        fs = np.flatnonzero([(a is None or arch_of[E["model_name"][i]] == a) for i in samp])
        if len(fs) < 10:
            continue
        sr, sf0, sf1 = s_orig[real], s_orig[samp[fs]], s_re[fs]
        yy = np.r_[np.zeros(len(real)), np.ones(len(fs))]
        Wr = rng.multinomial(len(real), np.full(len(real), 1 / len(real)), size=n_boot)
        Wf = rng.multinomial(len(fs), np.full(len(fs), 1 / len(fs)), size=n_boot)
        W = np.concatenate([Wr, Wf], 1)
        m0 = ev.metrics_matrix(yy, np.r_[sr, sf0], W, thr)
        m1 = ev.metrics_matrix(yy, np.r_[sr, sf1], W, thr)
        p0 = ev.metrics_w(yy, np.r_[sr, sf0], np.ones(len(yy)), thr)
        p1 = ev.metrics_w(yy, np.r_[sr, sf1], np.ones(len(yy)), thr)
        out["by_set"][name] = {"n_fake_sampled": int(len(fs)), "n_real": int(len(real)),
                               "auc_original": lib.r4(p0[0]), "auc_reencoded": lib.r4(p1[0]),
                               "auc_change": {"est": lib.r4(p1[0] - p0[0]), "ci": [lib.r4(x) for x in lib.pct(m1[:, 0] - m0[:, 0])]},
                               "fnr_original": lib.r4(p0[3]), "fnr_reencoded": lib.r4(p1[3]),
                               "fnr_change": {"est": lib.r4(p1[3] - p0[3]), "ci": [lib.r4(x) for x in lib.pct(m1[:, 3] - m0[:, 3])]}}
    return out


def rate_table(E, s, thr, by, label):
    """Share flagged (label 0: FPR; label 1: 1 - FNR) per value of a column, with Wilson 95 percent intervals."""
    out = {}
    m = E["label"] == label
    for v in sorted(set(by[m])):
        k = m & (by == v)
        n = int(k.sum())
        x = int((s[k] >= thr).sum()) if label == 0 else int((s[k] < thr).sum())
        z = 1.959964
        ph = x / n
        den = 1 + z * z / n
        c = (ph + z * z / (2 * n)) / den
        h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
        out[str(v)] = {"n": n, ("fpr" if label == 0 else "fnr"): lib.r4(ph), "wilson": [lib.r4(c - h), lib.r4(c + h)]}
    return out


def descriptive(E):
    lab, rs, mn = E["label"], E["real_source"], E["model_name"]
    face = E["n_faces"] >= 1
    d = {"face_routed_share": {}, "format_by_label": {}}
    for name, m in [("real", lab == 0), ("fake", lab == 1)]:
        d["face_routed_share"][name] = {"all": lib.r4(face[m].mean()), "n": int(m.sum())}
    for v in sorted(set(rs[lab == 0])):
        k = (lab == 0) & (rs == v)
        d["face_routed_share"][f"real:{v}"] = {"share": lib.r4(face[k].mean()), "n": int(k.sum())}
    for g in sorted(set(mn[lab == 1])):
        k = (lab == 1) & (mn == g)
        d["face_routed_share"][f"fake:{g}"] = {"share": lib.r4(face[k].mean()), "n": int(k.sum())}
    for name, m in [("real", lab == 0), ("fake", lab == 1)]:
        vals, cnt = np.unique(E["format"][m], return_counts=True)
        d["format_by_label"][name] = {str(v): int(c) for v, c in zip(vals, cnt)}
    small = (E["h"] < 32) | (E["w"] < 32)
    d["images_smaller_than_32px"] = {"real": int((small & (lab == 0)).sum()), "fake": int((small & (lab == 1)).sum())}
    return d


def main():
    partial = "--partial" in sys.argv
    t0 = time.time()
    E = ev.load_eval()
    n_expected = len(lib.subset_rows("eval"))
    n_err = len(E["errors"])
    complete = len(E["ti"]) + n_err == n_expected
    print("rows", len(E["ti"]), "errors", n_err, "expected", n_expected, "complete", complete, flush=True)
    if not complete and not partial:
        sys.exit("extraction incomplete; rerun extract.py or pass --partial for a labelled partial look")
    par = parity(E)
    print("parity", par, flush=True)
    if not par["pass"]:
        sys.exit("parity check failed: score every row one at a time (preregistered rule)")
    meta = json.load(open(lib.CKPT / "metadata.json"))
    t_ti = float(meta["tasks"]["T-I"]["threshold"]["value"])
    t_tv = float(meta["tasks"]["T-V"]["threshold"]["value"])
    st, gens, arch_of, sets = ev.define_sets(E)
    boot = ev.JointBoot(st)
    face = E["n_faces"] >= 1
    crop = ~np.isnan(E["tv"])
    fmt_png = (E["format"] == "PNG").astype(float)
    logpix = np.log(E["h"].astype(float) * E["w"].astype(float))
    scores = {"T-I": E["ti"]}
    thr = {"T-I": t_ti, "T-V": t_tv, "format_png": 0.5, "log_pixels": float(np.median(logpix))}
    main_res, _, _ = ev.evaluate(E, scores, thr, sets, boot, gens)
    masks = {"face_routed": face, "not_face_routed": ~face, "png_only": E["format"] == "PNG", "jpeg_only": E["format"] == "JPEG"}
    sub_sets = {k: v for k, v in sets.items() if k.startswith("arch:") or k == "pooled" or k == "cell:Hourglass|ffhq"}
    mask_res, _, _ = ev.evaluate(E, scores, thr, sub_sets, boot, gens, masks=masks)
    face_res, face_diff, _ = ev.evaluate(E, {"T-I": E["ti"], "T-V": E["tv"]}, thr, sub_sets, boot, gens, pairs=[("T-I", "T-V")],
                                         masks={"face_crop_scored": crop})
    cue_res, _, _ = ev.evaluate(E, {"format_png": fmt_png, "log_pixels": logpix}, thr, sets, boot, gens)
    re_ctl = reencode_control(E, "T-I", E["ti"], [r["ti"] for r in E["re"]], arch_of, sets, t_ti)
    errors = {"fpr_by_real_source": rate_table(E, E["ti"], t_ti, E["real_source"], 0),
              "fpr_by_real_format": rate_table(E, E["ti"], t_ti, E["format"], 0),
              "fnr_by_fake_format": rate_table(E, E["ti"], t_ti, E["format"], 1),
              "fnr_by_fake_resolution": rate_table(E, E["ti"], t_ti, E["resolution"], 1),
              "fpr_by_face_routed": rate_table(E, E["ti"], t_ti, np.where(face, "face", "no_face"), 0),
              "fnr_by_face_routed": rate_table(E, E["ti"], t_ti, np.where(face, "face", "no_face"), 1),
              "score_quantiles": {k: [lib.r4(q) for q in np.quantile(E["ti"][E["label"] == v], [0.05, 0.25, 0.5, 0.75, 0.95])]
                                  for k, v in [("real", 0), ("fake", 1)]}}
    res = {"experiment": "E-028", "run_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "partial": not complete,
           "n_rows_scored": int(len(E["ti"])), "n_expected": n_expected, "decode_errors": E["errors"],
           "n_fake": int((E["label"] == 1).sum()), "n_real": int((E["label"] == 0).sum()),
           "checkpoint": meta["checkpoint_id"], "threshold_T-I": t_ti, "threshold_T-V": t_tv, "parity": par,
           "architecture_of_generator": arch_of, "descriptive": descriptive(E),
           "main": main_res["all"], "masked": mask_res, "face_path": {"metrics": face_res, "paired": face_diff},
           "cue_only_scores": cue_res["all"], "reencode_control": re_ctl, "errors": errors,
           "bootstrap": {"n_boot": lib.N_BOOT, "seed": lib.SEED, "unit": "image, resampled within each (generator, label, real source) stratum; RAISE reals one stratum"},
           "seconds": round(time.time() - t0, 1)}
    json.dump(res, open(OUT / "e028_results.json", "w"), indent=1)
    with open(OUT / "e028_cells.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "scorer", "n_fake", "n_real"] + [f"{k}{x}" for k in ev.KEYS for x in ("", "_lo", "_hi")])
        for setname, v in main_res["all"].items():
            for sc, m in v.items():
                w.writerow([setname, sc, m.get("n_fake"), m.get("n_real")] + [y for k in ev.KEYS for y in (m[k]["est"], *m[k]["ci"])])
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
