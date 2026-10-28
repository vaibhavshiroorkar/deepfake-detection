"""E-029 evaluation: the frozen heads, once, on the E-028 Eval rows; paired against frozen T-I (preregistered).

Usage (repo root, PYTHONPATH=src): python scripts/modern_image/e029_eval.py --final
Refuses without --final, if data/_results/modern/e029/eval_look.json exists, or if any file hash differs from checkpoints/modern-E029/freeze.json.
Writes eval_look.json (marker, before scoring), e029_results.json and e029_cells.csv.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
import _env  # noqa: E402
import evalcore as ev  # noqa: E402
from e028 import rate_table, reencode_control  # noqa: E402

CK = lib.ROOT / "checkpoints" / "modern-E029"
OUT = _env.RES / "e029"


def main():
    if "--final" not in sys.argv:
        sys.exit("refused: the single Eval look needs --final")
    look = OUT / "eval_look.json"
    if look.exists():
        sys.exit("refused: eval_look.json exists (one look only)")
    fr = json.load(open(CK / "freeze.json"))
    for f, h in fr["sha256"].items():
        if hashlib.sha256(open(CK / f, "rb").read()).hexdigest() != h:
            sys.exit(f"refused: {f} differs from freeze.json")
    json.dump({"look_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "freeze": fr}, open(look, "w"), indent=1)
    from dfd.eval import calibration as C
    from dfd.infer.checkpoint import Head
    meta = json.load(open(CK / "meta.json"))
    ckm = json.load(open(lib.CKPT / "metadata.json"))
    t0 = time.time()
    E = ev.load_eval()
    assert len(E["ti"]) + len(E["errors"]) == len(lib.subset_rows("eval")), "Eval extraction incomplete"
    st, gens, arch_of, sets = ev.define_sets(E)
    boot = ev.JointBoot(st)
    scores = {"T-I": E["ti"]}
    thr = {"T-I": float(ckm["tasks"]["T-I"]["threshold"]["value"]), "T-V": float(ckm["tasks"]["T-V"]["threshold"]["value"])}
    re_scores = {"T-I": np.array([r["ti"] for r in E["re"]])}
    for name, hm in meta["heads"].items():
        z = np.load(CK / name)
        head = Head(z["mean"], z["std"], z["w"], float(z["b"]))
        key = "f13" if hm["encoder"] == "IMG-13" else "f15"
        sc = name.replace("head_", "").replace(".npz", "")
        scores[sc] = C.sigmoid(C.logit(head.prob_rows(E[key])) / hm["temperature"])
        re_scores[sc] = C.sigmoid(C.logit(head.prob_rows(np.stack([r[key] for r in E["re"]]))) / hm["temperature"])
        thr[sc] = hm["threshold"]
    heads = [k for k in scores if k != "T-I"]
    chosen = meta["frozen_candidate"].replace("head_", "").replace(".npz", "")
    main_res, main_diff, R = ev.evaluate(E, scores, thr, sets, boot, gens, pairs=[(h, "T-I") for h in heads])
    # seed means per encoder (each resample: mean over the three seeds), and their paired difference to T-I
    seedmean = {}
    for enc in ["IMG-13", "IMG-15"]:
        hs = [f"{enc}_seed{s}" for s in (0, 1, 2)]
        for setname in ["macro", "pooled"] + [k for k in sets if k.startswith("arch:")]:
            keys = [("all", setname, h) for h in hs]
            if not all(k in R for k in keys):
                continue
            pts = np.array([R[k][0] for k in keys])
            reps = np.mean([R[k][1] for k in keys], axis=0)
            d = reps - R[("all", setname, "T-I")][1]
            seedmean.setdefault(enc, {})[setname] = {
                kk: {"seed_mean": lib.r4(pts[:, i].mean()), "seed_sd": lib.r4(pts[:, i].std(ddof=1)), "ci": [lib.r4(x) for x in lib.pct(reps[:, i])],
                     "minus_T-I": {"est": lib.r4(pts[:, i].mean() - R[("all", setname, "T-I")][0][i]), "ci": [lib.r4(x) for x in lib.pct(d[:, i])]}}
                for i, kk in enumerate(ev.KEYS)}
    # related versus unrelated generators (overlap.json)
    ov = json.load(open(OUT / "overlap.json"))
    groups = {"exact_seen": ov["exact_overlap"], "related_proxy": sorted(ov["related_proxy"]), "unrelated": ov["unrelated"]}
    by_group = {}
    for gname, gl in groups.items():
        if not gl:
            by_group[gname] = {"generators": [], "note": "empty"}
            continue
        by_group[gname] = {"generators": gl}
        for sc in scores:
            keys = [("all", f"gen:{g}", sc) for g in gl]
            pt = np.mean([R[k][0] for k in keys], axis=0)
            reps = np.mean([R[k][1] for k in keys], axis=0)
            e = {"macro_auc": {"est": lib.r4(pt[0]), "ci": [lib.r4(x) for x in lib.pct(reps[:, 0])]},
                 "macro_fnr": {"est": lib.r4(pt[3]), "ci": [lib.r4(x) for x in lib.pct(reps[:, 3])]}}
            if sc != "T-I":
                rt = np.mean([R[("all", f"gen:{g}", "T-I")][1] for g in gl], axis=0)
                ptt = np.mean([R[("all", f"gen:{g}", "T-I")][0] for g in gl], axis=0)
                e["macro_auc_minus_T-I"] = {"est": lib.r4(pt[0] - ptt[0]), "ci": [lib.r4(x) for x in lib.pct(reps[:, 0] - rt[:, 0])]}
            by_group[gname][sc] = e
    face = E["n_faces"] >= 1
    crop = ~np.isnan(E["tv"])
    sub_sets = {k: v for k, v in sets.items() if k.startswith("arch:") or k in ("pooled", "cell:Hourglass|ffhq")}
    pick = {"T-I": scores["T-I"], chosen: scores[chosen], "IMG-13_seed0": scores["IMG-13_seed0"], "IMG-15_seed0": scores["IMG-15_seed0"]}
    mask_res, mask_diff, _ = ev.evaluate(E, pick, thr, sub_sets, boot, gens, pairs=[(chosen, "T-I")] + ([("IMG-15_seed0", "T-I")] if chosen != "IMG-15_seed0" else []),
                                         masks={"face_routed": face, "not_face_routed": ~face, "png_only": E["format"] == "PNG",
                                                "jpeg_only": E["format"] == "JPEG"})
    fp = {"T-I": E["ti"], "T-V": E["tv"], chosen: scores[chosen]}
    face_res, face_diff, _ = ev.evaluate(E, fp, thr, sub_sets, boot, gens, pairs=[(chosen, "T-V"), (chosen, "T-I"), ("T-I", "T-V")],
                                         masks={"face_crop_scored": crop})
    re_ctl = {sc: reencode_control(E, sc, scores[sc], re_scores[sc], arch_of, sets, thr[sc]) for sc in scores}
    s = scores[chosen]
    t = thr[chosen]
    errors = {"fpr_by_real_source": rate_table(E, s, t, E["real_source"], 0), "fpr_by_real_format": rate_table(E, s, t, E["format"], 0),
              "fnr_by_fake_format": rate_table(E, s, t, E["format"], 1), "fnr_by_fake_resolution": rate_table(E, s, t, E["resolution"], 1),
              "fpr_by_face_routed": rate_table(E, s, t, np.where(face, "face", "no_face"), 0),
              "fnr_by_face_routed": rate_table(E, s, t, np.where(face, "face", "no_face"), 1)}
    res = {"experiment": "E-029", "stage": "eval_look", "run_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "chosen": chosen,
           "thresholds": thr, "n_fake": int((E["label"] == 1).sum()), "n_real": int((E["label"] == 0).sum()),
           "main": main_res["all"], "paired_vs_T-I": main_diff.get("all", {}), "seed_mean": seedmean, "by_overlap_group": by_group,
           "masked": mask_res, "masked_paired": mask_diff, "face_path": {"metrics": face_res, "paired": face_diff},
           "reencode_control": re_ctl, "errors_chosen": errors, "seconds": round(time.time() - t0, 1)}
    json.dump(res, open(OUT / "e029_results.json", "w"), indent=1)
    with open(OUT / "e029_cells.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "scorer", "n_fake", "n_real"] + [f"{k}{x}" for k in ev.KEYS for x in ("", "_lo", "_hi")])
        for setname, v in main_res["all"].items():
            for sc, m in v.items():
                w.writerow([setname, sc, m.get("n_fake"), m.get("n_real")] + [y for k in ev.KEYS for y in (m[k]["est"], *m[k]["ci"])])
    print("done", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
