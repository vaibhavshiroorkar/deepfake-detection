"""E-025: empirical offset band for the lip-sync module (docs/EXPERIMENTS.md, E-025 preregistration). The module is NOT modified; it is a measuring instrument.

  PYTHONPATH=src .venv/Scripts/python.exe scripts/lipsync_eval/e025.py --keep <scratch>

Stage 1 writes data/_results/platform/e025_band.json (the frozen band) BEFORE stage 2 touches any validation or Wav2Lip clip.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("e024", HERE / "e024.py")
e024 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e024)
e022 = e024.e022

SEED = 20261028
OUT = e024.OUTDIR
wilson = e022.wilson
OLD_TOL = 2
MAX_WIDTH = 10


def used_before() -> dict:
    ex22 = set(e024.excluded_e022())
    d24 = json.load(open(OUT / "e024_lipsync_validation.json"))
    lav24 = set(d24["lavdf_clips"])
    fav24 = set(d24["fakeavceleb_B"]) | {s for v in d24["fakeavceleb_C"].values() for s in v}
    return {"lav": ex22 | lav24, "fav_sids": fav24, "fav_b_sids": set(d24["fakeavceleb_B"])}


def pools():
    u = used_before()
    lav = [f for f in e024.lavdf_pool() if f not in u["lav"]]
    random.Random(SEED).shuffle(lav)
    cal_l, val_l = lav[:150], lav[150:300]
    fav = e024.fav_rows()
    sid2grp = {r["sid"]: r["group"] for r in fav}
    used_groups = {sid2grp[s] for s in u["fav_b_sids"] if s in sid2grp}
    rv = [r for r in fav if r["cat"] == "RVRA" and r["sid"] not in u["fav_sids"] and r["group"] not in used_groups]
    picked = e024.pick_by_identity(rv, 160, SEED)
    cal_f, val_f = picked[:80], picked[80:160]
    strata = {"FVRA_wav2lip": ("FVRA", {"wav2lip"}), "FVFA_wav2lip": ("FVFA", {"wav2lip"}),
              "FVFA_faceswap-wav2lip": ("FVFA", {"faceswap-wav2lip"}), "FVFA_fsgan-wav2lip": ("FVFA", {"fsgan-wav2lip"})}
    W = {k: e024.pick_by_identity([r for r in fav if r["cat"] == c and r["method"] in m and r["sid"] not in u["fav_sids"]], 40, SEED)
         for k, (c, m) in strata.items()}
    assert len(cal_l) == 150 and len(val_l) == 150 and len(cal_f) == 80 and len(val_f) == 80
    assert not {r["group"] for r in cal_f} & {r["group"] for r in val_f}
    assert not set(cal_l) & set(val_l) and not (set(cal_l) | set(val_l)) & u["lav"]
    return cal_l, val_l, cal_f, val_f, W


def conf(recs):
    return [r for r in recs if r["outcome"] in ("consistent", "potential_mismatch") and r.get("offset") is not None]


def in_band(off, band):
    return band[0] <= off <= band[1]


def flag_rate(recs, band):
    c = conf(recs)
    k = sum(1 for r in c if not in_band(r["offset"], band))
    return {"flagged": k, "n_all": len(recs), "n_confident": len(c), "rate_all": k / len(recs) if recs else None, "wilson_all": wilson(k, len(recs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", required=True)
    a = ap.parse_args()
    from dfd.platform.modules.lip_sync import build_lip_sync_module
    mod = build_lip_sync_module(device="cuda")
    assert mod.version == "syncnet-v2-port-2" and mod.min_confidence == 3.0 and mod.offset_tolerance == 2
    work = Path(a.keep)
    work.mkdir(parents=True, exist_ok=True)
    cal_l, val_l, cal_f, val_f, W = pools()
    t0 = time.time()
    # ---------------- stage 1 (calibration pools only)
    c_l = [e022.run_one(mod, e024.LAVDF / c) | {"src": "lavdf"} for c in cal_l]
    c_f = [e022.run_one(mod, r["path"]) | {"src": "fav", "sid": r["sid"]} for r in cal_f]
    cal = c_l + c_f
    offs = np.array([r["offset"] for r in conf(cal)], dtype=int)
    lo, hi = int(np.percentile(offs, 2.5, method="lower")), int(np.percentile(offs, 97.5, method="higher"))
    stage1 = {"experiment": "E-025", "stage": 1, "seed": SEED, "n_cal": len(cal), "n_confident": int(len(offs)),
              "inconclusive_or_na": sum(1 for r in cal if r["outcome"] not in ("consistent", "potential_mismatch")),
              "offset_histogram": {str(k): int((offs == k).sum()) for k in sorted(set(offs.tolist()))},
              "band": [lo, hi], "width": hi - lo, "stop_rule_width_over": MAX_WIDTH, "stopped": (hi - lo) > MAX_WIDTH,
              "inside_fraction": float(np.mean((offs >= lo) & (offs <= hi))), "cal_lavdf": cal_l, "cal_fav": [r["sid"] for r in cal_f],
              "records": cal, "wall_s": round(time.time() - t0)}
    (OUT / "e025_band.json").write_text(json.dumps(stage1, indent=1, default=str), encoding="utf-8")
    print("stage 1:", stage1["band"], "n_conf", stage1["n_confident"], flush=True)
    if stage1["stopped"]:
        print("STOP RULE: band wider than", MAX_WIDTH)
        return 0
    band = [lo, hi]
    # ---------------- stage 2 (frozen band)
    v_l = [e022.run_one(mod, e024.LAVDF / c) | {"src": "lavdf"} for c in val_l]
    v_f = [e022.run_one(mod, r["path"]) | {"src": "fav", "sid": r["sid"]} for r in val_f]
    val = v_l + v_f
    wres = {}
    for k, rows in W.items():
        recs = [e022.run_one(mod, r["path"]) | {"sid": r["sid"]} for r in rows]
        wres[k] = {"records": recs, "band": flag_rate(recs, band), "old_band": flag_rate(recs, [-OLD_TOL, OLD_TOL]),
                   "offset_histogram": {str(o): int(sum(1 for r in conf(recs) if r["offset"] == o)) for o in range(-8, 5)}}
    durs = {}
    base = {}
    for r in v_l[:30]:
        base[r["file"]] = r.get("offset") if r["outcome"] in ("consistent", "potential_mismatch") else None
    shifted, log = {}, []
    src_by_name = {Path(c).name: e024.LAVDF / c for c in val_l[:30]}
    for ms in (-480, -240, -120, 120, 240, 480):
        recs = []
        for name, s in src_by_name.items():
            if name not in durs:
                durs[name] = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(s)],
                                                  capture_output=True, text=True).stdout.strip())
            p = work / f"S_{s.stem}_{ms}.mp4"
            if not p.exists():
                e022.shifted(s, ms, durs[name], p, log)
            if p.exists() and base.get(name) is not None:
                r = e022.run_one(mod, p)
                r["shift_ms"], r["expected_offset"] = ms, base[name] - ms / 40.0
                recs.append(r)
        shifted[str(ms)] = {"records": recs, "band": flag_rate(recs, band), "old_band": flag_rate(recs, [-OLD_TOL, OLD_TOL])}
    v1 = flag_rate(val, band)
    v2 = {"lavdf": flag_rate(v_l, band), "fav": flag_rate(v_f, band)}
    old = flag_rate(val, [-OLD_TOL, OLD_TOL])
    lines = {"V1": {**v1, "pass": v1["rate_all"] <= 0.10 and v1["wilson_all"][1] < 0.20, "weak": v1["rate_all"] <= 0.10 and v1["wilson_all"][1] >= 0.20},
             "V2": {k: {**v, "ok": v["rate_all"] <= 0.10} for k, v in v2.items()}, "V1_old_band_for_comparison": old}
    lines["new_version_allowed"] = bool(lines["V1"]["pass"] and all(v["ok"] for v in lines["V2"].values()))
    out = {"experiment": "E-025", "stage": 2, "band": band, "lines": lines,
           "W": {k: {kk: vv for kk, vv in v.items() if kk != "records"} for k, v in wres.items()},
           "shifted": {k: {kk: vv for kk, vv in v.items() if kk != "records"} for k, v in shifted.items()},
           "val_lavdf": val_l, "val_fav": [r["sid"] for r in val_f], "val_records": val, "W_records": {k: v["records"] for k, v in wres.items()},
           "shifted_records": {k: v["records"] for k, v in shifted.items()}, "ffmpeg_log": log, "wall_s": round(time.time() - t0)}
    (OUT / "e025_validation.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(lines, default=str)[:1500])
    print(json.dumps({k: v["band"]["rate_all"] for k, v in wres.items()}), "ffmpeg failures", len(log), "wall_s", out["wall_s"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
