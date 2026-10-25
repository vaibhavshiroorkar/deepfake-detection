"""E-024: independent validation of the frozen lip_sync module (docs/EXPERIMENTS.md, E-024 preregistration).

Reuses the media builders of e022.py. Does not modify the module. Usage:
  PYTHONPATH=src .venv/Scripts/python.exe scripts/lipsync_eval/e024.py --keep <scratch>
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("e022", HERE / "e022.py")
e022 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e022)

REPO = e022.REPO
LAVDF = e022.LAVDF
DATA = REPO / "data"
SEED = 20261027
OUTDIR = DATA / "_results" / "platform"
SPLIT = DATA / "_splits" / "fakeavceleb-79d3f7051972.csv"
MANIFEST = DATA / "_manifests" / "fakeavceleb.manifest.csv"
SHIFTS_D = [-1000, -480, -240, -120, 120, 240, 480, 1000]
IN_RANGE = [-480, -240, -120, 120, 240, 480]
wilson = e022.wilson


def lavdf_pool():
    meta = json.load(open(LAVDF / "metadata.min.json"))
    rows = sorted(r["file"] for r in meta if r["split"] == "train" and not r["modify_video"] and not r["modify_audio"]
                  and r["n_fakes"] == 0 and r["duration"] >= 3.0)
    assert all(f.startswith("train/") for f in rows)
    return rows


def excluded_e022():
    rows = lavdf_pool()
    random.Random(e022.SEED).shuffle(rows)
    ex = rows[:40]
    ref = json.load(open(OUTDIR / "e022_lipsync_cases_before.json"))["clips"]
    assert ex == ref, "recomputed E-022 clip list differs from the stored one"
    return ex


def fav_rows():
    asg = {}
    with open(SPLIT, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            asg[r["sample_id"]] = r["split"]
    out = []
    with open(MANIFEST, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if asg.get(r["sample_id"]) != "dev_x":          # test_x and anything else is never touched
                continue
            m = re.search(r"category=(\w+)", r["notes"])
            out.append({"sid": r["sample_id"], "path": DATA / r["path"], "cat": m.group(1) if m else "?", "method": r["method"],
                        "group": r["group_id"]})
    return out


def pick_by_identity(rows, n, seed):
    rows = sorted(rows, key=lambda r: r["sid"])
    random.Random(seed).shuffle(rows)
    seen, out = set(), []
    for r in rows:
        if r["group"] in seen:
            continue
        seen.add(r["group"])
        out.append(r)
        if len(out) == n:
            break
    return out


def summ(recs):
    s = e022.summarise(recs)
    s["frac"] = {k: v / s["n"] for k, v in s["counts"].items()} if s["n"] else {}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", required=True)
    ap.add_argument("--out", default=str(OUTDIR / "e024_lipsync_validation.json"))
    a = ap.parse_args()
    from dfd.platform.modules.lip_sync import build_lip_sync_module
    mod = build_lip_sync_module(device="cuda")
    run_one, ff = e022.run_one, e022.ff
    log: list = []
    work = Path(a.keep)
    work.mkdir(parents=True, exist_ok=True)

    ex = excluded_e022()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "e024_excluded_e022_clips.json").write_text(json.dumps({"rule": "e022.pick_clips, seed 20261026", "clips": ex}, indent=1))
    pool = [f for f in lavdf_pool() if f not in set(ex)]
    random.Random(SEED).shuffle(pool)
    clips = pool[:60]
    assert not set(clips) & set(ex)
    src = [LAVDF / c for c in clips]
    durs = {}
    for s in src:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(s)], capture_output=True, text=True)
        durs[s.name] = float(r.stdout.strip())

    fav = fav_rows()
    assert all(r["cat"] in ("RVRA", "RVFA", "FVRA", "FVFA") for r in fav)
    B_rows = pick_by_identity([r for r in fav if r["cat"] == "RVRA"], 60, SEED)
    strata = {"FVRA_wav2lip": ("FVRA", {"wav2lip"}), "FVFA_wav2lip": ("FVFA", {"wav2lip"}),
              "FVFA_faceswap-wav2lip": ("FVFA", {"faceswap-wav2lip"}), "FVFA_fsgan-wav2lip": ("FVFA", {"fsgan-wav2lip"})}
    C_rows = {k: pick_by_identity([r for r in fav if r["cat"] == c and r["method"] in m], 40, SEED) for k, (c, m) in strata.items()}

    cases: dict = {}
    t0 = time.time()
    A = [run_one(mod, s) for s in src]
    cases["A"] = A
    print("A done", round(time.time() - t0), flush=True)
    Bc = []
    for r in B_rows:
        rec = run_one(mod, r["path"])
        rec["sid"] = r["sid"]
        Bc.append(rec)
    cases["B"] = Bc
    print("B done", round(time.time() - t0), flush=True)
    Cc = {}
    for k, rows in C_rows.items():
        Cc[k] = []
        for r in rows:
            rec = run_one(mod, r["path"])
            rec["sid"] = r["sid"]
            Cc[k].append(rec)
    cases["C"] = Cc
    print("C done", round(time.time() - t0), flush=True)

    base = {r["file"]: (r.get("offset") if r["outcome"] in ("consistent", "potential_mismatch") else None) for r in A}
    for case, shifts in (("D", SHIFTS_D), ("E", [-40, 40])):
        recs = []
        for s in src[:30]:
            for ms in shifts:
                p = work / f"{case}_{s.stem}_{ms}.mp4"
                if not p.exists():
                    e022.shifted(s, ms, durs[s.name], p, log)
                if not p.exists():
                    continue
                r = run_one(mod, p)
                r["shift_ms"] = ms
                r["base_offset"] = base.get(s.name)
                b = base.get(s.name)
                if b is not None:
                    r["expected_offset"] = b - ms / 40.0
                    if r.get("offset") is not None:
                        r["offset_error"] = r["offset"] - r["expected_offset"]
                recs.append(r)
        cases[case] = recs
        print(case, "done", round(time.time() - t0), flush=True)
    F = []
    for s in src[:20]:
        for kind in ("white", "pink", "tone220", "tone440", "chords"):
            p = work / f"F_{s.stem}_{kind}.mp4"
            if not p.exists():
                ff(["-i", s, "-f", "lavfi", "-i", e022.synth(kind), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-ar", 16000,
                    "-t", durs[s.name], p], log)
            if p.exists():
                r = run_one(mod, p)
                r["kind"] = kind
                F.append(r)
    cases["F"] = F
    print("F done", round(time.time() - t0), flush=True)
    G: dict = {"g1_no_audio": [], "g2_no_face": [], "g3_short": [], "g4_audio_only": []}
    for i, s in enumerate(src[30:50]):
        p = work / f"G1_{s.stem}.mp4"
        if not p.exists():
            ff(["-i", s, "-an", "-c:v", "copy", p], log)
        if p.exists():
            G["g1_no_audio"].append(run_one(mod, p))
        p = work / f"G2_{s.stem}.mp4"
        if not p.exists():
            ff(["-f", "lavfi", "-i", f"testsrc2=size=224x224:rate=25:duration={durs[s.name]}", "-i", s, "-map", "0:v", "-map", "1:a:0",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-t", durs[s.name], p], log)
        if p.exists():
            G["g2_no_face"].append(run_one(mod, p))
        p = work / f"G3_{s.stem}.mp4"
        if not p.exists():
            ff(["-i", s, "-t", 1.5, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", p], log)
        if p.exists():
            G["g3_short"].append(run_one(mod, p))
        if i < 10:
            p = work / f"G4_{s.stem}.wav"
            if not p.exists():
                ff(["-i", s, "-vn", "-ac", 1, "-ar", 16000, p], log)
            if p.exists():
                G["g4_audio_only"].append(run_one(mod, p, media_type="audio"))
    cases["G"] = G

    # ---- summary and lines
    S: dict = {"A": summ(A), "B": summ(Bc), "C": {k: summ(v) for k, v in Cc.items()}}
    for case, shifts in (("D", SHIFTS_D), ("E", [-40, 40])):
        by = {}
        for ms in shifts:
            rr = [r for r in cases[case] if r["shift_ms"] == ms]
            sm = summ(rr)
            elig = [r for r in rr if r.get("expected_offset") is not None]
            errs = [abs(r["offset_error"]) for r in elig if r.get("offset_error") is not None and r["outcome"] in ("consistent", "potential_mismatch")]
            ok = [r for r in elig if (r.get("offset_error") is not None and r["outcome"] in ("consistent", "potential_mismatch") and abs(r["offset_error"]) <= 2)
                  or r["outcome"] == "potential_mismatch"]
            strict = [e for e in errs if e <= 2]
            sm["elig_n"] = len(elig)
            sm["ok_error_le2_or_flagged"] = [len(ok), len(elig), wilson(len(ok), len(elig))]
            sm["strict_error_le2_among_completed"] = [len(strict), len(errs), wilson(len(strict), len(errs))]
            sm["abs_offset_error_frames"] = ({"n": len(errs), "median": float(np.median(errs)), "mean": float(np.mean(errs))} if errs else None)
            by[str(ms)] = sm
        S[case] = by
    S["F"] = summ(F)
    S["F_by_kind"] = {k: summ([r for r in F if r["kind"] == k]) for k in ("white", "pink", "tone220", "tone440", "chords")}
    S["G"] = {k: summ(v) for k, v in G.items()}

    def cnt(recs, *outs):
        return sum(1 for r in recs if r["outcome"] in outs)

    L = {}
    for name, recs in (("L-A", A), ("L-B", Bc)):
        k, n = cnt(recs, "potential_mismatch"), len(recs)
        ki, _ = cnt(recs, "inconclusive", "not_applicable"), n
        L[name] = {"mismatch": [k, n, wilson(k, n)], "inconclusive_or_na": [ki, n, wilson(ki, n)],
                   "pass": (k / n) <= 0.10, "weak": wilson(k, n)[1] > 0.20, "usability_flag": ki / n > 0.30}
    din = [r for r in cases["D"] if r["shift_ms"] in IN_RANGE and r.get("expected_offset") is not None]
    ok = [r for r in din if (r.get("offset_error") is not None and r["outcome"] in ("consistent", "potential_mismatch") and abs(r["offset_error"]) <= 2) or r["outcome"] == "potential_mismatch"]
    L["L-D-in"] = {"ok": [len(ok), len(din), wilson(len(ok), len(din))], "pass": len(din) > 0 and len(ok) / len(din) >= 0.90}
    d480 = [r for r in cases["D"] if abs(r["shift_ms"]) == 480]
    k = cnt(d480, "consistent")
    L["L-D-480"] = {"false_consistent": [k, len(d480), wilson(k, len(d480))], "pass": k / len(d480) <= 0.10}
    d1k = [r for r in cases["D"] if abs(r["shift_ms"]) == 1000]
    k = cnt(d1k, "consistent")
    km = cnt(d1k, "potential_mismatch")
    L["L-D-out"] = {"consistent": [k, len(d1k), wilson(k, len(d1k))], "mismatch_concern": [km, len(d1k), wilson(km, len(d1k))],
                    "inconclusive_or_na": cnt(d1k, "inconclusive", "not_applicable"), "pass": k / len(d1k) <= 0.10}
    k = cnt(cases["E"], "potential_mismatch")
    L["L-E"] = {"mismatch": [k, len(cases["E"]), wilson(k, len(cases["E"]))], "pass": k / len(cases["E"]) <= 0.20}
    k = cnt(F, "consistent", "potential_mismatch")
    L["L-F"] = {"non_inconclusive": [k, len(F), wilson(k, len(F))], "pass": k / len(F) <= 0.05}
    allg = [r for v in G.values() for r in v]
    k = cnt(allg, "consistent", "potential_mismatch", "error")
    L["L-G"] = {"bad": [k, len(allg), wilson(k, len(allg))], "pass": k == 0}
    L["all_pass"] = all(v["pass"] for v in L.values() if isinstance(v, dict) and "pass" in v)
    out = {"experiment": "E-024", "date": time.strftime("%Y-%m-%d"), "seed": SEED, "module_version": mod.version,
           "min_confidence": mod.min_confidence, "offset_tolerance": mod.offset_tolerance, "lavdf_clips": clips,
           "excluded_e022_clips": ex, "fakeavceleb_B": [r["sid"] for r in B_rows],
           "fakeavceleb_C": {k: [r["sid"] for r in v] for k, v in C_rows.items()},
           "ffmpeg_log": log, "lines": L, "summary": S, "cases": cases, "scratch": str(work)}
    Path(a.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(L, default=str))
    print("ffmpeg failures:", len(log), "wall_s", round(time.time() - t0))


if __name__ == "__main__":
    sys.exit(main())
