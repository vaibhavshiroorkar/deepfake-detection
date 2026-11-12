"""E-022: semantic validation of the lip_sync module on controlled cases (docs/EXPERIMENTS.md, E-022 preregistration).

LAV-DF official train rows only. Media made with ffmpeg in a scratch folder. Usage:
  PYTHONPATH=src .venv/Scripts/python.exe scripts/lipsync_eval/e022.py --out data/_results/platform/e022_lipsync_cases_before.json --tag before --keep <scratch>
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAVDF = REPO / "data" / "lavdf" / "LAV-DF"
SEED = 20261026
SHIFTS_B = [-1000, -480, -240, -120, 120, 240, 480, 1000]
SHIFTS_C = [-40, 40]


def ff(args: list, log: list) -> bool:
    r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y"] + [str(a) for a in args], capture_output=True, text=True)
    if r.returncode != 0:
        log.append({"ffmpeg_failed": " ".join(map(str, args))[:300], "stderr": r.stderr[-300:]})
    return r.returncode == 0


def pick_clips() -> list:
    meta = json.load(open(LAVDF / "metadata.min.json"))
    rows = sorted(r["file"] for r in meta if r["split"] == "train" and not r["modify_video"] and not r["modify_audio"]
                  and r["n_fakes"] == 0 and r["duration"] >= 3.0)
    assert all(f.startswith("train/") for f in rows)
    random.Random(SEED).shuffle(rows)
    return rows[:40]


def shifted(src: Path, ms: int, dur: float, dst: Path, log: list) -> bool:
    if ms >= 0:
        af = f"adelay={ms}:all=1,atrim=0:{dur}"
    else:
        af = f"atrim=start={-ms / 1000.0},asetpts=PTS-STARTPTS,apad"
    return ff(["-i", src, "-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy", "-af", af, "-c:a", "aac", "-t", dur, dst], log)


def synth(kind: str) -> str:
    if kind == "white":
        return "anoisesrc=color=white:amplitude=0.2:sample_rate=16000"
    if kind == "pink":
        return "anoisesrc=color=pink:amplitude=0.2:sample_rate=16000"
    if kind == "tone220":
        return "sine=frequency=220:sample_rate=16000"
    if kind == "tone440":
        return "sine=frequency=440:sample_rate=16000"
    if kind == "chords":
        f = "261.63*pow(2,floor(mod(t,4))*5/12)"
        return f"aevalsrc='0.15*(sin(2*PI*{f}*t)+sin(2*PI*{f}*1.26*t)+sin(2*PI*{f}*1.5*t))*exp(-3*mod(t,1))':s=16000"
    raise ValueError(kind)


def probe_of(path: Path, media_type: str = "video"):
    from dfd.platform.module import Probe
    from dfd.prep.audio import has_audio_stream
    return Probe(path=str(path), filename=path.name, media_type=media_type, mime_type="video/mp4", size_bytes=path.stat().st_size,
                 sha256="0" * 64, duration_s=None, has_video=media_type == "video", has_audio=has_audio_stream(path))


def run_one(mod, path: Path, media_type="video") -> dict:
    from dfd.platform.module import Context
    t0 = time.perf_counter()
    pr = probe_of(path, media_type)
    ap = mod.applicability(pr)
    rec = {"file": path.name}
    if ap.kind != "applicable":
        rec.update(status="not_applicable", finding=None, outcome="not_applicable", reason=ap.reason, via="applicability")
        return rec
    try:
        res = mod.run(Context(probe=pr))
    except Exception as e:                                  # failures are logged, not hidden
        rec.update(status="error", finding=None, outcome="error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-400:])
        return rec
    obs = {o["code"]: o["value"] for o in (res.observations or [])}
    rec.update(status=res.status, finding=res.finding, outcome=res.finding if res.status == "completed" else res.status,
               warnings=[w["code"] for w in (res.warnings or [])], offset=obs.get("av_offset_frames"), conf=obs.get("sync_confidence"),
               reason=res.reason, wall_s=round(time.perf_counter() - t0, 2))
    return rec


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 3), round(float(c + h), 3)]


def summarise(recs: list) -> dict:
    n = len(recs)
    counts: dict = {}
    for r in recs:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    confs = [r["conf"] for r in recs if r.get("conf") is not None]
    out = {"n": n, "counts": counts, "wilson95": {k: wilson(v, n) for k, v in counts.items()}, "warning_codes": {}, "conf": None}
    for r in recs:
        for w in r.get("warnings", []):
            out["warning_codes"][w] = out["warning_codes"].get(w, 0) + 1
    if confs:
        out["conf"] = {"n": len(confs), "min": min(confs), "q25": float(np.percentile(confs, 25)), "median": float(np.median(confs)),
                       "q75": float(np.percentile(confs, 75)), "max": max(confs)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="before")
    ap.add_argument("--keep", default=None, help="scratch folder for built media (reused if files exist)")
    a = ap.parse_args()
    from dfd.platform.modules.lip_sync import build_lip_sync_module
    mod = build_lip_sync_module(device="cuda")
    log: list = []
    clips = pick_clips()
    work = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix="e022_"))
    work.mkdir(parents=True, exist_ok=True)
    src = [LAVDF / c for c in clips]
    durs = {}
    for s in src:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(s)], capture_output=True, text=True)
        durs[s.name] = float(r.stdout.strip())
    cases: dict = {}

    A = [run_one(mod, s) for s in src]
    cases["A"] = A
    base = {r["file"]: r.get("offset") for r in A}
    for case, shifts in (("B", SHIFTS_B), ("C", SHIFTS_C)):
        recs = []
        for s in src[:20]:
            for ms in shifts:
                p = work / f"{case}_{s.stem}_{ms}.mp4"
                if not p.exists():
                    shifted(s, ms, durs[s.name], p, log)
                if not p.exists():
                    continue
                r = run_one(mod, p)
                r["shift_ms"] = ms
                b = base.get(s.name)
                if r.get("offset") is not None and b is not None:
                    r["expected_offset"] = b - ms / 40.0
                    r["offset_error"] = r["offset"] - r["expected_offset"]
                recs.append(r)
        cases[case] = recs
    D = []
    for s in src[:10]:
        for kind in ("white", "pink", "tone220", "tone440", "chords"):
            p = work / f"D_{s.stem}_{kind}.mp4"
            if not p.exists():
                ff(["-i", s, "-f", "lavfi", "-i", synth(kind), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-ar", 16000,
                    "-t", durs[s.name], p], log)
            if p.exists():
                r = run_one(mod, p)
                r["kind"] = kind
                D.append(r)
    cases["D"] = D
    E: dict = {"E1_no_audio": [], "E2_no_face": [], "E3_short": [], "E4_two_faces": [], "E5_audio_only": []}
    for i, s in enumerate(src[:10]):
        p = work / f"E1_{s.stem}.mp4"
        if not p.exists():
            ff(["-i", s, "-an", "-c:v", "copy", p], log)
        if p.exists():
            E["E1_no_audio"].append(run_one(mod, p))
        p = work / f"E2_{s.stem}.mp4"
        if not p.exists():
            ff(["-f", "lavfi", "-i", f"testsrc2=size=224x224:rate=25:duration={durs[s.name]}", "-i", s, "-map", "0:v", "-map", "1:a:0",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-t", durs[s.name], p], log)
        if p.exists():
            E["E2_no_face"].append(run_one(mod, p))
        p = work / f"E3_{s.stem}.mp4"
        if not p.exists():
            ff(["-i", s, "-t", 1.5, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", p], log)
        if p.exists():
            E["E3_short"].append(run_one(mod, p))
        o = src[10 + i]
        p = work / f"E4_{s.stem}_{o.stem}.mp4"
        if not p.exists():
            ff(["-i", s, "-i", o, "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]", "-map", "0:a:0", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", p], log)
        if p.exists():
            E["E4_two_faces"].append(run_one(mod, p))
    for s in src[:5]:
        p = work / f"E5_{s.stem}.wav"
        if not p.exists():
            ff(["-i", s, "-vn", "-ac", 1, "-ar", 16000, p], log)
        if p.exists():
            E["E5_audio_only"].append(run_one(mod, p, media_type="audio"))
    cases["E"] = E

    summary: dict = {"A": summarise(A), "D": summarise(D)}
    for case in ("B", "C"):
        by = {}
        for ms in (SHIFTS_B if case == "B" else SHIFTS_C):
            rr = [r for r in cases[case] if r["shift_ms"] == ms]
            sm = summarise(rr)
            errs = [abs(r["offset_error"]) for r in rr if r.get("offset_error") is not None and r["outcome"] in ("consistent", "potential_mismatch")]
            sm["abs_offset_error_frames"] = ({"n": len(errs), "median": float(np.median(errs)), "mean": float(np.mean(errs)),
                                              "frac_within_1": float(np.mean(np.array(errs) <= 1))} if errs else None)
            by[str(ms)] = sm
        summary[case] = by
    summary["D_by_kind"] = {k: summarise([r for r in D if r["kind"] == k]) for k in ("white", "pink", "tone220", "tone440", "chords")}
    summary["E"] = {k: summarise(v) for k, v in E.items()}
    out = {"experiment": "E-022", "tag": a.tag, "date": time.strftime("%Y-%m-%d"), "seed": SEED, "clips": clips,
           "module_version": mod.version, "min_confidence": mod.min_confidence, "offset_tolerance": mod.offset_tolerance,
           "ffmpeg_log": log, "summary": summary, "cases": cases, "scratch": str(work)}
    Path(a.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, default=str))
    print("ffmpeg failures:", len(log), "scratch:", work)


if __name__ == "__main__":
    sys.exit(main())
