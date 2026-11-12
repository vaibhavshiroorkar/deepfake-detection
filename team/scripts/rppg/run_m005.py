"""M-005: positive control for the S-1 pulse check (docs/EXPERIMENTS.md M-005, D-062 d).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/rppg/run_m005.py --workers 8          # the full run, the 432 S-1 clips
    PYTHONPATH=src .venv/Scripts/python.exe scripts/rppg/run_m005.py --smoke              # 3 clips per dataset, output in M-005_smoke

CPU only, at most 8 worker processes. The clips, segments, face tracking, usability rules, POS, SNR rule and shuffled control are those of S-1
(dfd.rppg.pulse, scripts/rppg/run_s1.py); this script only adds a sinusoid to the skin pixels before the mean is taken. The frames are
read and the faces detected once per clip; every condition (amplitude, channel weights) gets its own trace and its own analysis with the
generator seeds of S-1. Nothing here changes an S-1 parameter or the S-1 verdict.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import run_s1 as S1  # noqa: E402
from dfd.rppg import pulse as P  # noqa: E402

F_INJ = 1.2
AMPS = (0.0, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032)
SIGS = {"pulse": (0.3, 1.0, 0.5), "green": (0.0, 1.0, 0.0)}      # (R, G, B) relative weights of the amplitude A
FREQ_TOL = 0.05
CONDS = [("control", 0.0)] + [(s, a) for s in SIGS for a in AMPS[1:]]   # (signature, amplitude); "control" = A 0


def skin_px(frame_bgr, box, p=P.PARAMS):
    """Skin pixels [n, 3] (BGR, uint8) of the central ROI, selected exactly as dfd.rppg.pulse.skin_rgb, or None."""
    import cv2
    H, W = frame_bgr.shape[:2]
    x, y, w, h = box
    x0 = int(round(x + p.roi_lo * w)); x1 = int(round(x + p.roi_hi * w))
    y0 = int(round(y + p.roi_lo * h)); y1 = int(round(y + p.roi_hi * h))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    roi = frame_bgr[y0:y1, x0:x1]
    ycc = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    m = ((ycc[..., 1] >= p.cr_lo) & (ycc[..., 1] <= p.cr_hi) & (ycc[..., 2] >= p.cb_lo) & (ycc[..., 2] <= p.cb_hi))
    if int(m.sum()) < max(p.min_skin_px, p.min_skin_frac * m.size):
        return None
    return roi[m]


def traces_for_clip(path, detector, start, n_frames, phi, rng, p=P.PARAMS):
    """Per-frame mean skin RGB for the control and for every (signature, amplitude). Same loop as dfd.rppg.pulse.extract_trace."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rgb = {c: np.full((n_frames, 3), np.nan) for c in CONDS}
    k, box = 0, None
    try:
        for _ in range(start):
            if not cap.grab():
                break
        while k < n_frames:
            ok, frame = cap.read()
            if not ok:
                break
            faces = detector.detect(frame)
            f = faces[0] if faces else None
            if f is not None and min(f.w, f.h) >= p.min_face_px:
                new = np.array([f.x, f.y, f.w, f.h], dtype=np.float64)
                if box is None or P._iou(box, new) < p.box_reset_iou:
                    box = new
                else:
                    box = (1.0 - p.box_ema) * box + p.box_ema * new
                px = skin_px(frame, box, p)
                if px is not None:
                    pxf = px.astype(np.float64)
                    rgb[CONDS[0]][k] = pxf.mean(axis=0)[::-1]           # the S-1 value (exact mean of the integer pixels)
                    s = np.sin(2.0 * np.pi * F_INJ * k / fps + phi)
                    u = rng.random(pxf.shape)                             # stochastic rounding, one draw per pixel and channel
                    for (sig, a) in CONDS[1:]:
                        w_bgr = np.array(SIGS[sig][::-1])
                        v = np.floor(pxf * (1.0 + a * s * w_bgr) + u)
                        rgb[(sig, a)][k] = np.clip(v, 0, 255).mean(axis=0)[::-1]
            k += 1
    finally:
        cap.release()
    return {c: dict(rgb=v[:k], fps=fps, n_read=k, n_total=n_total) for c, v in rgb.items()}


_DET = None


def _init():
    global _DET
    S1._init()
    _DET = S1._DET


def work(clip):
    t0 = time.perf_counter()
    import cv2
    cap = cv2.VideoCapture(str(S1.DATA / clip.path))
    n_total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    plan = P.segment_plan(n_total, fps)
    out = dict(dataset=clip.dataset, sample_id=clip.sample_id, ds=clip.ds, idx=clip.idx, conds={})
    if plan is None:
        return out
    prng = np.random.default_rng([P.PARAMS.seed, 11, clip.ds, clip.idx])
    phi = float(prng.uniform(0, 2 * np.pi))
    trs = traces_for_clip(S1.DATA / clip.path, S1._DET, plan[0], plan[1], phi, prng)
    if trs is None:
        return out
    for c, tr in trs.items():
        rng = np.random.default_rng([P.PARAMS.seed, 7, clip.ds, clip.idx])      # the generator of S-1
        r = P.analyse_clip(tr, rng)
        out["conds"][c] = dict(snr_db=r["snr_db"], peak_hz=r["peak_hz"], usable=bool(r["usable"]), reason=r["reason"], shuffled=r["shuffled"])
    out["seconds"] = time.perf_counter() - t0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    workers = max(1, min(8, a.workers))
    n_per = 3 if a.smoke else P.PARAMS.n_per_dataset
    out = Path(a.out) if a.out else ROOT / "data" / "_results" / ("M-005_smoke" if a.smoke else "M-005")
    out.mkdir(parents=True, exist_ok=True)
    clips = []
    for i, spec in enumerate(S1.DATASETS):
        c, _, _ = S1.sample(i, spec, n_per)
        clips += c
    print(f"{len(clips)} clips, {workers} workers", flush=True)
    t1, res = time.perf_counter(), []
    with mp.get_context("spawn").Pool(workers, initializer=_init) as pool:
        for k, r in enumerate(pool.imap_unordered(work, clips, chunksize=1), 1):
            res.append(r)
            if k % 10 == 0 or k == len(clips):
                print(f"  {k}/{len(clips)} clips, {time.perf_counter() - t1:.0f} s", flush=True)
    res.sort(key=lambda r: (r["ds"], r["idx"]))
    n = len(res)
    s1_thr = json.loads((ROOT / "data/_results/S-1/summary.json").read_text())["aggregate"]["threshold_db"]
    rows, summ = [], {}
    for c in CONDS:
        have = [r for r in res if c in r["conds"]]
        us = [r for r in have if r["conds"][c]["usable"]]
        null = np.concatenate([r["conds"][c]["shuffled"][np.isfinite(r["conds"][c]["shuffled"])] for r in us] or [np.zeros(0)])
        thr = float(np.percentile(null, 95)) if len(null) else float("nan")
        above = [r for r in us if r["conds"][c]["snr_db"] > thr]
        above_s1 = [r for r in us if r["conds"][c]["snr_db"] > s1_thr]
        near = [r for r in us if abs(r["conds"][c]["peak_hz"] - F_INJ) <= FREQ_TOL]
        both = [r for r in above if abs(r["conds"][c]["peak_hz"] - F_INJ) <= FREQ_TOL]
        lo, hi = P.wilson(len(above), n)
        summ["%s@%g" % c] = dict(signature=c[0], amplitude=c[1], n_sampled=n, n_usable=len(us), threshold_db=thr,
                                 above=len(above), fraction=len(above) / n, wilson95=[lo, hi],
                                 above_fixed_s1_threshold=len(above_s1), fraction_fixed=len(above_s1) / n,
                                 peak_within_tol=len(near), above_and_peak_within_tol=len(both),
                                 median_snr_db=float(np.median([r["conds"][c]["snr_db"] for r in us])) if us else float("nan"))
    ctl = [(r["sample_id"], r["conds"][CONDS[0]]["snr_db"], r["conds"][CONDS[0]]["peak_hz"]) for r in res if CONDS[0] in r["conds"]]
    ref = {}
    with open(ROOT / "data/_results/S-1/clips.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ref[row["sample_id"]] = (float(row["snr_db"]) if row["snr_db"] not in ("", "nan") else float("nan"),
                                     float(row["peak_hz"]) if row["peak_hz"] not in ("", "nan") else float("nan"))
    d_snr = max((abs(s - ref[i][0]) for i, s, _ in ctl if i in ref and np.isfinite(s) and np.isfinite(ref[i][0])), default=float("nan"))
    d_pk = max((abs(p_ - ref[i][1]) for i, _, p_ in ctl if i in ref and np.isfinite(p_) and np.isfinite(ref[i][1])), default=float("nan"))
    chk = dict(n_compared=sum(1 for i, *_ in ctl if i in ref), max_abs_diff_snr_db=d_snr, max_abs_diff_peak_hz=d_pk)
    summary = dict(experiment="M-005", smoke=a.smoke, date=time.strftime("%Y-%m-%d %H:%M:%S"), f_inj_hz=F_INJ, amplitudes=list(AMPS),
                   signatures=SIGS, freq_tol_hz=FREQ_TOL, s1_threshold_db=s1_thr, control_check_vs_S1=chk, conditions=summ,
                   wall_s=time.perf_counter() - t1, workers=workers)
    smallest = {}
    for sig in SIGS:
        ok = [summ["%s@%g" % (sig, am)] for am in AMPS[1:] if summ["%s@%g" % (sig, am)]["fraction"] >= 0.60]
        smallest[sig] = ok[0]["amplitude"] if ok else None
    summary["smallest_amplitude_with_fraction_at_least_0.60"] = smallest
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    md = ["### M-005 result block" + (" (SMOKE TEST, NOT A RESULT)" if a.smoke else ""), "",
          f"Control (A = 0) against S-1 on {chk['n_compared']} clips: max |difference| of the SNR {chk['max_abs_diff_snr_db']:.2e} dB, of the peak frequency {chk['max_abs_diff_peak_hz']:.2e} Hz.",
          f"Fraction of the {n} sampled clips recovered (SNR above the 95th percentile of the pooled shuffled control of the same condition; unusable clips count as not recovered):", "",
          "| signature | A (green) | threshold dB | recovered / sampled [Wilson 95%] | fixed S-1 threshold 1.118 dB | peak within 0.05 Hz of 1.2 Hz, of usable | recovered and peak within 0.05 Hz | median SNR dB |",
          "|---|---|---|---|---|---|---|---|"]
    for key, v in summ.items():
        md.append(f"| {v['signature']} | {v['amplitude']:g} | {v['threshold_db']:.3f} | {v['above']}/{v['n_sampled']} = {v['fraction']:.3f} [{v['wilson95'][0]:.3f}, {v['wilson95'][1]:.3f}] | "
                  f"{v['above_fixed_s1_threshold']}/{v['n_sampled']} = {v['fraction_fixed']:.3f} | {v['peak_within_tol']}/{v['n_usable']} | {v['above_and_peak_within_tol']} | {v['median_snr_db']:.2f} |")
    md += ["", "Smallest amplitude with a recovered fraction of at least 0.60: " + ", ".join(f"{k}: {('%g' % v) if v is not None else 'not reached up to %g' % AMPS[-1]}" for k, v in smallest.items()) + ".", ""]
    (out / "result_block.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"wall {summary['wall_s']:.0f} s; wrote {out}")


if __name__ == "__main__":
    main()
