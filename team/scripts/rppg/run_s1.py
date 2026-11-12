"""Signal-sanity check S-1 (docs/ITERATIONS.md section I-7): can a pulse be recovered from our real videos at all?

    PYTHONPATH=src .venv/Scripts/python.exe scripts/rppg/run_s1.py                # the full run, 150 real clips per dataset
    PYTHONPATH=src .venv/Scripts/python.exe scripts/rppg/run_s1.py --smoke        # 3 clips per dataset, output in S-1_smoke

CPU only (CUDA is hidden before any import), at most 8 worker processes. Real clips only, from splits that are never
final: FF++ c23 train and val (original sequences), Celeb-DF v2 dev_x (real), LAV-DF official train (n_fakes = 0).
The design is fixed in dfd.rppg.pulse.S1Params and in the spec (s1_spec.md); the pass rule is applied mechanically:

    threshold = 95th percentile of the POOLED shuffled-control SNR values (all usable clips x n_shuffles)
    S-1 passes if (pooled over the three datasets) at least 60 percent of the SAMPLED real clips have a peak SNR
    above the threshold (clips that are unusable count as not above; the share among usable clips is reported too).

Outputs in --out (default data/_results/S-1): clips.csv, clips.npz (shuffled SNR values), traces.npz (filled RGB
traces), summary.json, result_block.md.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU only: the GPU belongs to another job
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dfd.rppg import pulse as P  # noqa: E402

DATA = ROOT / "data"
ALLOWED_SPLITS = {"train", "val", "dev_x"}       # never test_in or test_x (CLAUDE.md rule 3)
DATASETS = (
    dict(name="ffpp", manifest="_manifests/ffpp.manifest.csv", split_csv="_splits/ffpp-037e2976fade.csv", splits=("train", "val")),
    dict(name="celebdf_v2", manifest="_manifests/celebdf_v2.manifest.csv", split_csv="_splits/celebdf_v2-82a3a70594e9.csv", splits=("dev_x",)),
    dict(name="lavdf", metadata="lavdf/LAV-DF/metadata.min.json", official_split="train"),
)
LAVDF_MIN_FRAMES = 250                           # round(10 s * 25 fps); LAV-DF is 25 fps (checked on the sampled files)


@dataclass
class Clip:
    ds: int
    dataset: str
    idx: int                                     # position among the sampled clips of the dataset
    sample_id: str
    path: str                                    # relative to data/
    group_id: str


def _probe(rel: str):
    import cv2
    c = cv2.VideoCapture(str(DATA / rel))
    n, f = int(c.get(cv2.CAP_PROP_FRAME_COUNT)), float(c.get(cv2.CAP_PROP_FPS))
    c.release()
    return n, f


def list_candidates(spec: dict):
    """Real clips of the allowed splits, sorted by sample_id: list of (sample_id, path, group_id) and the eligible subset."""
    if spec["name"] == "lavdf":
        meta = json.loads((DATA / spec["metadata"]).read_text(encoding="utf-8"))
        rows = [m for m in meta if m["split"] == spec["official_split"] and m["n_fakes"] == 0
                and not m["modify_video"] and not m["modify_audio"] and m["original"] is None]
        cand = sorted(((f"lavdf:{m['file']}", f"lavdf/LAV-DF/{m['file']}", f"lavdf:{m['file']}", m["video_frames"]) for m in rows))
        elig = [c[:3] for c in cand if c[3] >= LAVDF_MIN_FRAMES]
        return [c[:3] for c in cand], elig
    from dfd.data import loaders as L           # imports torch (CPU use only, nothing is moved to a device)
    cand = []
    for sp in spec["splits"]:
        assert sp in ALLOWED_SPLITS
        for r in L.select_records(DATA / spec["manifest"], DATA / spec["split_csv"], sp, modality="video"):
            if r.label == 0:
                cand.append((r.sample_id, r.path, r.group_id))
    cand.sort()
    with ThreadPoolExecutor(8) as ex:
        probes = list(ex.map(lambda c: _probe(c[1]), cand))
    elig = [c for c, (n, f) in zip(cand, probes) if P.segment_plan(n, f) is not None]
    return cand, elig


def sample(spec_idx: int, spec: dict, n: int):
    cand, elig = list_candidates(spec)
    rng = np.random.default_rng(P.PARAMS.seed)
    take = np.sort(rng.permutation(len(elig))[:n])
    clips = [Clip(spec_idx, spec["name"], i, *elig[j]) for i, j in enumerate(take)]
    return clips, len(cand), len(elig)


# ---------------------------------------------------------------- worker

_DET = None


def _init():
    global _DET
    import cv2
    cv2.setNumThreads(1)
    from dfd.prep import face as F
    _DET = F.make_detector("yunet", ROOT / F.DEFAULT_MODEL_DIR)


def work(clip: Clip) -> dict:
    t0 = time.perf_counter()
    import cv2
    cap = cv2.VideoCapture(str(DATA / clip.path))
    n_total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    plan = P.segment_plan(n_total, fps)
    rng = np.random.default_rng([P.PARAMS.seed, 7, clip.ds, clip.idx])
    if plan is None:
        tr = dict(rgb=np.zeros((0, 3)), fps=fps, n_read=0, n_total=n_total, width=0, height=0, error="short_file")
    else:
        tr = P.extract_trace(DATA / clip.path, _DET, plan[0], plan[1])
    res = P.analyse_clip(tr, rng)
    res.update(dataset=clip.dataset, sample_id=clip.sample_id, group_id=clip.group_id, path=clip.path, ds=clip.ds,
               idx=clip.idx, start=plan[0] if plan else -1, seconds=time.perf_counter() - t0)
    if "filled" in res:
        res["filled"] = res["filled"].astype(np.float32)
    return res


# ---------------------------------------------------------------- aggregation

def frac_block(k: int, n: int) -> dict:
    lo, hi = P.wilson(k, n)
    return dict(k=int(k), n=int(n), fraction=(k / n if n else float("nan")), wilson95=[lo, hi])


def aggregate(results: list[dict], avail: dict, p: P.S1Params) -> dict:
    pooled_null = np.concatenate([r["shuffled"][np.isfinite(r["shuffled"])] for r in results if r["usable"]] or [np.zeros(0)])
    thr = float(np.percentile(pooled_null, p.percentile)) if len(pooled_null) else float("nan")
    for r in results:
        r["above"] = bool(r["usable"] and np.isfinite(thr) and r["snr_db"] > thr)
    out = dict(threshold_db=thr, n_pooled_null=int(len(pooled_null)),
               null_quantiles_db={q: float(np.percentile(pooled_null, q)) for q in (5, 25, 50, 75, 95, 99)} if len(pooled_null) else {},
               per_dataset={})
    names = [d["name"] for d in DATASETS]
    for nm in names + ["pooled"]:
        rs = results if nm == "pooled" else [r for r in results if r["dataset"] == nm]
        if not rs:
            continue
        us = [r for r in rs if r["usable"]]
        reasons = {}
        for r in rs:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        snr = np.array([r["snr_db"] for r in us])
        d = dict(n_sampled=len(rs), n_usable=len(us), unusable_reasons={k: v for k, v in reasons.items() if k != "ok"},
                 primary_above_over_sampled=frac_block(sum(r["above"] for r in rs), len(rs)),
                 above_over_usable=frac_block(sum(r["above"] for r in us), len(us)),
                 n_groups_sampled=len({r["group_id"] for r in rs}),
                 real_snr_db_quantiles=({q: float(np.percentile(snr, q)) for q in (5, 25, 50, 75, 95)} if len(snr) else {}),
                 median_face_share=float(np.median([r["face_share"] for r in rs])),
                 mean_seconds_per_clip=float(np.mean([r["seconds"] for r in rs])))
        if nm != "pooled":
            d.update(avail[nm])
        out["per_dataset"][nm] = d
    pr = out["per_dataset"]["pooled"]["primary_above_over_sampled"]
    out["pass_rule"] = dict(rule=f"pooled share of sampled real clips with SNR above the {p.percentile:g}th percentile of the pooled "
                                 f"shuffled control is at least {p.pass_fraction}",
                            pooled_fraction=pr["fraction"], required=p.pass_fraction,
                            S1_passes=bool(np.isfinite(thr) and pr["fraction"] >= p.pass_fraction))
    return out


def fmt(fb: dict) -> str:
    return f"{fb['k']}/{fb['n']} = {fb['fraction']:.3f} [{fb['wilson95'][0]:.3f}, {fb['wilson95'][1]:.3f}]"


def result_block(summary: dict, smoke: bool) -> str:
    L = []
    L.append("### S-1 result block" + (" (SMOKE TEST, NOT A RESULT)" if smoke else ""))
    L.append("")
    thr = summary["aggregate"]["threshold_db"]
    L.append(f"Threshold (95th percentile of the pooled shuffled SNR, {summary['aggregate']['n_pooled_null']} values): {thr:.3f} dB.")
    L.append("")
    L.append("| Dataset | Eligible / candidates | Sampled | Usable | Above threshold, of sampled (Wilson 95%) | Above, of usable (Wilson 95%) | Median real SNR dB (usable) |")
    L.append("|---|---|---|---|---|---|---|")
    for nm, d in summary["aggregate"]["per_dataset"].items():
        el = f"{d['n_eligible']} / {d['n_candidates']}" if "n_eligible" in d else "all"
        med = d["real_snr_db_quantiles"].get(50, float("nan")) if d["real_snr_db_quantiles"] else float("nan")
        L.append(f"| {nm} | {el} | {d['n_sampled']} | {d['n_usable']} | {fmt(d['primary_above_over_sampled'])} | {fmt(d['above_over_usable'])} | {med:.3f} |")
    L.append("")
    pr = summary["aggregate"]["pass_rule"]
    L.append(f"Pass rule (mechanical): {pr['rule']}. Pooled fraction {pr['pooled_fraction']:.3f}, required {pr['required']}: "
             f"S-1 passes = {pr['S1_passes']}.")
    L.append("")
    L.append("Unusable clips (counted as not above the threshold): " + json.dumps(
        {nm: d["unusable_reasons"] for nm, d in summary["aggregate"]["per_dataset"].items()}) + ".")
    return "\n".join(L) + "\n"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="3 clips per dataset, output to data/_results/S-1_smoke (not a result)")
    ap.add_argument("--workers", type=int, default=8, help="worker processes (at most 8)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    workers = max(1, min(8, a.workers))
    p = P.PARAMS
    n_per = 3 if a.smoke else p.n_per_dataset
    out = Path(a.out) if a.out else DATA / "_results" / ("S-1_smoke" if a.smoke else "S-1")
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    clips, avail = [], {}
    for i, spec in enumerate(DATASETS):
        c, n_cand, n_elig = sample(i, spec, n_per)
        clips += c
        avail[spec["name"]] = dict(n_candidates=n_cand, n_eligible=n_elig, n_target=n_per)
        print(f"{spec['name']}: candidates {n_cand}, eligible {n_elig}, sampled {len(c)}", flush=True)
    print(f"sampling took {time.perf_counter() - t0:.1f} s; {len(clips)} clips, {workers} workers", flush=True)

    t1 = time.perf_counter()
    results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init) as pool:
        for k, r in enumerate(pool.imap_unordered(work, clips, chunksize=1), 1):
            results.append(r)
            if k % 10 == 0 or k == len(clips):
                print(f"  {k}/{len(clips)} clips, {time.perf_counter() - t1:.0f} s", flush=True)
    wall = time.perf_counter() - t1
    results.sort(key=lambda r: (r["ds"], r["idx"]))

    agg = aggregate(results, avail, p)
    thr = agg["threshold_db"]

    cols = ["dataset", "sample_id", "group_id", "path", "fps", "n_frames", "start_frame", "face_share", "max_gap_frames",
            "peak_hz", "bpm", "snr_db", "usable", "reason", "above_threshold", "shuffled_mean_db", "shuffled_max_db", "seconds"]
    with open(out / "clips.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in results:
            sh = r["shuffled"]
            w.writerow([r["dataset"], r["sample_id"], r["group_id"], r["path"], f"{r['fps']:.4f}", r["n_frames"], r["start"],
                        f"{r['face_share']:.4f}", r["max_gap_frames"], f"{r['peak_hz']:.5f}", f"{r['bpm']:.3f}",
                        f"{r['snr_db']:.5f}", int(r["usable"]), r["reason"], int(r["above"]),
                        f"{np.nanmean(sh):.5f}" if np.isfinite(sh).any() else "", f"{np.nanmax(sh):.5f}" if np.isfinite(sh).any() else "",
                        f"{r['seconds']:.2f}"])
    np.savez(out / "clips.npz", sample_id=np.array([r["sample_id"] for r in results]), dataset=np.array([r["dataset"] for r in results]),
             snr_db=np.array([r["snr_db"] for r in results]), usable=np.array([r["usable"] for r in results]),
             shuffled_snr_db=np.stack([r["shuffled"] for r in results]), threshold_db=thr)
    np.savez_compressed(out / "traces.npz", **{f"t{k:04d}": r["filled"] for k, r in enumerate(results) if "filled" in r},
                        index=np.array([r["sample_id"] for r in results]), fps=np.array([r["fps"] for r in results]))

    from dfd.prep import face as F
    summary = dict(experiment="S-1 rPPG signal-sanity check", smoke=a.smoke,
                   status=("SMOKE TEST, NOT A RESULT" if a.smoke else "full run"),
                   date=time.strftime("%Y-%m-%d %H:%M:%S"), params=P.params_dict(p), seed=p.seed,
                   datasets=[{k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()} for d in DATASETS],
                   detector=dict(name="yunet", model_sha256=sha256(ROOT / F.DEFAULT_MODEL_DIR / F.MODEL_FILES["yunet"])),
                   env=dict(python=sys.version.split()[0], numpy=np.__version__, platform=platform.platform(), workers=workers,
                            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")),
                   timing=dict(wall_s=wall, clips=len(results), wall_s_per_clip=wall / max(1, len(results)),
                               cpu_s_per_clip_mean=float(np.mean([r["seconds"] for r in results]))),
                   aggregate=agg)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    (out / "result_block.md").write_text(result_block(summary, a.smoke), encoding="utf-8")
    print((out / "result_block.md").read_text(encoding="utf-8"))
    print(f"wall {wall:.1f} s for {len(results)} clips ({wall / len(results):.2f} s per clip wall, "
          f"{summary['timing']['cpu_s_per_clip_mean']:.2f} s per clip per worker); wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
