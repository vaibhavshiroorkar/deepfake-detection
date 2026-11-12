"""Detector comparison on a fixed, repeatable list of random frames.

    # 1. draw the frame list (seeded; written to a CSV so the run can be repeated exactly)
    python -m dfd.prep.detcmp frames --manifest data/_manifests/ffpp.manifest.csv \
        --split-file data/_splits/ffpp-037e2976fade.csv --split val --n 300 --out data/_prep_bench/detcmp_frames_ffpp.csv
    # 2. run the three detectors on exactly those frames
    python -m dfd.prep.detcmp run --frames data/_prep_bench/detcmp_frames_ffpp.csv --tag ffpp

Frames only: no label or score is used in any result. The label and method columns of the
manifest are used only to balance the draw (half real, the other half spread evenly over the
manipulation methods present). Detector settings are the defaults of dfd.prep.face (D-034 is
not changed). Speed is single thread, in process, each frame timed REPEATS times per detector,
interleaved so that load changes hit all detectors alike; other processes were running.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import random
import statistics as st
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2  # noqa: E402

from dfd.prep import face as F  # noqa: E402
from dfd.prep import video as V  # noqa: E402

BENCH = Path("data/_prep_bench")
SEED = 20261024  # same seed value as the split files, so the draw is tied to one documented number
REPEATS = 3
COLORS = {"yunet": (0, 200, 0), "ultraface": (255, 80, 0), "scrfd": (0, 0, 255)}


def cmd_frames(a) -> None:
    rng = random.Random(a.seed)
    with open(a.split_file, newline="", encoding="utf-8") as fh:
        in_split = {r["sample_id"] for r in csv.DictReader(fh) if r["split"] == a.split}
    with open(a.manifest, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["sample_id"] in in_split and r.get("modality", "video") == "video"]
    if not rows:
        raise SystemExit(f"no manifest rows in split {a.split!r}")
    real = [r for r in rows if r["label"] == "0"]
    fake = [r for r in rows if r["label"] == "1"]
    methods = sorted({r["method"] for r in fake})
    n_real = min(a.n // 2, len(real))  # a split with fewer real videos than half gives the rest to the fakes
    n_fake = a.n - n_real
    quota = {"real": n_real}
    for i, m in enumerate(methods):  # 150 over 4 methods: 38, 38, 37, 37
        quota[m] = n_fake // len(methods) + (1 if i < n_fake % len(methods) else 0)
    groups = {"real": real, **{m: [r for r in fake if r["method"] == m] for m in methods}}
    out, skipped = [], {}
    for g, rs in groups.items():
        rs = sorted(rs, key=lambda r: r["sample_id"])
        rng.shuffle(rs)
        got = 0
        for r in rs:
            if got == quota[g]:
                break
            p = Path("data") / r["path"]
            if not V.is_stable(p, 20):
                skipped[g] = skipped.get(g, 0) + 1
                continue
            try:
                info = V.probe(p)
            except V.VideoOpenError:
                skipped[g] = skipped.get(g, 0) + 1
                continue
            fi = rng.randrange(info.n_frames)
            out.append(dict(sample_id=r["sample_id"], path=r["path"], method=g, frame_index=fi,
                            width=info.width, height=info.height, seed=a.seed))
            got += 1
        if got < quota[g]:
            print(f"warning: group {g}: only {got} of {quota[g]} usable videos")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    print(f"wrote {len(out)} frames ({len(set(o['sample_id'] for o in out))} distinct videos) to {a.out}; "
          f"quota {quota}; skipped (unstable or unreadable) {skipped}")


def _iou(a: F.Face, b: F.Face) -> float:
    iw = max(0.0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    ih = max(0.0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = iw * ih
    return inter / (a.area + b.area - inter + 1e-9)


def _q(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def cmd_run(a) -> None:
    cv2.setNumThreads(1)
    names = ["yunet", "ultraface", "scrfd"]
    dets = {n: F.make_detector(n, a.model_dir) for n in names}  # defaults of face.py
    for d in dets.values():
        d.detect(np.zeros((480, 640, 3), np.uint8))  # warm-up
    with open(a.frames, newline="", encoding="utf-8") as fh:
        frames = list(csv.DictReader(fh))
    try:
        import psutil
        load = psutil.cpu_percent(interval=3.0)
    except ImportError:
        load = None
    print(f"frames {len(frames)} from {a.frames}; system CPU load before run: {load}%")
    res = {n: [] for n in names}   # per frame: list of faces
    times = {n: [] for n in names}
    imgs = []
    for k, r in enumerate(frames):
        got = V.read_frames(Path("data") / r["path"], indices=[int(r["frame_index"])], backend="cv_seq")
        if not got:
            raise SystemExit(f"cannot decode frame {r['frame_index']} of {r['sample_id']}")
        fr = got[0][1]
        imgs.append(fr)
        for n, d in dets.items():
            ts = []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                faces = d.detect(fr)
                ts.append(time.perf_counter() - t0)
            res[n].append(faces)
            times[n].append(st.mean(ts) * 1000)
    N = len(frames)
    print(f"\n[{a.tag}] per detector (all {N} frames; speed = mean of {REPEATS} timings per frame)")
    print("detector    frames_with_face  ms/frame mean  median  p90   frames with >1 box")
    for n in names:
        found = sum(len(f) > 0 for f in res[n])
        multi = sum(len(f) > 1 for f in res[n])
        t = times[n]
        print(f"{n:10s}  {found:4d} ({100 * found / N:6.2f}%)  {st.mean(t):13.2f}  {st.median(t):6.2f} {_q(t, .9):5.2f}  {multi} ({100 * multi / N:.1f}%)")
    print("\npairwise on the LARGEST box of each detector")
    print("pair                 both  A_only B_only neither  disagree_on_existence  mean_IoU(both)  IoU>=0.5(of both)  "
          "A<0.5*B  B<0.5*A")
    detail = []
    for A_, B_ in itertools.combinations(names, 2):
        both = ao = bo = nei = 0
        ious, small_a, small_b = [], 0, 0
        for k in range(N):
            a_, b_ = F.largest(res[A_][k]), F.largest(res[B_][k])
            if a_ and b_:
                both += 1
                ious.append(_iou(a_, b_))
                small_a += a_.area < 0.5 * b_.area
                small_b += b_.area < 0.5 * a_.area
            elif a_:
                ao += 1
            elif b_:
                bo += 1
            else:
                nei += 1
        ge = sum(i >= 0.5 for i in ious)
        print(f"{A_ + ' vs ' + B_:20s} {both:4d}  {ao:5d}  {bo:5d}  {nei:5d}   {ao + bo:4d} ({100 * (ao + bo) / N:5.2f}%)"
              f"   {st.mean(ious) if ious else float('nan'):10.3f}   {ge:4d} ({100 * ge / max(1, both):5.1f}%)"
              f"   {small_a:3d} ({100 * small_a / max(1, both):4.1f}%)  {small_b:3d} ({100 * small_b / max(1, both):4.1f}%)")
        ratios, shifts = [], []
        for k in range(N):
            a_, b_ = F.largest(res[A_][k]), F.largest(res[B_][k])
            if a_ and b_:
                ratios.append(b_.area / a_.area)
                shifts.append(float(np.hypot((b_.x + b_.w / 2) - (a_.x + a_.w / 2), (b_.y + b_.h / 2) - (a_.y + a_.h / 2)))
                              / max(a_.w, a_.h))
        print(f"   {B_} box area / {A_} box area: median {st.median(ratios):.3f}, p10 {_q(ratios, .1):.3f}, p90 {_q(ratios, .9):.3f}; "
              f"centre shift as a fraction of {A_} box side: median {st.median(shifts):.3f}, p90 {_q(shifts, .9):.3f}")
        detail.append((A_, B_))
    # raw per-frame rows
    rawf = BENCH / f"detcmp_raw_{a.tag}.csv"
    with open(rawf, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["frame_row", "sample_id", "frame_index", "detector", "n_faces", "x", "y", "w", "h", "score", "ms"])
        for k, r in enumerate(frames):
            for n in names:
                f = F.largest(res[n][k])
                w.writerow([k, r["sample_id"], r["frame_index"], n, len(res[n][k])] +
                           ([f"{f.x:.1f}", f"{f.y:.1f}", f"{f.w:.1f}", f"{f.h:.1f}", f"{f.score:.3f}"] if f else ["", "", "", "", ""]) +
                           [f"{times[n][k]:.2f}"])
    # disagreement frames: existence disagreement, or any pair IoU < 0.5, or an area ratio below 0.5
    dis = []
    for k in range(N):
        tops = {n: F.largest(res[n][k]) for n in names}
        why = []
        if len({t is not None for t in tops.values()}) > 1:
            why.append("existence")
        for A_, B_ in itertools.combinations(names, 2):
            if tops[A_] and tops[B_]:
                if _iou(tops[A_], tops[B_]) < 0.5:
                    why.append(f"iou<0.5 {A_}/{B_}")
        if why:
            dis.append((k, why))
    print(f"\nframes with any disagreement (existence or IoU<0.5): {len(dis)} of {N}; raw rows {rawf}")
    if len(dis) < a.max_images:
        # also show the frames with the lowest agreement (smallest pairwise IoU) so that something is always visible
        chosen = {k for k, _ in dis}
        def min_iou(k):
            tops = [F.largest(res[n][k]) for n in names]
            return min((_iou(x, y) for x, y in itertools.combinations(tops, 2) if x and y), default=1.0)
        extra = sorted((k for k in range(N) if k not in chosen), key=min_iou)[: min(8, a.max_images - len(dis))]
        dis += [(k, [f"lowest agreement, min pairwise IoU {min_iou(k):.2f}"]) for k in extra]
    tiles = []
    for k, why in dis[: a.max_images]:
        im = imgs[k].copy()
        t = max(2, im.shape[0] // 200)
        for n in names:
            for i, f in enumerate(res[n][k]):
                top = f is F.largest(res[n][k])
                cv2.rectangle(im, (int(f.x), int(f.y)), (int(f.x + f.w), int(f.y + f.h)), COLORS[n], t * (2 if top else 1))
        h, w = im.shape[:2]
        s = 360 / max(h, w)
        tiles.append(cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA))
        print(f"  tile {len(tiles) - 1}: {frames[k]['sample_id']} frame {frames[k]['frame_index']} ({'; '.join(why)})")
    if tiles:
        th, tw = 360, 360
        pad = []
        for t in tiles:
            c = np.zeros((th, tw, 3), np.uint8)
            c[: t.shape[0], : t.shape[1]] = t
            pad.append(c)
        while len(pad) % 4:
            pad.append(np.zeros((th, tw, 3), np.uint8))
        sheet = np.vstack([np.hstack(pad[i:i + 4]) for i in range(0, len(pad), 4)])
        outp = BENCH / f"detcmp_disagree_{a.tag}.jpg"
        cv2.imwrite(str(outp), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"saved {outp} (green = yunet, orange/blue = ultraface, red = scrfd; thick box = largest of that detector)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frames")
    f.add_argument("--manifest", required=True)
    f.add_argument("--split-file", required=True)
    f.add_argument("--split", required=True)
    f.add_argument("--n", type=int, default=300)
    f.add_argument("--seed", type=int, default=SEED)
    f.add_argument("--out", required=True)
    f.set_defaults(fn=cmd_frames)
    r = sub.add_parser("run")
    r.add_argument("--frames", required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--model-dir", default=str(F.DEFAULT_MODEL_DIR))
    r.add_argument("--max-images", type=int, default=24)
    r.set_defaults(fn=cmd_run)
    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
