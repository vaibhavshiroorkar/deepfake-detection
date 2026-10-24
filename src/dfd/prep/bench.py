"""Speed and quality measurements for the data-preparation tools.

    python -m dfd.prep.bench sample --out data/_prep_bench/sample_c23.csv --per-method 25
    python -m dfd.prep.bench decode --manifest data/_prep_bench/sample_c23.csv --every 3
    python -m dfd.prep.bench detect --manifest data/_prep_bench/sample_c23.csv
    python -m dfd.prep.bench audio

Every command writes its raw numbers as CSV under data/_prep_bench/ and prints a
summary. These are indicative numbers: other processes (downloads, archive
extraction) are using CPU and disk while they run. Each output records the CPU
load measured just before the run.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2  # noqa: E402

from dfd.prep import audio as A  # noqa: E402
from dfd.prep import face as F  # noqa: E402
from dfd.prep import video as V  # noqa: E402

BENCH = Path("data/_prep_bench")
FFPP = Path("data/ffpp")
METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def _load_avg(seconds: float = 3.0) -> float | None:
    try:
        import psutil
        return psutil.cpu_percent(interval=seconds)
    except ImportError:
        return None


def ffprobe_ok(path: Path) -> bool:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_frames,width,height", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() != ""


def _q(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


# ------------------------------------------------------------------ sample
def cmd_sample(a) -> None:
    rng = random.Random(a.seed)
    groups = {"real": FFPP / "original_sequences/youtube/c23/videos"}
    for m in METHODS:
        groups[m] = FFPP / f"manipulated_sequences/{m}/c23/videos"
    n_for = {"real": a.per_method * len(METHODS)}
    n_for.update({m: a.per_method for m in METHODS})
    rows = []
    for g, d in groups.items():
        files = sorted(p for p in d.glob("*.mp4") if V.is_stable(p, 20))
        rng.shuffle(files)
        picked = []
        for p in files:
            if len(picked) == n_for[g]:
                break
            if ffprobe_ok(p):
                picked.append(p)
        if len(picked) < n_for[g]:
            print(f"warning: only {len(picked)} complete files for {g}", file=sys.stderr)
        for p in sorted(picked):
            rows.append(dict(sample_id=f"{g}_{p.stem}", path=p.relative_to("data").as_posix(),
                             label=0 if g == "real" else 1, method=g))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample_id", "path", "label", "method"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.out}: " + ", ".join(
        f"{g}={sum(r['method'] == g for r in rows)}" for g in groups))


def _read(manifest):
    """Manifest rows; relative paths are relative to data/ (as in the data manifests) and returned resolved."""
    with open(manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        if not Path(r["path"]).is_absolute():
            r["path"] = str(Path("data") / r["path"])
    return rows


# ------------------------------------------------------------------ decode
def cmd_decode(a) -> None:
    cv2.setNumThreads(1)
    rows = _read(a.manifest)[:: a.every]
    backends = a.backends
    load = _load_avg()
    print(f"system CPU load before run: {load}%; videos {len(rows)}; frames per video {a.n}; threads {a.threads}")
    out = []
    mism = 0
    for k, r in enumerate(rows):
        order = backends[k % len(backends):] + backends[:k % len(backends)]  # rotate to balance cold reads
        ref = None
        info = V.probe(r["path"])
        for b in order:
            t0 = time.perf_counter()
            frames = V.read_frames(r["path"], n=a.n, backend=b, threads=a.threads, info=info)
            dt = time.perf_counter() - t0
            out.append(dict(sample_id=r["sample_id"], backend=b, seconds=dt, n_frames=len(frames),
                            width=info.width, height=info.height, container_frames=info.n_frames))
            if ref is None:
                ref = frames
            else:
                same = len(ref) == len(frames) and all(
                    i == j and np.array_equal(x, y) for (i, x), (j, y) in zip(ref, frames))
                mism += 0 if same else 1
    # also include probe cost (header read) once per video, backend independent
    t0 = time.perf_counter()
    for r in rows:
        V.probe(r["path"])
    probe_s = (time.perf_counter() - t0) / len(rows)
    BENCH.mkdir(parents=True, exist_ok=True)
    f = BENCH / f"decode_threads{a.threads}{a.tag}.csv"
    with open(f, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    print(f"raw rows: {f}; header probe (warm): {probe_s * 1000:.1f} ms per video; "
          f"videos where backends gave different pixels or indices: {mism}")
    print("backend   videos  mean_s/video  median  p90    ms/frame   frames_total")
    for b in backends:
        s = [o["seconds"] for o in out if o["backend"] == b]
        nf = sum(o["n_frames"] for o in out if o["backend"] == b)
        print(f"{b:8s} {len(s):6d}  {st.mean(s):11.3f}  {st.median(s):6.3f}  {_q(s, .9):5.3f}  "
              f"{1000 * sum(s) / nf:9.2f}   {nf}")


# ------------------------------------------------------------------ detect
def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    return inter / (a.area + b.area - inter + 1e-9)


def cmd_detect(a) -> None:
    cv2.setNumThreads(1)
    rows = _read(a.manifest)[:: a.every]
    specs = a.detectors
    dets = {}
    for s in specs:
        if s == "yunet_full":
            dets[s] = F.YuNet(Path(a.model_dir) / F.MODEL_FILES["yunet"], max_side=None)
        else:
            dets[s] = F.make_detector(s, a.model_dir)
    load = _load_avg()
    print(f"system CPU load before run: {load}%; videos {len(rows)}; frames per video {a.n}")
    for d in dets.values():  # warm-up (first call allocates)
        d.detect(np.zeros((480, 640, 3), np.uint8))
    recs = []
    fails = []
    for r in rows:
        try:
            frames = V.read_frames(r["path"], n=a.n, backend="cv_seq")
        except Exception as e:  # noqa: BLE001
            fails.append((r["sample_id"], str(e)))
            continue
        for k, (src, fr) in enumerate(frames):
            h, w = fr.shape[:2]
            for name, d in dets.items():
                t0 = time.perf_counter()
                faces = d.detect(fr)
                dt = time.perf_counter() - t0
                top = F.largest(faces)
                recs.append(dict(sample_id=r["sample_id"], method=r["method"], label=r["label"], frame=k, src=src,
                                 w=w, h=h, detector=name, n_faces=len(faces), seconds=dt,
                                 x=top.x if top else "", y=top.y if top else "", bw=top.w if top else "",
                                 bh=top.h if top else "", score=top.score if top else ""))
    BENCH.mkdir(parents=True, exist_ok=True)
    f = BENCH / f"detect_frames{a.tag}.csv"
    with open(f, "w", newline="", encoding="utf-8") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(recs[0]), lineterminator="\n")
        w_.writeheader()
        w_.writerows(recs)
    print(f"raw rows: {f}; videos that failed to decode: {len(fails)} {fails[:3]}")
    summarise_detect(recs, specs)


def summarise_detect(recs, specs) -> None:
    by = {s: [r for r in recs if r["detector"] == s] for s in specs}
    n = len(by[specs[0]])
    print(f"\nframes: {n}")
    print("detector      det_rate  real   manip  multi_face  ms/frame(mean)  median  p90   mean_face_px")
    for s in specs:
        rs = by[s]
        det = [r for r in rs if r["n_faces"] > 0]
        real = [r for r in rs if r["label"] == "0"]
        man = [r for r in rs if r["label"] == "1"]
        rate = lambda xs: 100 * sum(r["n_faces"] > 0 for r in xs) / max(1, len(xs))  # noqa: E731
        multi = 100 * sum(r["n_faces"] > 1 for r in rs) / len(rs)
        t = [r["seconds"] * 1000 for r in rs]
        fp = np.mean([max(r["bw"], r["bh"]) for r in det]) if det else float("nan")
        print(f"{s:12s} {rate(rs):8.2f} {rate(real):6.2f} {rate(man):6.2f} {multi:10.2f}  {st.mean(t):14.2f}  "
              f"{st.median(t):6.2f} {_q(t, .9):5.2f}  {fp:10.1f}")
    print("\ndetection rate by manipulation method (percent of frames with at least one face)")
    methods = sorted({r["method"] for r in recs})
    print("detector      " + "  ".join(f"{m:>14s}" for m in methods))
    for s in specs:
        print(f"{s:12s}  " + "  ".join(
            f"{100 * sum(r['n_faces'] > 0 for r in by[s] if r['method'] == m) / max(1, sum(r['method'] == m for r in by[s])):14.2f}"
            for m in methods))
    # agreement of the largest face between detector pairs
    print("\nagreement (proxy, no ground truth): share of frames where both detect and the largest boxes have IoU >= 0.5")
    idx = {s: {(r["sample_id"], r["frame"]): r for r in by[s]} for s in specs}

    def box(r):
        return F.Face(r["x"], r["y"], r["bw"], r["bh"], r["score"])
    for i, s1 in enumerate(specs):
        for s2 in specs[i + 1:]:
            both = agree = only1 = only2 = 0
            for key, r1 in idx[s1].items():
                r2 = idx[s2][key]
                d1, d2 = r1["n_faces"] > 0, r2["n_faces"] > 0
                if d1 and d2:
                    both += 1
                    agree += _iou(box(r1), box(r2)) >= 0.5
                elif d1:
                    only1 += 1
                elif d2:
                    only2 += 1
            print(f"{s1} vs {s2}: both {both}, IoU>=0.5 {agree} ({100 * agree / max(1, both):.1f}% of both), "
                  f"only {s1} {only1}, only {s2} {only2}, neither {n - both - only1 - only2}")
    # time by resolution
    print("\nms/frame by frame height")
    heights = sorted({r["h"] for r in recs})
    for hh in heights:
        line = f"h={hh:5d} frames={sum(r['h'] == hh for r in by[specs[0]]):5d}  "
        for s in specs:
            t = [r["seconds"] * 1000 for r in by[s] if r["h"] == hh]
            line += f"{s}={st.mean(t):.2f}  "
        print(line)


# ------------------------------------------------------------------ decode throughput with a pool
def _decode_one(args):
    path, n, backend = args
    cv2.setNumThreads(1)
    t0 = time.perf_counter()
    fr = V.read_frames(path, n=n, backend=backend, threads=1)
    return len(fr), time.perf_counter() - t0


def cmd_decodepool(a) -> None:
    from concurrent.futures import ProcessPoolExecutor
    rows = _read(a.manifest)
    paths = [r["path"] for r in rows] * a.repeat
    print(f"system CPU load before run: {_load_avg()}%; videos {len(paths)} ({len(rows)} files x {a.repeat}); workers {a.workers}")
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(_decode_one, [(p, a.n, a.backend) for p in paths[: a.workers * 2]]))  # warm-up
        t0 = time.perf_counter()
        res = list(pool.map(_decode_one, [(p, a.n, a.backend) for p in paths], chunksize=1))
        wall = time.perf_counter() - t0
    nf = sum(r[0] for r in res)
    print(f"decode only, {a.backend}: {len(paths) / wall:.2f} videos/s, {nf / wall:.1f} frames/s, "
          f"{wall / len(paths):.4f} s/video wall, mean in-worker {st.mean(r[1] for r in res):.3f} s/video")


# ------------------------------------------------------------------ jpeg size and fidelity
def cmd_jpeg(a) -> None:
    cv2.setNumThreads(1)
    det = F.make_detector("yunet", a.model_dir)
    rows = _read(a.manifest)[:: a.every]
    crops = []
    for r in rows:
        for _, fr in V.read_frames(r["path"], n=a.n, backend="cv_seq"):
            c, f, _ = F.crop_largest_face(fr, det.detect(fr), F.DEFAULT_ENLARGE, a.size)
            if c is not None:
                crops.append(c)
    print(f"crops: {len(crops)} from {len(rows)} videos, size {a.size}x{a.size}")
    print("quality  mean_bytes/crop  mean_PSNR_dB_vs_uncompressed_crop")
    for q in a.qualities:
        sz, ps = [], []
        for c in crops:
            ok, buf = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, q])
            sz.append(len(buf))
            ps.append(cv2.PSNR(c, cv2.imdecode(buf, 1)))
        print(f"{q:7d}  {st.mean(sz):15.0f}  {st.mean(ps):10.2f}")


# ------------------------------------------------------------------ stress
def _transforms():
    def canvas_small(img, s):
        h, w = img.shape[:2]
        c = np.zeros_like(img)
        small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        c[:small.shape[0], :small.shape[1]] = small
        return c
    return {
        "identity": lambda x: x,
        "rot90": lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
        "rot180": lambda x: cv2.rotate(x, cv2.ROTATE_180),
        "blur_sigma4": lambda x: cv2.GaussianBlur(x, (0, 0), 4),
        "scale_0.25": lambda x: canvas_small(x, 0.25),
        "scale_0.125": lambda x: canvas_small(x, 0.125),
        "jpeg_q10": lambda x: cv2.imdecode(cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, 10])[1], 1),
        "dark_x0.15": lambda x: (x.astype(np.float32) * 0.15).astype(np.uint8),
        "left_half_only": lambda x: np.ascontiguousarray(x[:, : x.shape[1] // 2]),
    }


def cmd_stress(a) -> None:
    cv2.setNumThreads(1)
    dets = {s: F.make_detector(s, a.model_dir) for s in a.detectors}
    rows = _read(a.manifest)
    rng = random.Random(1)
    # 1 frame per video (the middle sampled frame)
    frames = []
    for r in rows:
        fr = V.read_frames(r["path"], n=3, backend="cv_seq")
        frames.append(fr[1][1])
    print(f"stress frames: {len(frames)} (one mid-video frame of each manifest video)")
    print("transform        " + "  ".join(f"{s:>10s}" for s in dets))
    for name, fn in _transforms().items():
        line = f"{name:16s} "
        for s, d in dets.items():
            line += f"  {100 * sum(len(d.detect(fn(f))) > 0 for f in frames) / len(frames):9.1f}%"
        print(line)
    # images without faces
    imgs = []
    root = Path("data/forensynths/cnn_synth_subset")
    for pat in ["biggan/0_real/*.png", "cyclegan/horse/0_real/*.png", "cyclegan/zebra/0_real/*.png",
                "cyclegan/apple/0_real/*.png", "cyclegan/orange/0_real/*.png"]:
        files = sorted(root.glob(pat))
        imgs += rng.sample(files, min(a.n_noface // 5, len(files)))
    fp = {s: [] for s in dets}
    for p in imgs:
        im = cv2.imread(str(p))
        if im is None:
            continue
        for s, d in dets.items():
            if d.detect(im):
                fp[s].append(str(p))
    print("\nimages without a face (ImageNet-style objects, animals, landscapes; a few may still contain a person): "
          f"{len(imgs)} images")
    for s in dets:
        print(f"  {s}: {len(fp[s])} images with at least one detection ({100 * len(fp[s]) / len(imgs):.1f}%)")
    (BENCH / "stress_false_positive_images.json").write_text(json.dumps(fp, indent=1), encoding="utf-8")
    # trivial inputs
    for nm, im in {"black": np.zeros((480, 640, 3), np.uint8),
                   "noise": np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)}.items():
        print(f"  {nm} frame: " + ", ".join(f"{s}={len(d.detect(im))} faces" for s, d in dets.items()))


# ------------------------------------------------------------------ audio
def cmd_audio(a) -> None:
    load = _load_avg()
    print(f"system CPU load before run: {load}%")
    rng = random.Random(0)
    asv = sorted(Path("data/asvspoof2019_la/LA").glob("ASVspoof2019_LA_*/flac/*.flac"))
    if asv:
        sel = rng.sample(asv, min(a.n_files, len(asv)))
        t_total = dur_total = 0.0
        import soundfile as sf
        for p in sel:
            t0 = time.perf_counter()
            x = A.load_audio(p)
            t_total += time.perf_counter() - t0
            dur_total += len(x) / A.DEFAULT_SR
        print(f"ASVspoof2019 LA flac: {len(sel)} random files of {len(asv)}; load_audio {1000 * t_total / len(sel):.2f} ms/file, "
              f"{dur_total / len(sel):.2f} s audio/file, {dur_total / t_total:.0f}x real time; sample rates in files: "
              f"{sorted({sf.info(str(p)).samplerate for p in sel})}")
    else:
        print("no ASVspoof2019 LA flac files found")
    # videos with audio: Celeb-DF v2 if present and complete
    cands = [p for p in sorted(Path("data/celebdf_v2").rglob("*.mp4")) if V.is_stable(p, 20)]
    if cands:
        sel = rng.sample(cands, min(a.n_files // 4, len(cands)))
        t_total = dur_total = 0.0
        n_audio = 0
        for p in sel:
            t0 = time.perf_counter()
            x = A.extract_audio(p)
            t_total += time.perf_counter() - t0
            dur_total += len(x) / A.DEFAULT_SR
            n_audio += len(x) > 0
        print(f"Celeb-DF v2 mp4: {len(sel)} random complete files; extract_audio {1000 * t_total / len(sel):.1f} ms/file, "
              f"{dur_total / len(sel):.2f} s audio/file, {n_audio}/{len(sel)} had audio, {dur_total / t_total:.0f}x real time")
    else:
        print("no complete Celeb-DF v2 mp4 files found")
    # FF++ (no audio stream expected)
    ff = sorted(p for p in (FFPP / "original_sequences/youtube/c23/videos").glob("*.mp4") if V.is_stable(p, 20))[:20]
    t0 = time.perf_counter()
    lens = [len(A.extract_audio(p)) for p in ff]
    print(f"FF++ c23 originals: {len(ff)} files, extract_audio {1000 * (time.perf_counter() - t0) / len(ff):.1f} ms/file, "
          f"files with non-empty audio: {sum(l > 0 for l in lens)}")
    # synthetic: 10 s test tone mux
    BENCH.mkdir(parents=True, exist_ok=True)
    tone = BENCH / "tone10s.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=10",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=10", "-c:v", "libx264",
                    "-c:a", "aac", "-shortest", str(tone)], check=True)
    t0 = time.perf_counter()
    for _ in range(20):
        x = A.extract_audio(tone)
    print(f"SYNTHETIC 10 s tone (44.1 kHz AAC in mp4, made with ffmpeg): extract_audio "
          f"{1000 * (time.perf_counter() - t0) / 20:.1f} ms/file, {len(x)} samples ({len(x) / A.DEFAULT_SR:.2f} s)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--out", default=str(BENCH / "sample_c23.csv"))
    s.add_argument("--per-method", type=int, default=25)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_sample)
    d = sub.add_parser("decode")
    d.add_argument("--manifest", default=str(BENCH / "sample_c23.csv"))
    d.add_argument("--every", type=int, default=3)
    d.add_argument("--n", type=int, default=V.DEFAULT_N_FRAMES)
    d.add_argument("--threads", type=int, default=1)
    d.add_argument("--tag", default="", help="suffix of the raw output file")
    d.add_argument("--backends", nargs="+", default=list(V.BACKENDS))
    d.set_defaults(fn=cmd_decode)
    t = sub.add_parser("detect")
    t.add_argument("--manifest", default=str(BENCH / "sample_c23.csv"))
    t.add_argument("--n", type=int, default=V.DEFAULT_N_FRAMES)
    t.add_argument("--every", type=int, default=1, help="use every K-th manifest row")
    t.add_argument("--tag", default="", help="suffix of the raw output file")
    t.add_argument("--model-dir", default=str(F.DEFAULT_MODEL_DIR))
    t.add_argument("--detectors", nargs="+", default=list(F.DETECTORS) + ["yunet_full"])
    t.set_defaults(fn=cmd_detect)
    q = sub.add_parser("decodepool")
    q.add_argument("--manifest", default=str(BENCH / "sample_c23.csv"))
    q.add_argument("--workers", type=int, default=16)
    q.add_argument("--repeat", type=int, default=1)
    q.add_argument("--n", type=int, default=V.DEFAULT_N_FRAMES)
    q.add_argument("--backend", default="cv_seq", choices=V.BACKENDS)
    q.set_defaults(fn=cmd_decodepool)
    j = sub.add_parser("jpeg")
    j.add_argument("--manifest", default=str(BENCH / "sample_c23.csv"))
    j.add_argument("--every", type=int, default=10)
    j.add_argument("--n", type=int, default=V.DEFAULT_N_FRAMES)
    j.add_argument("--size", type=int, default=F.DEFAULT_CROP_SIZE)
    j.add_argument("--model-dir", default=str(F.DEFAULT_MODEL_DIR))
    j.add_argument("--qualities", type=int, nargs="+", default=[100, 95, 90, 85, 75])
    j.set_defaults(fn=cmd_jpeg)
    x = sub.add_parser("stress")
    x.add_argument("--manifest", default=str(BENCH / "sample_c23.csv"))
    x.add_argument("--model-dir", default=str(F.DEFAULT_MODEL_DIR))
    x.add_argument("--detectors", nargs="+", default=list(F.DETECTORS))
    x.add_argument("--n-noface", type=int, default=500)
    x.set_defaults(fn=cmd_stress)
    u = sub.add_parser("audio")
    u.add_argument("--n-files", type=int, default=200)
    u.set_defaults(fn=cmd_audio)
    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
