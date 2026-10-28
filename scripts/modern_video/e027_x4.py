"""E-027 X4: crop and frame-count ablation on the open subset (secondary; never changes the closed-look candidate).

    PYTHONPATH="src;scripts/modern_video" .venv/Scripts/python.exe scripts/modern_video/e027_x4.py encode     # ffmpeg views of the development clips
    PYTHONPATH="src;scripts/modern_video" .venv/Scripts/python.exe scripts/modern_video/e027_x4.py features   # chosen encoder, 16 frames per clip
    PYTHONPATH="src;scripts/modern_video" .venv/Scripts/python.exe scripts/modern_video/e027_x4.py analyse    # LOFO, paired against the chosen candidate

Views (each re-encoded with the canonical-v3 codec settings; only the filter graph and the window differ): C16 canonical geometry,
LB16 letterboxed whole frame, NC16 native-resolution 224 x 224 centre crop; 5 fps over the whole clip, then 16 frames picked uniformly.
Clips are DF26-derived and stay in data/_results/modern/e027/x4_clips/ (git-ignored).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import e027_lib as L

X4 = L.OUT / "x4_clips"
SET = "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv"
VF = {"C16": "fps=5,scale=224:224:force_original_aspect_ratio=increase:flags=area,crop=224:224," + SET,
      "LB16": "fps=5,scale=224:224:force_original_aspect_ratio=decrease:flags=area,pad=224:224:(ow-iw)/2:(oh-ih)/2:black," + SET,
      "NC16": "fps=5,crop=224:224," + SET}


def args(src, dst, vf):
    return ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-map_metadata", "-1", "-map_chapters", "-1", "-an", "-sn",
            "-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-g", "14",
            "-bf", "2", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
            "-x264-params", "no-info=1:scenecut=0:b-adapt=0", "-movflags", "+faststart", str(dst)]


def read16(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(out) < 16:
        raise ValueError(f"{len(out)} frames, 16 needed")
    idx = np.linspace(0, len(out) - 1, 16).round().astype(int)
    return np.stack([out[i] for i in idx])


def encode() -> int:
    rows = L.dev_rows()
    jobs = []
    for v, vf in VF.items():
        for r in rows:
            dst = X4 / v / r["video_path"]
            if not dst.exists():
                jobs.append((L.DF26 / r["video_path"], dst, vf))
    for _, d, _ in jobs:
        d.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    bad = []

    def run(j):
        r = subprocess.run(args(*j), capture_output=True, text=True, timeout=300)
        return j[1], r.returncode, r.stderr[-200:]
    with ThreadPoolExecutor(8) as ex:
        for k, (d, rc, err) in enumerate(ex.map(run, jobs), 1):
            if rc != 0:
                bad.append({"clip": str(d), "error": err})
            if k % 500 == 0:
                print(f"{k}/{len(jobs)} {time.time() - t0:.0f} s", flush=True)
    L.jdump({"n_jobs": len(jobs), "failures": bad, "seconds": round(time.time() - t0, 1)}, L.OUT / "x4_encode.json")
    print(f"encoded {len(jobs)}, failures {len(bad)}, {time.time() - t0:.0f} s")
    return 0


def features() -> int:
    from e027_features import extract, save
    dev = json.loads((L.OUT / "dev_results.json").read_text(encoding="utf-8"))
    cid = dev["selection"]["chosen"].split("-", 1)[1]
    rows = L.dev_rows()
    for v in VF:
        out = L.OUT / f"features_x4_{v}_{cid}_dev.npz"
        if out.exists():
            continue
        embs, keep, bad, info = extract(rows, cid, root=X4 / v, reader=read16)
        save(out, embs, keep, bad, info, cid)
        print(v, len(keep), len(bad), info, flush=True)
    return 0


def analyse() -> int:
    if (L.OUT / "x4_results.json").exists():
        print("x4_results.json exists: refusing to overwrite")
        return 2
    from e027_dev import load_dev, summarise, diff
    dev = json.loads((L.OUT / "dev_results.json").read_text(encoding="utf-8"))
    ch = dev["selection"]["chosen"]
    view_c, cid = ch.split("-", 1)
    rows, ids, y, gen, grp, stored, emb = load_dev()
    Xc = stored["cls_mean"] if (cid == "IMG-07" and view_c == "P8") else L.view(emb[cid], view_c)
    mb = L.MacroBoot(y, gen, grp)
    M, out = {}, {"experiment": "E-027", "stage": "X4", "chosen": ch, "encoder": cid, "views": {}}
    M["chosen"], out["chosen_LOFO"] = summarise(mb, [L.oof_scores(Xc, y, gen, grp, "LOFO", s) for s in L.SEEDS], "LOFO")
    for nm in ("P14", "PT14"):
        M[nm], _ = summarise(mb, [L.oof_scores(L.view(emb[cid], nm), y, gen, grp, "LOFO", s) for s in L.SEEDS], "LOFO")
    for v in VF:
        f = np.load(L.OUT / f"features_x4_{v}_{cid}_dev.npz", allow_pickle=False)
        p2 = {k: i for i, k in enumerate(f["id"])}
        miss = [i for i in ids if i not in p2]
        if miss:
            out["views"][v] = {"failed": f"{len(miss)} clips missing"}
            continue
        E = f["emb"][[p2[i] for i in ids]].astype(np.float64)
        out["views"][v] = {}
        for nm in ("P16", "PT16"):
            m, d = summarise(mb, [L.oof_scores(L.view(E, nm), y, gen, grp, "LOFO", s) for s in L.SEEDS], "LOFO")
            d["minus_chosen"] = diff(m, M["chosen"])
            d["minus_canonical_same_pooling"] = diff(m, M["P14" if nm == "P16" else "PT14"])
            out["views"][v][nm] = d
    L.jdump(out, L.OUT / "x4_results.json")
    for v, d in out["views"].items():
        for nm, r in d.items():
            print(v, nm, r if isinstance(r, str) else (r["macro_auc"], r["ci95"], r["minus_chosen"], r["minus_canonical_same_pooling"]))
    print("chosen", out["chosen_LOFO"]["macro_auc"], out["chosen_LOFO"]["ci95"])
    return 0


if __name__ == "__main__":
    raise SystemExit({"encode": encode, "features": features, "analyse": analyse}[sys.argv[1]]())
