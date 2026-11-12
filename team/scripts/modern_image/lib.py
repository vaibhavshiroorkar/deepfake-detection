"""Shared code of the W1-I scripts (E-028, E-029): shard reading, the product's decode and preprocessing, metrics with bootstrap intervals.

The scoring path is the product's (src/dfd/infer/real_adapter.py, read only): bytes -> cv2.imdecode(IMREAD_COLOR) -> RGB -> PIL -> the encoder's
model-card processor (dfd.sweep.encoders.HFImageProcessor) -> Encoder.encode_pixels -> Head.prob_rows -> temperature (real_adapter._calibrate).
Workers run the CPU part (decode, processor, face detection, face crop); the main process runs the GPU part with the adapter's own objects.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import _env  # noqa: E402  (adds the pyarrow folder)

CKPT = ROOT / "checkpoints" / "I-5-freeze-candidate"
SEED = 20261027
N_BOOT = 2000
IMG13_MEAN, IMG13_STD = np.array([0.48145466, 0.4578275, 0.40821073], np.float32), np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
IMG15_MEAN, IMG15_STD = np.array([0.5, 0.5, 0.5], np.float32), np.array([0.5, 0.5, 0.5], np.float32)

# ------------------------------------------------------------------ index and shards


def subset_rows(which: str) -> list[dict]:
    base = _env.CF_EVAL if which == "eval" else _env.CF_SMALL
    sub = json.load(open(base / "_meta" / "subset.json"))
    files = {s["file"] for s in sub["shards"]}
    return [r for r in csv.DictReader(open(base / "_meta" / "index.csv", encoding="utf-8")) if r["shard"] in files]


def verified_shards(which: str) -> dict[str, Path]:
    base = _env.CF_EVAL if which == "eval" else _env.CF_SMALL
    logf = base / "_meta" / "download_log.jsonl"
    ok = {}
    if logf.exists():
        for line in open(logf):
            if line.strip():
                r = json.loads(line)
                if r.get("match"):
                    ok[r["file"]] = base / "shards" / r["file"]
    return {k: v for k, v in ok.items() if v.exists()}


# ------------------------------------------------------------------ worker (CPU part)
_W = {}


def _init_worker(proc_dir: str, want_faces: bool):
    import cv2
    cv2.setNumThreads(1)
    from dfd.sweep.encoders import HFImageProcessor
    _W["proc"] = HFImageProcessor(proc_dir)
    _W["faces"] = want_faces
    if want_faces:
        from dfd.prep import face as F
        meta = json.load(open(CKPT / "metadata.json"))
        cfg = meta["tasks"]["T-V"]["preprocess_config"]
        _W["det"] = F.make_detector(cfg["detector"], F.DEFAULT_MODEL_DIR)
        _W["cfg"] = cfg


def _pixels(bgr):
    from PIL import Image
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    return _W["proc"](Image.fromarray(rgb)).numpy().astype(np.float32)


def work(item):
    """item = (key, image bytes, reencode flag). Returns dict with pixels [3,224,224] float32 (and of the JPEG-90 re-encode if flagged),
    image size, number of faces, the T-V face crop (RGB uint8, as the adapter makes it) or None, and an error string."""
    import cv2
    key, data, reenc = item
    out = {"key": key, "ok": False, "err": "", "h": 0, "w": 0, "n_faces": -1, "crop": None, "px": None, "px_re": None}
    try:
        bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            out["err"] = "cv2.imdecode returned None"
            return out
        out["h"], out["w"] = bgr.shape[:2]
        out["px"] = _pixels(bgr)
        if reenc:
            okj, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            out["px_re"] = _pixels(cv2.imdecode(buf, cv2.IMREAD_COLOR))
        if _W["faces"]:
            from dfd.prep import face as F
            from dfd.infer.real_adapter import RealAdapter
            faces = _W["det"].detect(bgr)
            out["n_faces"] = len(faces)
            if faces:
                cfg = _W["cfg"]
                crop, _, _ = F.crop_largest_face(bgr, faces, cfg["enlarge"], cfg["size"])
                if crop is not None:
                    out["crop"] = np.ascontiguousarray(RealAdapter._jpeg(crop, cfg["quality"])[:, :, ::-1])
        out["ok"] = True
    except Exception as e:  # noqa: BLE001
        out["err"] = repr(e)[:300]
    return out


def iter_shard(path: Path, batch=32, columns=("image_data", "image_name", "label", "model_name", "nsfw_flag")):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(path))
    row = 0
    for b in pf.iter_batches(batch_size=batch, columns=list(columns)):
        d = b.to_pydict()
        for k in range(b.num_rows):
            yield row, {c: d[c][k] for c in columns}
            row += 1


# ------------------------------------------------------------------ metrics


def auc_fast(y, s):
    """Mann-Whitney AUC with average ranks for ties (equals dfd.eval.metrics.roc_auc; checked in the scripts)."""
    from scipy.stats import rankdata
    y = np.asarray(y)
    s = np.asarray(s, float)
    n1 = int(y.sum())
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ap_score(y, s):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s)) if 0 < np.sum(y) < len(y) else float("nan")


def at_threshold(y, s, t):
    y = np.asarray(y).astype(bool)
    p = np.asarray(s) >= t
    tp, fp = int((p & y).sum()), int((p & ~y).sum())
    fn, tn = int((~p & y).sum()), int((~p & ~y).sum())
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    fnr = fn / (fn + tp) if fn + tp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (tp + fp) and (tp + fn) and (prec + rec) > 0 else float("nan")
    ba = 1 - (fpr + fnr) / 2 if fp + tn and fn + tp else float("nan")
    return {"fpr": fpr, "fnr": fnr, "precision": prec, "recall": rec, "f1": f1, "balanced_accuracy": ba, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def metric_set(y, s, t):
    d = {"auc": auc_fast(y, s), "pr_auc": ap_score(y, s)}
    d.update(at_threshold(y, s, t))
    return d


KEYS = ["auc", "pr_auc", "fpr", "fnr", "precision", "recall", "f1", "balanced_accuracy"]


class Boot:
    """Stratified bootstrap over images: reals and fakes resampled separately within each cell, the same draws for every scorer (paired).
    cells: dict name -> (idx_real, idx_fake) into the score arrays. draws[name] = list of (r_idx, f_idx) per resample."""

    def __init__(self, cells: dict, n_boot=N_BOOT, seed=SEED):
        rng = np.random.default_rng(seed)
        self.cells = cells
        self.n_boot = n_boot
        self.draws = {}
        for name in sorted(cells):
            r, f = cells[name]
            self.draws[name] = (rng.integers(0, len(r), (n_boot, len(r))) if len(r) else None,
                                rng.integers(0, len(f), (n_boot, len(f))) if len(f) else None)

    def idx(self, name, b):
        r, f = self.cells[name]
        dr, df = self.draws[name]
        return r[dr[b]], f[df[b]]


def pct(a, level=0.95):
    a = np.asarray([x for x in a if x == x])
    if a.size == 0:
        return [float("nan"), float("nan")]
    lo, hi = np.percentile(a, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return [float(lo), float(hi)]


def r4(x):
    return None if x is None or x != x else round(float(x), 4)
