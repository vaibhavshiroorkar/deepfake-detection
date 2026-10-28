"""E-026: source-confound test of the installed experimental generated-video head on real videos from other sources (roadmap X2),
gated by the X1 reproduction of E-020's C2 out-of-fold result. Protocol: docs/EXPERIMENTS.md E-026 (preregistered before this ran).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/genvideo/e026.py

Development data only: FF++ val originals through the split guard, LAV-DF official-train real clips, DeepAction stored features. DF26 and
every sealed split are never opened. Nothing is fitted on the new clips, and no head, threshold, checkpoint or ledger entry is written.
Outputs: data/_results/genvideo/e026/ only (refuses to overwrite an existing result).
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = DATA / "_results" / "genvideo" / "e026"
HEAD = ROOT / "checkpoints" / "genvideo-E020" / "head.npz"
HEAD_SHA = "b67d01ac0001ec69f53ce659c88ab8b4415bc3fb810ce30041651de3bfa7fafb"
EXPERIMENTS = ROOT / "docs" / "EXPERIMENTS.md"
SEED = 20261027
N_LAV = 300
N_BOOT = 2000


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 4), round(float(c + h), 4)]


def load_e020():
    spec = importlib.util.spec_from_file_location("e020", ROOT / "scripts" / "genvideo" / "e020.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def x1_reproduction(E) -> tuple[dict, np.ndarray]:
    """C2 out-of-fold LOGO scores recomputed with E-020's own code (cmd_dev's loop, C2 only, seed 0). Returns (record, oof score per real)."""
    z = np.load(E.FEAT / "features_deepaction.npz", allow_pickle=False)
    y, gen, grp = z["label"].astype(int), z["generator"], z["group"]
    gens = sorted(set(gen[y == 1]))
    fold = E.folds_of(grp)
    X = z["cls_mean"].astype(np.float64)
    S = np.full((len(gens), len(y)), np.nan)
    for gi, g in enumerate(gens):
        for f in range(E.FOLDS):
            tr = (gen != g) & (fold != f)
            te = ((gen == g) | (y == 0)) & (fold == f)
            if te.any():
                S[gi, te] = E.predict(E.fit("C2", X[tr], y[tr], 0), X[te])
    macro = float(np.mean([E.roc_auc(y[(gen == g) | (y == 0)], S[gi][(gen == g) | (y == 0)]) for gi, g in enumerate(gens)]))
    recorded = json.loads((E.FEAT / "e020_dev_results.json").read_text(encoding="utf-8"))["candidates"]["C2"]["macro_logo_auc"]
    real_oof = np.nanmean(S[:, y == 0], axis=0)
    rec = {"recomputed_macro_logo_auc": round(macro, 4), "recorded_macro_logo_auc": recorded, "pass": round(macro, 4) == recorded,
           "n_real": int((y == 0).sum()), "real_ids": [str(i) for i in z["id"][y == 0]], "real_oof_scores": [round(float(v), 6) for v in real_oof],
           "real_in_sample_dims": int(X.shape[1])}
    return rec, X[y == 0]


def clips_ffpp() -> list[tuple[str, Path]]:
    from dfd.data import loaders as L
    recs = L.select_records(DATA / "_manifests" / "ffpp.manifest.csv", DATA / "_splits" / "ffpp-037e2976fade.csv", "val")   # refuses test splits
    return [(r.sample_id, DATA / r.path) for r in recs if r.label == 0]


def clips_lavdf() -> tuple[list[tuple[str, Path]], int]:
    meta = json.loads((DATA / "lavdf" / "LAV-DF" / "metadata.min.json").read_text(encoding="utf-8"))
    ok = sorted(m["file"] for m in meta if m["split"] == "train" and m["n_fakes"] == 0 and not m["modify_video"] and not m["modify_audio"])
    assert all(f.startswith("train/") for f in ok)
    pick = np.random.default_rng(SEED).choice(len(ok), size=N_LAV, replace=False)
    files = [ok[i] for i in sorted(pick)]
    return [("lavdf:" + f, DATA / "lavdf" / "LAV-DF" / f) for f in files], len(ok)


def auc_vs_ref(other: np.ndarray, ref: np.ndarray, rng) -> dict:
    from dfd.eval.metrics import roc_auc
    def a(o, r):
        return float(roc_auc(np.r_[np.ones(len(o)), np.zeros(len(r))], np.r_[o, r]))
    reps = [a(rng.choice(other, len(other)), rng.choice(ref, len(ref))) for _ in range(N_BOOT)]
    return {"auc": round(a(other, ref), 4), "ci95": [round(float(np.percentile(reps, 2.5)), 4), round(float(np.percentile(reps, 97.5)), 4)]}


def describe(scores: np.ndarray, thr: float) -> dict:
    k, n = int((scores >= thr).sum()), int(len(scores))
    return {"n": n, "flagged": k, "fpr": round(k / n, 4) if n else None, "wilson95": wilson(k, n),
            "quantiles": {q: round(float(np.percentile(scores, q)), 4) for q in (5, 25, 50, 75, 95)} if n else None}


def main() -> int:
    if (OUT / "e026_results.json").exists():
        print("E-026 result already exists; refusing to overwrite (a rerun needs a new id or a recorded reason)")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    text = EXPERIMENTS.read_text(encoding="utf-8")
    entry = text[text.index("## E-026:"):]
    assert "- Result: (to be appended below after the run)" in entry and "E-026 RESULT" not in entry, "preregistration missing or already has a result"
    (OUT / "prereg.json").write_text(json.dumps({"written_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "experiments_md_sha256": sha256(EXPERIMENTS),
                                                 "e026_entry_sha256": hashlib.sha256(entry.encode("utf-8")).hexdigest(), "committed": False}, indent=1),
                                     encoding="utf-8")
    assert sha256(HEAD) == HEAD_SHA, "installed head differs from the preregistered hash"
    E = load_e020()
    x1, _ = x1_reproduction(E)
    (OUT / "x1_reproduction.json").write_text(json.dumps(x1, indent=1), encoding="utf-8")
    print("X1:", x1["recomputed_macro_logo_auc"], "recorded", x1["recorded_macro_logo_auc"], "PASS" if x1["pass"] else "FAIL")
    if not x1["pass"]:
        print("X1 failed: stopping before any clip is scored (preregistered stop rule)")
        return 1

    from dfd.platform.genvideo import canonical
    from dfd.platform.genvideo.features import shared_runner
    from dfd.platform.genvideo.model import load_head
    head = load_head(HEAD)
    dev = json.loads((E.FEAT / "e020_dev_results.json").read_text(encoding="utf-8"))
    thr = float(head.threshold)
    assert abs(thr - dev["selection"]["threshold"]) < 1e-9, "head threshold differs from E-020's recorded threshold"
    assert head.meta.get("feature") == "cls_mean" and head.meta.get("protocol") == canonical.PROTOCOL_ID
    runner = shared_runner()

    ff = clips_ffpp()
    lav, n_lav_eligible = clips_lavdf()
    rows = []
    t0 = time.perf_counter()
    for source, clips in (("R-FF", ff), ("R-LAV", lav)):
        for i, (sid, path) in enumerate(clips):
            row = {"source": source, "sample_id": sid, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "score": "", "error": ""}
            try:
                with tempfile.TemporaryDirectory() as td:
                    dst = Path(td) / "c.mp4"
                    canonical.canonicalise(path, dst)
                    frames = canonical.read_frames(dst)
                feats = runner.features(frames)
                row["score"] = round(float(head.score(feats["cls_mean"])[0]), 6)
            except Exception as e:  # noqa: BLE001 - listed and excluded (preregistered)
                row["error"] = f"{type(e).__name__}: {str(e)[:160]}"
            rows.append(row)
            if (i + 1) % 50 == 0:
                print(f"{source} {i + 1}/{len(clips)} ({time.perf_counter() - t0:.0f} s)", flush=True)
    with open(OUT / "scores.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    rng = np.random.default_rng(SEED)
    ref = np.array(x1["real_oof_scores"])
    # in-sample: the installed full-data head on DeepAction reals (it saw them in training); reported separately, never as the reference
    z = np.load(E.FEAT / "features_deepaction.npz", allow_pickle=False)
    in_sample = np.array([float(head.score(v)[0]) for v in z["cls_mean"][z["label"].astype(int) == 0]])
    res = {"experiment": "E-026", "date": time.strftime("%Y-%m-%d"), "head": {"path": "checkpoints/genvideo-E020/head.npz", "sha256": HEAD_SHA,
           "head_id": head.meta.get("head_id"), "threshold": thr}, "x1": {k: x1[k] for k in ("recomputed_macro_logo_auc", "recorded_macro_logo_auc", "pass")},
           "sources": {}, "failures": {}, "n_lavdf_eligible": n_lav_eligible,
           "environment": {"python": platform.python_version(), "platform": platform.platform(), "dino_revision": runner.revision,
                           "device": getattr(runner, "_dev", None), "seconds_scoring": round(time.perf_counter() - t0, 1)}}
    res["sources"]["R-DA (DeepAction real, out of fold)"] = describe(ref, thr)
    res["sources"]["R-DA in-sample (installed head; saw these clips in training)"] = describe(in_sample, thr)
    pooled = []
    for src in ("R-FF", "R-LAV"):
        ok = np.array([r["score"] for r in rows if r["source"] == src and r["error"] == ""], dtype=float)
        bad = [r for r in rows if r["source"] == src and r["error"]]
        res["failures"][src] = {"n": len(bad), "share": round(len(bad) / max(1, len(bad) + len(ok)), 4), "examples": [r["error"] for r in bad[:5]]}
        d = describe(ok, thr)
        d["failed_over_10_percent"] = res["failures"][src]["share"] > 0.10
        d["auc_vs_RDA_oof"] = auc_vs_ref(ok, ref, rng) if len(ok) else None
        res["sources"][src] = d
        if not d["failed_over_10_percent"]:
            pooled.append(ok)
    P = np.concatenate(pooled) if pooled else np.array([])
    pd = describe(P, thr)
    pd["auc_vs_RDA_oof"] = auc_vs_ref(P, ref, rng) if len(P) else None
    res["sources"]["pooled R-FF + R-LAV"] = pd
    ref_fpr = res["sources"]["R-DA (DeepAction real, out of fold)"]["fpr"]
    lo, hi = pd["wilson95"]
    verdict = "SUPPORTED" if (lo is not None and lo > ref_fpr and lo > 0.5) else ("NOT SUPPORTED" if (hi is not None and hi <= ref_fpr) else "INCONCLUSIVE")
    res["h_x2"] = {"verdict": verdict, "rule": "SUPPORTED if pooled Wilson lower bound > R-DA OOF FPR and > 0.5; NOT SUPPORTED if pooled upper bound <= R-DA OOF FPR; "
                   "otherwise INCONCLUSIVE (preregistered)", "reference_fpr": ref_fpr, "pooled_wilson95": [lo, hi]}
    (OUT / "e026_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("x1", "h_x2")}, indent=1))
    for k, v in res["sources"].items():
        print(f"{k}: n={v['n']} flagged={v['flagged']} fpr={v['fpr']} {v['wilson95']} median={v['quantiles'][50] if v['quantiles'] else None}"
              + (f" auc_vs_ref={v['auc_vs_RDA_oof']}" if v.get("auc_vs_RDA_oof") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
