"""E-015: the frozen system on the sealed identity half of FakeAVCeleb (D-070). Protocol: docs/EXPERIMENTS.md E-015.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/e015/e015.py score --final      # writes predictions, reads no label
    PYTHONPATH=src .venv/Scripts/python.exe scripts/e015/e015.py evaluate --final   # reads labels, computes metrics, appends the ledger entry once

``score`` does not use any label: it iterates over the sealed sample ids of the split and writes the model outputs. ``evaluate`` refuses a second run
(ledger) and refuses to run unless the predictions file covers every sealed clip. Nothing here changes a checkpoint, a threshold or a temperature.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = DATA / "_results" / "e015"
PRED = OUT / "predictions_test_x.csv"
SPLIT = DATA / "_splits" / "fakeavceleb-79d3f7051972.csv"
MANIFEST = DATA / "_manifests" / "fakeavceleb.manifest.csv"
CKPT = ROOT / "checkpoints" / "I-5-freeze-candidate"
LEDGER = ROOT / "ledger" / "final_ledger.jsonl"
COLS = ["sample_id", "visual_available", "visual_score", "visual_n_frames", "audio_available", "audio_score", "audio_n_units", "warnings", "error", "ms"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sealed_ids(side: str = "test_x") -> list[str]:
    from dfd.data.splits import read_split
    asg = read_split(SPLIT)
    return sorted(s for s, v in asg.items() if v == side)


def cmd_score(args) -> int:
    from dfd.data.manifest import read_manifest
    from dfd.infer.library import inspect
    from dfd.infer.real_adapter import RealAdapter
    smoke = args.smoke_dev > 0
    ids = sealed_ids("dev_x")[: args.smoke_dev] if smoke else sealed_ids()
    pred_path = OUT / "smoke_dev.csv" if smoke else PRED     # the smoke run scores a few DEVELOPMENT clips to test the code; it is not part of E-015
    paths = {r.sample_id: r.path for r in read_manifest(MANIFEST)}       # path only; labels are not used here
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if pred_path.exists():
        with open(pred_path, encoding="utf-8", newline="") as f:
            done = {r["sample_id"] for r in csv.DictReader(f)}
    todo = [s for s in ids if s not in done]
    print(f"sealed clips {len(ids)}, already scored {len(done)}, to score {len(todo)}", flush=True)
    if not todo:
        return 0
    ad = RealAdapter(CKPT)
    new = not pred_path.exists()
    t_all = time.time()
    with open(pred_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        for k, sid in enumerate(todo, 1):
            p = DATA / paths[sid]
            row = dict.fromkeys(COLS, "")
            row["sample_id"] = sid
            t = time.time()
            try:
                res = ad.analyse(inspect(p, p.name))
                v, a = res.per_modality.get("visual"), res.per_modality.get("audio")
                if v is not None:
                    row["visual_available"], row["visual_score"], row["visual_n_frames"] = int(v.available), "" if v.score is None else v.score, v.n_units or 0
                if a is not None:
                    row["audio_available"], row["audio_score"], row["audio_n_units"] = int(a.available), "" if a.score is None else a.score, a.n_units or 0
                row["warnings"] = "|".join(x["code"] for x in res.warnings)
            except Exception as e:  # noqa: BLE001 - recorded, never retried silently
                row["error"] = f"{type(e).__name__}: {str(e)[:150]}"
            row["ms"] = round((time.time() - t) * 1000)
            w.writerow(row)
            if k % 50 == 0:
                f.flush()
                print(f"{k}/{len(todo)} {round(time.time() - t_all)} s", flush=True)
    return 0


def _auc_ci(y, s, groups, n_boot):
    from dfd.eval.bootstrap import cluster_bootstrap
    r = cluster_bootstrap(np.asarray(y), np.asarray(s), np.asarray(groups), metric="auc", n_boot=n_boot)
    return {"auc": round(r.estimate, 4), "low": round(r.low, 4), "high": round(r.high, 4), "n": int(len(y)), "n_fake": int(np.sum(y)),
            "n_real": int(len(y) - np.sum(y)), "n_groups": int(len(set(groups)))}


def verdict_transfer(ci):
    return "supported" if ci["low"] > 0.70 else ("not supported" if ci["high"] < 0.60 else "inconclusive")


def verdict_independent(ci):
    return "supported" if ci["low"] >= 0.40 and ci["high"] <= 0.60 else ("violated" if ci["low"] > 0.60 or ci["high"] < 0.40 else "inconclusive")


def cmd_evaluate(args) -> int:
    from dfd.data.manifest import read_manifest
    from dfd.eval.ledger import append_entries, read_ledger
    if any(e.get("iteration") == "E-015" and e.get("dataset") == "fakeavceleb" for e in read_ledger(LEDGER)):
        print("E-015 is already in the ledger: refused (a second look needs a documented repeat reason)", file=sys.stderr)
        return 2
    ids = sealed_ids()
    with open(PRED, encoding="utf-8", newline="") as f:
        pred = {r["sample_id"]: r for r in csv.DictReader(f)}
    if set(ids) - set(pred):
        print(f"predictions cover {len(pred)} of {len(ids)} sealed clips: score first", file=sys.stderr)
        return 1
    pred_sha = sha256(PRED)
    meta = json.loads((CKPT / "metadata.json").read_text(encoding="utf-8"))
    thr = {"visual": float(meta["tasks"]["T-V"]["threshold"]["value"]), "audio": float(meta["tasks"]["T-A"]["threshold"]["value"])}
    recs = {r.sample_id: r for r in read_manifest(MANIFEST)}
    from dfd.data.fakeavceleb_split import note_field
    rows = []
    for sid in ids:
        r, p = recs[sid], pred[sid]
        rows.append({"sid": sid, "cat": note_field(r.notes, "category"), "method": r.method, "group": r.group_id, "cell": note_field(r.notes, "cell"),
                     "v": float(p["visual_score"]) if p["visual_available"] == "1" and p["visual_score"] != "" else None,
                     "a": float(p["audio_score"]) if p["audio_available"] == "1" and p["audio_score"] != "" else None,
                     "error": p["error"], "warn": p["warnings"]})
    nb = args.n_boot
    out: dict = {"experiment": "E-015", "date": time.strftime("%Y-%m-%d"), "predictions_sha256": pred_sha, "split": SPLIT.name, "thresholds": thr,
                 "checkpoint_id": meta["checkpoint_id"], "n_sealed": len(ids)}
    out["coverage"] = {"errors": sum(1 for x in rows if x["error"]),
                       "visual_scored": sum(1 for x in rows if x["v"] is not None), "audio_scored": sum(1 for x in rows if x["a"] is not None),
                       "no_face_flag": sum(1 for x in rows if "NO_FACE_DETECTED" in x["warn"]),
                       "by_category": {c: sum(1 for x in rows if x["cat"] == c) for c in ("RVRA", "RVFA", "FVRA", "FVFA")}}

    def sub(key, cats_real, cats_fake, methods=None):
        xs = [x for x in rows if x[key] is not None and (x["cat"] in cats_real or (x["cat"] in cats_fake and (methods is None or x["method"] in methods)))]
        y = [1 if x["cat"] in cats_fake else 0 for x in xs]
        if len(set(y)) < 2:
            return None
        ci = _auc_ci(y, [x[key] for x in xs], [x["group"] for x in xs], nb)
        s = np.array([x[key] for x in xs]); yy = np.array(y)
        t = thr["visual" if key == "v" else "audio"]
        ci["sensitivity_at_frozen_threshold"] = round(float(np.mean(s[yy == 1] >= t)), 4)
        ci["specificity_at_frozen_threshold"] = round(float(np.mean(s[yy == 0] < t)), 4)
        return ci

    res = {}
    res["H1_visual_pooled"] = sub("v", {"RVRA", "RVFA"}, {"FVRA", "FVFA"})
    res["H2_audio_pooled"] = sub("a", {"RVRA", "FVRA"}, {"RVFA", "FVFA"})
    res["H3_visual_RVRA_vs_RVFA"] = sub("v", {"RVRA"}, {"RVFA"})
    res["H3_audio_RVRA_vs_FVRA"] = sub("a", {"RVRA"}, {"FVRA"})
    res["verdicts"] = {"H1": verdict_transfer(res["H1_visual_pooled"]), "H2": verdict_transfer(res["H2_audio_pooled"]),
                       "H3_visual": verdict_independent(res["H3_visual_RVRA_vs_RVFA"]), "H3_audio": verdict_independent(res["H3_audio_RVRA_vs_FVRA"])}
    res["secondary_visual_by_method"] = {m: sub("v", {"RVRA", "RVFA"}, {"FVRA", "FVFA"}, {m}) for m in ("faceswap", "fsgan", "wav2lip", "faceswap-wav2lip", "fsgan-wav2lip")}
    res["secondary_audio_by_category"] = {c: sub("a", {"RVRA", "FVRA"}, {c}) for c in ("RVFA", "FVFA")}
    res["secondary_visual_by_category"] = {c: sub("v", {"RVRA", "RVFA"}, {c}) for c in ("FVRA", "FVFA")}
    out["results"] = res
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e015_results.json").write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    append_entries([{"date": time.strftime("%Y-%m-%d"), "iteration": "E-015", "dataset": "fakeavceleb", "split": "test_x", "run_ids": [meta["checkpoint_id"]],
                     "modalities": ["visual", "audio"], "score_file_sha256": pred_sha}], LEDGER)
    print(json.dumps({"verdicts": res["verdicts"], "H1": res["H1_visual_pooled"], "H2": res["H2_audio_pooled"]}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["score", "evaluate"])
    ap.add_argument("--final", action="store_true", help="required: this reads the sealed half (CLAUDE.md rule 3)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--smoke-dev", type=int, default=0, help="score N development (dev_x) clips into smoke_dev.csv to test the code; not E-015")
    args = ap.parse_args(argv)
    if args.smoke_dev > 0 and args.cmd == "score":
        return cmd_score(args)
    if not args.final:
        print("refused: the sealed half of FakeAVCeleb needs --final", file=sys.stderr)
        return 2
    return cmd_score(args) if args.cmd == "score" else cmd_evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
