"""Print compact tables from e028_results.json and e029_results.json (for writing the RESULT lines; no computation of new numbers).

Usage: python scripts/modern_image/summarise.py e028|e029|val
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _env  # noqa: E402


def f(m, k):
    v = m[k]
    return f"{v['est']} [{v['ci'][0]}, {v['ci'][1]}]"


def table(block, scorers, keys=("auc", "pr_auc", "fpr", "fnr", "precision", "recall", "f1")):
    for setname in block:
        for sc in scorers:
            if sc in block[setname]:
                m = block[setname][sc]
                print(f"{setname:32s} {sc:14s} n={m.get('n_fake')}/{m.get('n_real')} " + " ".join(f"{k}={f(m, k)}" for k in keys))


which = sys.argv[1]
if which == "e028":
    R = json.load(open(_env.RES / "e028" / "e028_results.json"))
    print({k: R[k] for k in ("n_rows_scored", "n_expected", "n_fake", "n_real", "partial", "parity")})
    print("decode errors", len(R["decode_errors"]))
    table(R["main"], ["T-I"])
    print("--- masked")
    for mk, blk in R["masked"].items():
        print("##", mk)
        table(blk, ["T-I"], ("auc", "fpr", "fnr"))
    print("--- face path")
    for mk, blk in R["face_path"]["metrics"].items():
        table(blk, ["T-I", "T-V"], ("auc", "fpr", "fnr"))
    print(json.dumps(R["face_path"]["paired"], indent=0)[:3000])
    print("--- cue only")
    table(R["cue_only_scores"], ["format_png", "log_pixels"], ("auc",))
    print("--- reencode", json.dumps(R["reencode_control"], indent=0))
    print("--- descriptive", json.dumps(R["descriptive"], indent=0))
    print("--- errors", json.dumps(R["errors"], indent=0))
elif which == "val":
    R = json.load(open(_env.RES / "e029" / "val_results.json"))
    print(json.dumps(R, indent=1, default=str)[:6000])
else:
    R = json.load(open(_env.RES / "e029" / "e029_results.json"))
    print("chosen", R["chosen"], R["thresholds"])
    sc = list(R["thresholds"].keys())
    table({k: v for k, v in R["main"].items()}, [s for s in sc if s != "T-V"], ("auc", "pr_auc", "fpr", "fnr", "precision", "f1"))
    print("--- paired vs T-I (auc, fpr, fnr)")
    for setname, v in R["paired_vs_T-I"].items():
        for name, m in v.items():
            print(f"{setname:32s} {name:28s} auc={f(m, 'auc')} fpr={f(m, 'fpr')} fnr={f(m, 'fnr')}")
    print("--- seed mean", json.dumps(R["seed_mean"], indent=0)[:8000])
    print("--- overlap groups", json.dumps(R["by_overlap_group"], indent=0))
    print("--- masked")
    for mk, blk in R["masked"].items():
        print("##", mk)
        table(blk, [R["chosen"], "T-I"], ("auc", "fpr", "fnr"))
    print(json.dumps(R["masked_paired"], indent=0)[:4000])
    print("--- face", json.dumps(R["face_path"], indent=0)[:6000])
    print("--- reencode", json.dumps({k: v for k, v in R["reencode_control"].items()}, indent=0)[:6000])
    print("--- errors", json.dumps(R["errors_chosen"], indent=0))
