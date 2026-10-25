"""Summary of the sealed evaluation (E-014): staircase table and the verdicts of the three confirmatory tests, from the report files only.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/freeze/summarise_final.py

Reads data/_final/eval/<tag>.json (dfd.eval.run --final) and data/_final/eval/compare_T1.json, compare_T2.json, compare_T3.json
(dfd.eval.run --compare --final). Writes data/_results/E-014/summary.json and sealed_block.md. Verdict rules as declared (E-014, D-061, D-066):
T1 and T2 improved if the lower bound of the 95 percent interval is above 0; T3 superior if above 0, non-inferior if above -delta (delta 0.0100),
"not distinguished, non-inferior by the rule of D-025" if the interval contains 0 but not -delta; the same rule at the 98.33 percent level gives the
robustness line. Numbers are copied, nothing is recomputed.
"""
from __future__ import annotations

import json
from pathlib import Path

EV = Path("data/_final/eval")
OUT = Path("data/_results/E-014")
DELTA = 0.0100


def f(b):
    return f"{b['estimate']:.4f} [{b['low']:.4f}, {b['high']:.4f}]"


def verdict(kind, low, high):
    if kind in ("T1", "T2"):
        return "improved" if low > 0 else "not shown"
    if low > 0:
        return "superior (and non-inferior)"
    if low > -DELTA:
        return "not distinguished, non-inferior by the rule of D-025" if high >= 0 else "non-inferior"
    return "not shown non-inferior"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stair, lines = [], ["| tag | modality | dataset | split | n | groups | AUC [95% interval] | seed sd |", "|---|---|---|---|---|---|---|---|"]
    for p in sorted(EV.glob("*.json")):
        if p.name.startswith("compare_"):
            continue
        d = json.loads(p.read_text())
        for c in d["candidates"]:
            for b in c["blocks"]:
                if b["split"] not in ("test_in", "test_x"):
                    continue
                if "auc" not in b["metrics"]:   # one-class block (Celeb-DF++ has no real videos, D-045): no AUC by construction
                    lines.append(f"| {c['iteration']} | {c['modality']} | {b['dataset']} | {b['split']} | {b['n_samples']} | {b['n_groups']} | one class only, no AUC | n/a |")
                    continue
                a = b["metrics"]["auc"]
                stair.append(dict(tag=c["iteration"], modality=c["modality"], dataset=b["dataset"], split=b["split"], n=b["n_samples"], groups=b["n_groups"], auc=a))
                lines.append(f"| {c['iteration']} | {c['modality']} | {b['dataset']} | {b['split']} | {b['n_samples']} | {b['n_groups']} | {f(a)} | {a['seed_sd']:.4f} |"
                             if a.get("seed_sd") is not None else f"| {c['iteration']} | {c['modality']} | {b['dataset']} | {b['split']} | {b['n_samples']} | {b['n_groups']} | {f(a)} | n/a |")
    tests = {}
    tl = ["", "| test | block | A (final system) | B | A - B, 95% (declared) | 98.33% (robustness) | verdict (declared rule) | at 98.33% |", "|---|---|---|---|---|---|---|---|"]
    for k, name in (("T1", "T1 video, final minus I-0"), ("T2", "T2 speech, final minus I-0"), ("T3", "T3 video, final minus baseline (delta 0.0100)")):
        p = EV / f"compare_{k}.json"
        if not p.is_file():
            continue
        r = json.loads(p.read_text())
        d, rb = r["difference"], r["robustness"][0]
        v, vr = verdict(k, d["low"], d["high"]), verdict(k, rb["low"], rb["high"])
        tests[k] = dict(name=name, block=r["block"], a=r["a"]["estimate"], b=r["b"]["estimate"], diff=d, robust=rb, verdict=v, verdict_at_robust_level=vr, robust_agrees=(v == vr))
        tl.append(f"| {name} | {r['block']['dataset']}/{r['block']['split']} n={r['block']['n_samples']} groups={r['block']['n_groups']} | {f(r['a'])} | {f(r['b'])} | "
                  f"{d['estimate']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}] | [{rb['low']:+.4f}, {rb['high']:+.4f}] | {v} | {vr} |")
    (OUT / "summary.json").write_text(json.dumps(dict(staircase=stair, tests=tests, delta=DELTA), indent=1))
    (OUT / "sealed_block.md").write_text("\n".join(lines + tl) + "\n", encoding="utf-8")
    print("\n".join(lines + tl))


if __name__ == "__main__":
    main()
