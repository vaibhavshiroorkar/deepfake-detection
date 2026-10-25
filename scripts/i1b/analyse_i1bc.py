"""E-016 (I-1b) and E-017 (I-1c): paired comparison with the I-1 fine-tuned IMG-17 (3 seeds) on validation and dev_x, and the entry rule of D-062 (c).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/i1b/analyse_i1bc.py [--write]

Entry rule (EXPERIMENTS.md E-016 and E-017, written before the runs): S1 lower bound of the paired dev_x difference above 0 and S2 the point
estimate of the paired validation difference not below minus the half-width of the 95 percent interval of the I-1 validation AUC. If both levers
pass, the one with the larger dev_x point estimate enters alone (an untested combination does not enter). Exploratory (D-041). Writes
data/_results/I-1b/e016.{md,json}, data/_results/I-1c/e017.{md,json} and, with --write, the result blocks into docs/EXPERIMENTS.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dfd.eval import bootstrap as B

I1, I1B, I1C = Path("data/_results/I-1"), Path("data/_results/I-1b"), Path("data/_results/I-1c")
SPL = (("validation", "val"), ("dev_x", "dev"))


def fmt(b):
    return f"{b['estimate']:.4f} [{b['low']:.4f}, {b['high']:.4f}]"


def dfmt(b):
    return f"{b['estimate']:+.4f} [{b['low']:+.4f}, {b['high']:+.4f}]"


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load(dirs):
    return [np.load(d / "scores.npz", allow_pickle=True) for d in dirs]


def stack(z, key):
    return np.array([x[f"{key}_scores"] for x in z])


def compare(name, new, base, ref):
    """paired bootstrap of new minus base per split; entry rule."""
    res = {}
    for nm, key in SPL:
        y, g = ref[f"{key}_y"], ref[f"{key}_group"]
        sn, sb = new[nm], base[nm]
        ok = np.isfinite(sn).all(axis=0) & np.isfinite(sb).all(axis=0)
        d, bn, bb = B.paired_bootstrap(y[ok], sn[:, ok], sb[:, ok], g[ok], "auc", n_boot=B.N_BOOT)
        res[nm] = dict(new=bn.as_dict(), base=bb.as_dict(), diff=d.as_dict(), n_dropped_nonfinite=int((~ok).sum()))
    hw = (res["validation"]["base"]["high"] - res["validation"]["base"]["low"]) / 2
    s1 = res["dev_x"]["diff"]["low"] > 0
    s2 = res["validation"]["diff"]["estimate"] >= -hw
    return dict(splits=res, val_halfwidth_base=hw, S1=bool(s1), S2=bool(s2), enters=bool(s1 and s2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    base_z = load([I1 / f"IMG-17_lr1e-05_s{s}" for s in range(3)])
    ref = base_z[0]
    for z in base_z[1:]:
        for key in ("val", "dev"):
            assert np.array_equal(z[f"{key}_sample_id"], ref[f"{key}_sample_id"])
    base = {nm: stack(base_z, key) for nm, key in SPL}
    out = {}

    # ---- I-1b
    if all((I1B / f"IMG-17_lr1e-05_s{s}_aug" / "scores.npz").is_file() for s in range(3)):
        z = load([I1B / f"IMG-17_lr1e-05_s{s}_aug" for s in range(3)])
        for x in z:
            for key in ("val", "dev"):
                assert np.array_equal(x[f"{key}_sample_id"], ref[f"{key}_sample_id"])
        new = {nm: stack(z, key) for nm, key in SPL}
        r = compare("I-1b", new, base, ref)
        r["runs"] = [json.loads((I1B / f"IMG-17_lr1e-05_s{s}_aug" / "result.json").read_text()) for s in range(3)]
        r["train_subset_auc"] = [x["train_subset_auc"] for x in r["runs"]]
        out["I-1b"] = r

    # ---- I-1c
    def member(c, s):
        return (I1 if (c == "IMG-17" or s == 0) else I1C) / f"{c}_lr1e-05_s{s}"
    try:
        mem = {c: load([member(c, s) for s in range(3)]) for c in ("IMG-17", "IMG-08", "IMG-07")}
    except FileNotFoundError:
        mem = None
    if mem:
        for c, zz in mem.items():
            for x in zz:
                for key in ("val", "dev"):
                    assert np.array_equal(x[f"{key}_sample_id"], ref[f"{key}_sample_id"]), c
        ens3 = {nm: np.mean([logit(stack(mem[c], key)) for c in mem], axis=0) for nm, key in SPL}
        ens2 = {nm: np.mean([logit(stack(mem[c], key)) for c in ("IMG-17", "IMG-08")], axis=0) for nm, key in SPL}
        r = compare("I-1c", ens3, base, ref)
        r["secondary_IMG17_IMG08"] = compare("I-1c-2", ens2, base, ref)
        r["members"] = {}
        for c, zz in mem.items():
            sc = {nm: stack(zz, key) for nm, key in SPL}
            r["members"][c] = {nm: B.cluster_bootstrap(ref[f"{key}_y"], sc[nm], ref[f"{key}_group"], "auc", n_boot=B.N_BOOT).as_dict() for nm, key in SPL}
        out["I-1c"] = r

    # ---- entry decision
    cand = [k for k in ("I-1b", "I-1c") if k in out and out[k]["enters"]]
    if len(cand) == 2:
        win = max(cand, key=lambda k: out[k]["splits"]["dev_x"]["diff"]["estimate"])
    else:
        win = cand[0] if cand else None
    out["entry"] = dict(candidates_passing=cand, enters=win)
    Path("data/_results/I-1b").mkdir(parents=True, exist_ok=True)
    Path("data/_results/I-1c").mkdir(parents=True, exist_ok=True)
    (I1B / "e016_e017.json").write_text(json.dumps(out, indent=1, default=float))
    for k, d in (("I-1b", I1B), ("I-1c", I1C)):
        if k in out:
            (d / ("e016.json" if k == "I-1b" else "e017.json")).write_text(json.dumps(dict(out[k], entry=out["entry"]), indent=1, default=float))

    blocks = {}
    if "I-1b" in out:
        r = out["I-1b"]
        md = ["| split | I-1b (3 seeds) | I-1 IMG-17 (3 seeds) | I-1b minus I-1, paired |", "|---|---|---|---|"]
        for nm, _ in SPL:
            s = r["splits"][nm]
            md.append(f"| {nm} | {fmt(s['new'])} (seed sd {s['new']['seed_sd']:.4f}) | {fmt(s['base'])} | {dfmt(s['diff'])} |")
        md += ["", "Per seed: " + "; ".join(f"seed {x['seed']}: validation {x['val_auc']:.4f}, dev_x {x['devx_auc']:.4f}, train subset {x['train_subset_auc']:.4f}, training {x['train_seconds']:.0f} s, wall {x['wall_seconds']:.0f} s, peak VRAM {x['peak_vram_gb']} GiB, non-finite losses {x['nonfinite_losses']}, status {x['status']}" for x in r["runs"]) + ".", "",
               f"S1 (lower bound of the dev_x difference above 0): {'met' if r['S1'] else 'not met'} ({r['splits']['dev_x']['diff']['low']:+.4f}). S2 (validation difference not below -{r['val_halfwidth_base']:.4f}): {'met' if r['S2'] else 'not met'} ({r['splits']['validation']['diff']['estimate']:+.4f}).", "",
               f"**Verdict (mechanical, exploratory, D-041): I-1b {'ENTERS' if r['enters'] else 'DOES NOT ENTER'} the final system by the rule of D-062 (c).**"]
        blocks["E-016"] = "\n".join(md)
    if "I-1c" in out:
        r = out["I-1c"]
        md = ["| split | ensemble of 3 (3 seeds) | I-1 IMG-17 (3 seeds) | ensemble minus I-1, paired |", "|---|---|---|---|"]
        for nm, _ in SPL:
            s = r["splits"][nm]
            md.append(f"| {nm} | {fmt(s['new'])} (seed sd {s['new']['seed_sd']:.4f}) | {fmt(s['base'])} | {dfmt(s['diff'])} |")
        md += ["", "Members (fine-tuned, learning rate 1e-5, 3 seeds each): " + "; ".join(f"{c}: validation {fmt(v['validation'])}, dev_x {fmt(v['dev_x'])}" for c, v in r["members"].items()) + ".", ""]
        sec = r["secondary_IMG17_IMG08"]
        md += [f"Secondary diagnostic (does not decide entry), mean of IMG-17 and IMG-08: validation {dfmt(sec['splits']['validation']['diff'])}, dev_x {dfmt(sec['splits']['dev_x']['diff'])} against I-1 IMG-17.", "",
               f"S1 (lower bound of the dev_x difference above 0): {'met' if r['S1'] else 'not met'} ({r['splits']['dev_x']['diff']['low']:+.4f}). S2 (validation difference not below -{r['val_halfwidth_base']:.4f}): {'met' if r['S2'] else 'not met'} ({r['splits']['validation']['diff']['estimate']:+.4f}).", "",
               f"**Verdict (mechanical, exploratory, D-041): I-1c {'ENTERS' if r['enters'] else 'DOES NOT ENTER'} the final system by the rule of D-062 (c).**"]
        blocks["E-017"] = "\n".join(md)
    e = out["entry"]
    txt = f"\n\nEntry decision (D-062 c): lever(s) passing both lines: {', '.join(e['candidates_passing']) or 'none'}; enters the final system: {e['enters'] or 'none, the final system stays as D-061'}."
    for k, v in blocks.items():
        print(f"== {k}\n{v}")
    print(txt)
    if a.write:
        p = Path("docs/EXPERIMENTS.md")
        t = p.read_text(encoding="utf-8")
        for k, v in blocks.items():
            mb, me = f"<!-- {k} RESULT begin (generated by scripts/i1b/analyse_i1bc.py) -->", f"<!-- {k} RESULT end -->"
            body = f"- Result (generated by scripts/i1b/analyse_i1bc.py from data/_results, rerunnable):\n{mb}\n\n{v}{txt}\n\n{me}"
            if mb in t:
                i, j = t.index(mb), t.index(me) + len(me)
                t = t[:i].rstrip("\n").rsplit("\n", 1)[0] + "\n" + body + t[j:]
            else:
                old = f"- Result: (written by script after the run, {k})"
                assert old in t
                t = t.replace(old, body)
        p.write_text(t, encoding="utf-8")


if __name__ == "__main__":
    main()
