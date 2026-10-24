"""Train the linear heads of a task and rank its candidates by the D-036 rule, by script (no choice is made by hand).

    PYTHONPATH=src .venv/Scripts/python.exe -m dfd.sweep.rank --task T-V

For every candidate: 3 seeds of the head (heads.py), clip-level scores on the in-dataset validation split and on dev_x, then
  * AUC on dev_x with the group-bootstrap interval of PROTOCOL section 6 (2,000 resamples, seed-averaged);
  * the leader is the candidate with the highest mean dev_x AUC; the tie set is every candidate whose paired 95 percent interval
    (candidate minus leader) includes zero; the winner is the cheapest member of the tie set (lowest end-to-end extraction time per
    1,000 inputs, then fewer parameters);
  * the candidates for I-1 are the tie set (cheapest first) followed by the rest by mean dev_x AUC (ordering decided autonomously,
    ITERATIONS.md); the D-042 fit rule is applied later and is not part of the ranking;
  * a sensitivity check beside the bootstrap (D-035): an exact paired sign-flip permutation test over groups on the AUC of the
    seed-averaged scores (Monte Carlo above 16 groups); it never changes the selection;
  * the pre-declared success and failure criteria of the experiment entry, evaluated mechanically;
  * the from-scratch reference is read from data/_results/<task>/reference_scores.npz if it exists; it is reported and excluded.
Results go to data/_results/<task>/ (rank.json, rank.md, scores.npz).
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from dfd.eval import bootstrap as B
from dfd.eval import calibration as C
from dfd.eval import metrics as M
from dfd.eval.threshold import choose_threshold_max_balanced_accuracy
from dfd.sweep import heads
from dfd.sweep.tasks import RESULTS, TASKS, Task, feature_path

SEEDS = (0, 1, 2)


# ---------------------------------------------------------------- loading
def load_split(cand: str, task: Task, dataset: str) -> tuple[heads.Split, dict]:
    p = feature_path(cand, task, dataset)
    z = np.load(p, allow_pickle=False)
    meta = json.loads(p.with_suffix(".json").read_text())
    f = z["feats"]
    if f.ndim == 2:
        f = f[:, None, :]
        mask = np.ones(f.shape[:2], bool)
    else:
        mask = z["mask"]
    n, t, d = f.shape
    clip_of = np.repeat(np.arange(n), t).reshape(n, t)[mask]
    X = f[mask].astype(np.float32)
    return heads.Split(X, clip_of, z["label"].astype(int), z["group_id"].astype(str), z["sample_id"].astype(str)), meta


def cost_of(cand: str, task: Task) -> dict:
    """End-to-end extraction cost per 1,000 inputs and the parameter count, from the extraction sidecars (one run each)."""
    rec = RESULTS / "cost_record.json"
    if rec.is_file():
        r = json.loads(rec.read_text()).get(cand)
        if r:
            return dict(s_per_1000=r["end_to_end_s_per_1000"], params=r["params"], source="cost_record.json", peak_vram_gb=r.get("peak_vram_gb"))
    secs = n = 0
    params = dim = size = vram = None
    for ds in (task.train, task.val, task.devx):
        m = json.loads(feature_path(cand, task, ds).with_suffix(".json").read_text())
        secs += m["seconds"]
        n += m.get("n_frames", m.get("n_items", 0))
        params, dim, size = m["params"], m["dim"], m.get("input_size")
        vram = max(vram or 0, m.get("peak_vram_gb", 0))
    return dict(s_per_1000=1000 * secs / max(n, 1), params=params, dim=dim, input_size=size, peak_vram_gb=vram, source="extraction sidecar (end-to-end, one run)")


# ---------------------------------------------------------------- statistics
def sign_flip_p(y, groups, a, b, max_exact=16, n_mc=20000, seed=20261025) -> dict:
    """Paired sign-flip permutation test over groups on the AUC of the seed-averaged scores of two candidates.

    Under the null the two candidates are exchangeable within a group, so swapping all scores of a group between them does not
    change the distribution of the difference. Exact (all 2^G swaps, half by symmetry) up to `max_exact` groups, Monte Carlo above.
    """
    a, b = np.asarray(a).mean(0), np.asarray(b).mean(0)
    ug, gi = np.unique(groups, return_inverse=True)
    g = ug.size
    obs = M.roc_auc(y, a) - M.roc_auc(y, b)
    if g < 6:
        return dict(p=None, obs=obs, n_groups=int(g), note="fewer than 6 groups: the permutation test cannot reach 0.05")
    if g <= max_exact:
        masks = (np.array(m, bool) for m in itertools.product([False, True], repeat=g - 1))
        masks = ((np.concatenate([[False], m])) for m in masks)  # fix group 0 unswapped: the complement gives the negated difference
        total, count = 2 ** (g - 1), 0
        exact = True
    else:
        rng = np.random.default_rng(seed)
        masks = (rng.random(g) < 0.5 for _ in range(n_mc))
        total, count, exact = n_mc, 0, False
    for m in masks:
        flip = m[gi]
        d = M.roc_auc(y, np.where(flip, b, a)) - M.roc_auc(y, np.where(flip, a, b))
        count += abs(d) >= abs(obs) - 1e-12
    return dict(p=float(count / total) if exact else float((count + 1) / (total + 1)), obs=float(obs), n_groups=int(g), exact=exact)


def select(cands: list[str], scores: dict, y, groups, costs: dict, n_boot: int) -> dict:
    """The D-036 rule. `scores[c]` has shape (n_seeds, n)."""
    boot = {c: B.cluster_bootstrap(y, scores[c], groups, "auc", n_boot=n_boot) for c in cands}
    leader = max(cands, key=lambda c: boot[c].estimate)
    diffs, perm = {}, {}
    for c in cands:
        if c == leader:
            continue
        d, _, _ = B.paired_bootstrap(y, scores[c], scores[leader], groups, "auc", n_boot=n_boot)
        diffs[c] = d
        perm[c] = sign_flip_p(y, groups, scores[c], scores[leader])
    tie = [leader] + [c for c in cands if c != leader and diffs[c].contains(0.0)]
    key = lambda c: (costs[c]["s_per_1000"], costs[c]["params"])  # noqa: E731
    tie_sorted = sorted(tie, key=key)
    winner = tie_sorted[0]
    rest = sorted([c for c in cands if c not in tie], key=lambda c: -boot[c].estimate)
    return dict(boot=boot, leader=leader, diffs=diffs, perm=perm, tie_set=tie_sorted, winner=winner,
                i1_order=tie_sorted + rest, cost_decided=len(tie) > 1)


def criteria(task: Task, cands, scores, y, groups, boot, n_boot, reference=None) -> dict:
    """The success and failure criteria written in the experiment entry before the run, evaluated mechanically."""
    fa = [c for c in task.family_a if c in cands]
    fb = [c for c in task.family_b if c in cands]
    out = {}
    best_all = max(cands, key=lambda c: boot[c].estimate)
    out["best_candidate"] = best_all
    out["best_lower"] = boot[best_all].low
    out["b_lower_above"] = dict(threshold=task.success_lower, met=bool(boot[best_all].low > task.success_lower))
    out["inconclusive"] = bool(boot[best_all].low <= task.inconclusive_below)
    if fa and fb:
        ba = max(fa, key=lambda c: boot[c].estimate)
        bb = max(fb, key=lambda c: boot[c].estimate)
        d, _, _ = B.paired_bootstrap(y, scores[ba], scores[bb], groups, "auc", n_boot=n_boot)
        out["a_family_vs_baseline"] = dict(a=ba, b=bb, **d.as_dict(), met=bool(d.low > 0), includes_zero=bool(d.contains(0.0)))
    if task.reference_needed:
        if reference is None:
            out["a_family_vs_reference"] = dict(status="NOT EVALUATED: from-scratch reference scores are missing")
        else:
            d, _, _ = B.paired_bootstrap(y, scores[max(fa, key=lambda c: boot[c].estimate)], reference, groups, "auc", n_boot=n_boot)
            out["a_family_vs_reference"] = dict(**d.as_dict(), met=bool(d.low > 0))
    parts = [out.get("a_family_vs_baseline", {}).get("met")]
    if task.reference_needed:
        parts.append(out["a_family_vs_reference"].get("met"))
    parts.append(out["b_lower_above"]["met"])
    if any(p is None for p in parts):
        out["verdict"] = "NOT EVALUATED (a comparison could not be made)"
    else:
        out["verdict"] = "SUPPORTED" if all(parts) else ("INCONCLUSIVE (best lower bound at or below %.2f)" % task.inconclusive_below if out["inconclusive"] else "NOT SUPPORTED")
    return out


# ---------------------------------------------------------------- run
def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def run_task(task: Task, seeds=SEEDS, n_boot=B.N_BOOT, out_dir: Path | None = None, candidates=None) -> dict:
    torch.set_num_threads(4)
    t0 = time.perf_counter()
    out = (out_dir or RESULTS) / task.id
    out.mkdir(parents=True, exist_ok=True)
    cands = list(candidates) if candidates else list(task.candidates)
    per, dev_scores, val_scores, costs, ref_ids = {}, {}, {}, {}, None
    y_dev = g_dev = y_val = sid_dev = None
    for c in cands:
        tr, _ = load_split(c, task, task.train)
        va, _ = load_split(c, task, task.val)
        dv, _ = load_split(c, task, task.devx)
        if sid_dev is None:
            sid_dev, y_dev, g_dev, y_val = dv.sample_id, dv.y, dv.groups, va.y
        elif not np.array_equal(sid_dev, dv.sample_id):
            raise RuntimeError(f"{c}: dev_x sample order differs from the first candidate")
        costs[c] = cost_of(c, task)
        vs, ds, hr = [], [], []
        for s in seeds:
            h = heads.train_head(tr, va, s)
            hr.append(h)
            vs.append(heads.clip_scores(h.prob_rows(va.X), va.clip_of, va.y.size))
            ds.append(heads.clip_scores(h.prob_rows(dv.X), dv.clip_of, dv.y.size))
        val_scores[c], dev_scores[c] = np.array(vs), np.array(ds)
        per[c] = dict(best_epochs=[h.best_epoch for h in hr], epochs=[h.epochs for h in hr], val_auc_seed=[h.val_auc for h in hr])
        del tr
    # clips without a score in any candidate are excluded for all (face detection does not depend on the encoder)
    ok_dev = ~np.any([np.isnan(dev_scores[c]).any(0) for c in cands], axis=0)
    ok_val = ~np.any([np.isnan(val_scores[c]).any(0) for c in cands], axis=0)
    y, g = y_dev[ok_dev], g_dev[ok_dev]
    sc = {c: dev_scores[c][:, ok_dev] for c in cands}
    reference = None
    ref_file = out / "reference_scores.npz"
    if ref_file.is_file():
        z = np.load(ref_file, allow_pickle=False)
        order = {s: i for i, s in enumerate(z["sample_id"])}
        reference = np.asarray(z["scores"])[:, [order[s] for s in sid_dev[ok_dev]]]
    sel = select(cands, sc, y, g, costs, n_boot)
    crit = criteria(task, cands, sc, y, g, sel["boot"], n_boot, reference)
    for c in cands:
        b = sel["boot"][c]
        yv = y_val[ok_val]
        vauc = [M.roc_auc(yv, val_scores[c][k][ok_val]) for k in range(len(seeds))]
        eers, bas, ece0, ece1, br0, br1, temps = [], [], [], [], [], [], []
        for k in range(len(seeds)):
            pv, pd = val_scores[c][k][ok_val], sc[c][k]
            thr = choose_threshold_max_balanced_accuracy(yv, pv).threshold
            T = C.fit_temperature(C.logit(pv), yv).temperature
            pd_cal = C.sigmoid(C.logit(pd) / T)
            eers.append(M.eer(y, pd))
            bas.append(M.balanced_accuracy(y, pd, thr))
            ece0.append(M.ece(y, pd)); ece1.append(M.ece(y, pd_cal))
            br0.append(M.brier(y, pd)); br1.append(M.brier(y, pd_cal)); temps.append(T)
        mean = lambda v: float(np.mean(v))  # noqa: E731
        per[c].update(dev_auc=b.as_dict(), val_auc_mean=mean(vauc), val_auc_sd=float(np.std(vauc, ddof=1)),
                      gap=mean(vauc) - b.estimate, dev_eer=mean(eers), dev_balacc_at_val_threshold=mean(bas),
                      dev_ece_before=mean(ece0), dev_ece_after=mean(ece1), dev_brier_before=mean(br0), dev_brier_after=mean(br1),
                      temperature_mean=mean(temps), cost=costs[c],
                      diff_vs_leader=None if c == sel["leader"] else sel["diffs"][c].as_dict(),
                      permutation_vs_leader=None if c == sel["leader"] else sel["perm"][c])
    res = dict(task=task.id, experiment=task.experiment, title=task.title, date=time.strftime("%Y-%m-%d %H:%M"), commit=git_commit(),
               torch=torch.__version__, seeds=list(seeds), n_boot=n_boot, bootstrap_seed=B.BASE_SEED,
               head=dict(lr=heads.LR, weight_decay=heads.WD, batch=heads.BATCH, max_epochs=heads.MAX_EPOCHS, patience=heads.PATIENCE),
               data=dict(train=task.train, val=task.val, devx=task.devx, n_dev=int(ok_dev.sum()), n_dev_excluded_no_score=int((~ok_dev).sum()),
                         n_val=int(ok_val.sum()), n_dev_groups=int(np.unique(g).size), dev_real=int((y == 0).sum()), dev_fake=int((y == 1).sum())),
               leader=sel["leader"], tie_set=sel["tie_set"], winner=sel["winner"], i1_order=sel["i1_order"], cost_decided=sel["cost_decided"],
               criteria=crit, reference="used" if reference is not None else "not run (excluded from selection in any case)",
               partial=bool(candidates) and set(cands) != set(task.candidates), candidates=per, seconds=round(time.perf_counter() - t0, 1),
               command=f"PYTHONPATH=src .venv/Scripts/python.exe -m dfd.sweep.rank --task {task.id}")
    (out / "rank.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    np.savez(out / "scores.npz", sample_id=sid_dev[ok_dev], y=y, group=g,
             **{f"dev_{c}": sc[c] for c in cands}, **{f"val_{c}": val_scores[c][:, ok_val] for c in cands}, val_y=y_val[ok_val])
    (out / "rank.md").write_text(render_md(res))
    return res


def render_md(res: dict) -> str:
    L = [f"### {res['experiment']} ranking ({res['task']}): {res['title']}",
         f"Run {res['date']}, commit {res['commit']}, seeds {res['seeds']}, {res['n_boot']} bootstrap resamples (base seed {res['bootstrap_seed']}), "
         f"head: linear, class-balanced BCE, AdamW lr {res['head']['lr']}, weight decay {res['head']['weight_decay']}, batch {res['head']['batch']}, "
         f"at most {res['head']['max_epochs']} epochs, patience {res['head']['patience']}. Train {res['data']['train']}, validation {res['data']['val']}, "
         f"dev_x {res['data']['devx']}: {res['data']['n_dev']} clips ({res['data']['dev_real']} real, {res['data']['dev_fake']} fake) in "
         f"{res['data']['n_dev_groups']} groups, {res['data']['n_dev_excluded_no_score']} excluded without a score.", "",
         "| Rank (I-1 order) | Candidate | dev_x AUC (mean of seeds) | 95% interval | seed sd | validation AUC | gap | diff vs leader [95%] | permutation p | tied with leader |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(res["i1_order"], 1):
        p = res["candidates"][c]
        d, pm = p["dev_auc"], p["permutation_vs_leader"]
        diff = "leader" if p["diff_vs_leader"] is None else f"{p['diff_vs_leader']['estimate']:+.3f} [{p['diff_vs_leader']['low']:+.3f}, {p['diff_vs_leader']['high']:+.3f}]"
        pp = "n/a" if pm is None else ("n/a (" + pm["note"] + ")" if pm.get("p") is None else f"{pm['p']:.3f}{'' if pm.get('exact') else ' (MC)'}")
        L.append(f"| {i} | {c} | {d['estimate']:.4f} | [{d['low']:.4f}, {d['high']:.4f}] | {d['seed_sd']:.4f} | {p['val_auc_mean']:.4f} | {p['gap']:+.4f} | {diff} | {pp} | {'yes' if c in res['tie_set'] else 'no'} |")
    L += ["", f"Leader (highest mean dev_x AUC): {res['leader']}. Tie set (paired interval against the leader includes zero), cheapest first: {', '.join(res['tie_set'])}. "
          f"Winner under the D-036 rule (cheapest in the tie set): {res['winner']}" + (" (the cost rule decided)." if res["cost_decided"] else " (no tie)."),
          "Cost used for the rule: end-to-end extraction time per 1,000 inputs, then parameters, from "
          + res["candidates"][res["winner"]]["cost"]["source"] + ".",
          "Order for I-1 (tie set first, then the rest by AUC; ordering decided autonomously, D-042 fit rule applied later): " + ", ".join(res["i1_order"]) + ".",
          f"From-scratch reference: {res['reference']}.", "",
          "| Candidate | dev_x EER | balanced accuracy at the validation threshold | ECE before / after temperature | Brier before / after | mean temperature |", "|---|---|---|---|---|---|"]
    for c in res["i1_order"]:
        p = res["candidates"][c]
        L.append(f"| {c} | {p['dev_eer']:.4f} | {p['dev_balacc_at_val_threshold']:.4f} | {p['dev_ece_before']:.4f} / {p['dev_ece_after']:.4f} | {p['dev_brier_before']:.4f} / {p['dev_brier_after']:.4f} | {p['temperature_mean']:.3f} |")
    cr = res["criteria"]
    L += ["", f"Pre-declared criteria, evaluated by script: best candidate {cr['best_candidate']}, lower bound {cr['best_lower']:.4f} (criterion (b): above {cr['b_lower_above']['threshold']}: {'met' if cr['b_lower_above']['met'] else 'not met'}; inconclusive line: {'crossed' if cr['inconclusive'] else 'not crossed'})."]
    if "a_family_vs_baseline" in cr:
        a = cr["a_family_vs_baseline"]
        L.append(f"Family comparison, best {a['a']} minus best {a['b']}: {a['estimate']:+.4f} [{a['low']:+.4f}, {a['high']:+.4f}], {'interval above zero (met)' if a['met'] else 'interval includes zero or is below (not met)'}.")
    if "a_family_vs_reference" in cr:
        L.append("Comparison with the from-scratch reference: " + json.dumps(cr["a_family_vs_reference"]))
    L += [f"Mechanical verdict on the pre-declared criteria: {cr['verdict']}. This is an exploratory dev_x result (D-041); the sensitivity column is a check, not a second rule.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--n-boot", type=int, default=B.N_BOOT)
    ap.add_argument("--candidates", default=None, help="comma separated subset, for a dry run only; use with --out-dir")
    ap.add_argument("--out-dir", default=None, help="results root (default data/_results); a dry run must not write there")
    a = ap.parse_args(argv)
    if a.candidates and not a.out_dir:
        ap.error("--candidates needs --out-dir so that a partial ranking never lands in data/_results")
    res = run_task(TASKS[a.task], n_boot=a.n_boot, out_dir=Path(a.out_dir) if a.out_dir else None,
                   candidates=a.candidates.split(",") if a.candidates else None)
    print(render_md(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
