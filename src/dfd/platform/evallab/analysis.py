"""The computation of one lab run: integrity checks, threshold, metrics, curves, breakdown.

Built on dfd.eval: scorefile (reader and split vocabulary), guard (check_compare_guard), ledger (read only),
threshold (max balanced accuracy), metrics (make_evaluator, roc_auc) and bootstrap (group bootstrap with the
seed-averaging rule of PROTOCOL.md section 6). Nothing here writes the ledger, a checkpoint or a threshold of
the product, and the evaluated rows are never used to fit anything (the val_fit threshold comes from val rows).
"""

from __future__ import annotations

import math

import numpy as np

from dfd.eval import bootstrap as bs
from dfd.eval.guard import GuardError, check_compare_guard
from dfd.eval.ledger import read_ledger
from dfd.eval.metrics import OneClassError, make_evaluator, roc_auc
from dfd.eval.run import _block_rows
from dfd.eval.scorefile import DEV_SPLITS, FINAL_SPLITS, ScoreFileError
from dfd.eval.threshold import choose_threshold_max_balanced_accuracy

from .common import BREAKDOWN_COLUMNS, METRICS, ROLES, LabError

ROC_MAX_POINTS = 200
EXAMPLES = 5


class Refused(LabError):
    """The run is refused by an integrity rule (state "refused")."""


# ------------------------------------------------------------------ evaluators

def _one_minus(f):
    return lambda w: 1.0 - f(w)


def metric_evaluators(y, s, t: float) -> dict:
    """Weight-evaluators for all lab metrics of one run. Reuses dfd.eval.metrics.make_evaluator and derives the
    rest from it: specificity = 2 BA - recall, fpr = 1 - specificity, fnr = 1 - recall, f1 = 2 P R / (P + R)."""
    auc = make_evaluator("auc", y, s)
    eer = make_evaluator("eer", y, s)
    prec = make_evaluator("precision", y, s, threshold=t)
    rec = make_evaluator("recall", y, s, threshold=t)
    ba = make_evaluator("balanced_accuracy", y, s, threshold=t)
    acc = make_evaluator("accuracy", y, s, threshold=t)

    def spec(w):
        return 2.0 * ba(w) - rec(w)

    def f1(w):
        p, r = prec(w), rec(w)
        if math.isnan(p) or p + r <= 0:
            return float("nan")
        return 2.0 * p * r / (p + r)

    return {"auc": auc, "eer": eer, "precision": prec, "recall": rec, "specificity": spec,
            "fpr": _one_minus(spec), "fnr": _one_minus(rec), "f1": f1, "accuracy": acc, "balanced_accuracy": ba}


def confusion_at(y, s, t: float) -> dict:
    pred = s >= t
    pos = y == 1
    return {"tp": int((pos & pred).sum()), "fp": int((~pos & pred).sum()),
            "tn": int((~pos & ~pred).sum()), "fn": int((pos & ~pred).sum())}


def roc_curve(y, s):
    """Full ROC (fpr, tpr) with one point per distinct score, starting at (0, 0)."""
    order = np.argsort(-s, kind="stable")
    ss, yy = s[order], y[order]
    tps, fps = np.cumsum(yy), np.cumsum(1 - yy)
    last = np.concatenate((np.flatnonzero(np.diff(ss)), [ss.size - 1]))
    n1, n0 = max(int(yy.sum()), 1), max(int(yy.size - yy.sum()), 1)
    return (np.concatenate(([0.0], fps[last] / n0)), np.concatenate(([0.0], tps[last] / n1)))


def roc_points(y, scores_per_run) -> dict:
    """At most ROC_MAX_POINTS points. One run with few points: exact. Otherwise the per-run curves are
    interpolated on a common fpr grid and averaged over runs (seeds)."""
    curves = [roc_curve(y, s) for s in scores_per_run]
    if len(curves) == 1 and curves[0][0].size <= ROC_MAX_POINTS:
        f, t = curves[0]
    else:
        f = np.linspace(0.0, 1.0, ROC_MAX_POINTS)
        t = np.mean([np.interp(f, cf, ct) for cf, ct in curves], axis=0)
    return {"fpr": [float(v) for v in f], "tpr": [float(v) for v in t]}


# ------------------------------------------------------------------- integrity

def check_final_guard(cfg: dict, sub, file_sha256: str, ledger_path) -> list:
    """Rule (a). Returns the ledger entries that authorise the re-analysis, or raises Refused."""
    if cfg["split"] not in FINAL_SPLITS:
        return []
    if cfg["purpose"] != "final_reanalysis":
        raise Refused("FINAL_PURPOSE", f"{cfg['split']} is a final held-out split: a run on it must declare "
                      "purpose 'final_reanalysis' (the lab only re-analyses a logged final evaluation)")
    block = sub.select((sub.dataset == cfg["dataset"]) & (sub.split == cfg["split"]))
    try:
        check_compare_guard(block, final=True, iteration=cfg["iteration"], ledger_path=ledger_path)
    except GuardError as e:
        raise Refused("FINAL_GUARD", str(e)) from None
    entries = [e for e in read_ledger(ledger_path)
               if e["iteration"] == cfg["iteration"] and e["dataset"] == cfg["dataset"]
               and e.get("split", cfg["split"]) == cfg["split"]]
    if not entries:
        raise Refused("FINAL_GUARD", f"refused: no ledger entry for iteration {cfg['iteration']!r}, dataset "
                      f"{cfg['dataset']!r}, split {cfg['split']!r}")
    if not any(e.get("score_file_sha256") == file_sha256 for e in entries):
        logged = sorted({str(e.get("score_file_sha256"))[:12] for e in entries})
        raise Refused("FINAL_GUARD", "refused: this score file is not the one logged for the final evaluation "
                      f"(file sha256 {file_sha256[:12]}..., ledger {logged}); the lab only re-analyses the logged scores")
    return entries


def leakage_report(cfg: dict, sub, rows, extras) -> dict:
    """Rule (c). `rows` is the aligned index list of the block. Never raises: the caller decides."""
    own = cfg["split"]
    out = {"group_overlap": None, "duplicate_sample_rows": None, "hash": None, "leaks": False}

    block_groups = set(map(str, sub.group_id[rows[0]]))
    other = np.isin(sub.split, [s for s in ("train", "val") if s != own])
    shared = sorted(block_groups & set(map(str, sub.group_id[other])))
    out["group_overlap"] = {
        "checked_against": sorted({s for s in ("train", "val") if s != own and (sub.split == s).any()}),
        "n_shared_groups": len(shared), "examples": shared[:EXAMPLES],
    }
    if shared:
        out["leaks"] = True

    # duplicate sample ids inside the block: raw rows repeated for one (run, sample) are frame rows,
    # merged to one clip row by the reader
    r0 = str(sub.run_id[rows[0]][0])
    er = extras["rows"]
    cnt: dict[str, int] = {}
    for run, ds, sp, sid in zip(er["run_id"], er["dataset"], er["split"], er["sample_id"]):
        if run == r0 and ds == cfg["dataset"] and sp == own:
            cnt[sid] = cnt.get(sid, 0) + 1
    dups = sorted(k for k, v in cnt.items() if v > 1)
    out["duplicate_sample_rows"] = {
        "n_samples": len(dups), "examples": dups[:EXAMPLES],
        "note": "repeated rows of one sample id are frame rows and were merged to one clip row (mean score) by the reader",
    }

    col = extras["hash_column"]
    if col is None:
        out["hash"] = {"column": None, "checked": False, "n_shared_with_development_splits": 0,
                       "n_shared_with_other_splits": 0, "n_duplicates_within_block": 0, "examples": []}
    else:
        in_block, dev_h, other_h = [], set(), set()
        for run, ds, sp, h in zip(er["run_id"], er["dataset"], er["split"], extras["hashes"]):
            if not h or run != r0:
                continue
            if ds == cfg["dataset"] and sp == own:
                in_block.append(h)
            elif sp != own:
                (dev_h if sp in ("train", "val") else other_h).add(h)
        uniq = set(in_block)
        shared_dev = sorted(uniq & dev_h)
        shared_other = sorted(uniq & other_h)
        out["hash"] = {"column": col, "checked": True, "n_shared_with_development_splits": len(shared_dev),
                       "n_shared_with_other_splits": len(shared_other),
                       "n_duplicates_within_block": len(in_block) - len(uniq),
                       "examples": (shared_dev or shared_other)[:EXAMPLES]}
        if shared_dev:
            out["leaks"] = True
    return out


# ---------------------------------------------------------------------- breakdown

def breakdown_table(cfg, extras, block_ids, y, scores_per_run, thresholds):
    cols = extras["breakdown"]
    col = cfg.get("breakdown_column") or next((c for c in BREAKDOWN_COLUMNS if c in cols), None)
    if col is None or col not in cols:
        return None
    er = extras["rows"]
    r0 = er["run_id"][0] if er["run_id"] else None
    value_of = {}
    for run, ds, sp, sid, v in zip(er["run_id"], er["dataset"], er["split"], er["sample_id"], cols[col]):
        if run == r0 and ds == cfg["dataset"] and sp == cfg["split"]:
            value_of.setdefault(sid, v)
    vals = np.array([value_of.get(str(i), "") for i in block_ids], dtype=object)
    rows = []
    for v in sorted({str(x) for x in vals}):
        m = vals == v
        yy = y[m]
        n_fake, n_real = int(yy.sum()), int(yy.size - yy.sum())
        aucs, fprs, fnrs = [], [], []
        for s, t in zip(scores_per_run, thresholds):
            ss = s[m]
            if n_fake and n_real:
                aucs.append(roc_auc(yy, ss))
            if n_real:
                fprs.append(float((ss[yy == 0] >= t).mean()))
            if n_fake:
                fnrs.append(float((ss[yy == 1] < t).mean()))
        rows.append({"value": v if v else "(blank)", "n_fake": n_fake, "n_real": n_real,
                     "auc": float(np.mean(aucs)) if aucs else None,
                     "fpr": float(np.mean(fprs)) if fprs else None,
                     "fnr": float(np.mean(fnrs)) if fnrs else None})
    return {"column": col, "rows": rows}


# ------------------------------------------------------------------------ the run

def compute(cfg: dict, table, extras: dict, *, ledger_path, frozen_sha256: str, progress=lambda *a: None) -> dict:
    """Run the analysis for a validated config. Returns the result fields of the run record, or raises Refused."""
    progress("checks", 0, 6)
    if table.sha256 != frozen_sha256:
        raise Refused("FILE_CHANGED", "the score file changed after the configuration was frozen "
                      f"(sha256 {frozen_sha256[:12]}... then {table.sha256[:12]}...)")
    run_ids = list(cfg["run_ids"])
    sub = table.select((table.iteration == cfg["iteration"]) & (table.modality == cfg["modality"])
                       & np.isin(table.run_id, run_ids))
    have = {str(r) for r in sub.run_id}
    if set(run_ids) - have:
        raise Refused("UNKNOWN_RUN", f"run ids not found for {cfg['iteration']}/{cfg['modality']}: "
                      f"{sorted(set(run_ids) - have)}")
    seeds = [int(sub.seed[sub.run_id == r][0]) for r in run_ids]
    if len(set(seeds)) != len(seeds):
        raise Refused("DUPLICATE_SEEDS", f"two selected runs share a seed ({seeds}); one lab run is one candidate")

    entries = check_final_guard(cfg, sub, frozen_sha256, ledger_path)        # rule (a)
    progress("checks", 1, 6)

    try:
        rows = _block_rows(sub, run_ids, cfg["dataset"], cfg["split"])
    except ScoreFileError as e:
        raise Refused("RUNS_NOT_ALIGNED", str(e)) from None
    y = sub.label[rows[0]]
    groups = sub.group_id[rows[0]]
    ids = sub.sample_id[rows[0]]
    scores = [sub.score[idx].astype(float) for idx in rows]
    n, n_fake = int(y.size), int(y.sum())
    n_real = n - n_fake
    if n_fake == 0 or n_real == 0:
        raise Refused("ONE_CLASS", f"{cfg['dataset']}/{cfg['split']} has one class only; AUC, EER and the rates are undefined")

    leak = leakage_report(cfg, sub, rows, extras)                              # rule (c)
    if leak["leaks"] and cfg["split"] in FINAL_SPLITS:
        parts = []
        if leak["group_overlap"]["n_shared_groups"]:
            parts.append(f"{leak['group_overlap']['n_shared_groups']} group id(s) shared with train/val rows")
        if leak["hash"]["n_shared_with_development_splits"]:
            parts.append(f"{leak['hash']['n_shared_with_development_splits']} file hash(es) shared with train/val rows")
        raise Refused("LEAKAGE", "refused: leakage in a final split: " + "; ".join(parts))
    progress("checks", 2, 6)

    # rule (b): the threshold comes from val rows of the same runs, or is given with its provenance
    progress("threshold", 3, 6)
    warnings = list(table.warnings)
    fitted = False
    per_run = {}
    if cfg["threshold_source"] == "fixed":
        per_run = {r: float(cfg["threshold"]) for r in run_ids}
        provenance = cfg["threshold_provenance"]
    else:
        for r in run_ids:
            m = (sub.run_id == r) & (sub.split == "val")
            if not m.any():
                raise Refused("NO_VAL_ROWS", f"run {r!r} has no val rows: the val_fit threshold is fitted on val")
            try:
                per_run[r] = choose_threshold_max_balanced_accuracy(sub.label[m], sub.score[m]).threshold
            except (OneClassError, ValueError) as e:
                raise Refused("THRESHOLD_FIT", f"run {r!r}: cannot fit the threshold on val ({e})") from None
        fitted = cfg["split"] == "val"
        provenance = ("max balanced accuracy on the val rows of the same runs (dfd.eval.threshold), one threshold per run, "
                      "val rows of all datasets pooled as in dfd.eval.run")
        if fitted:
            warnings.append("the threshold was fitted on these rows (split val): metrics that depend on it are optimistic")
    thresholds = [per_run[r] for r in run_ids]

    # metrics with group bootstrap
    progress("bootstrap", 4, 6)
    ev: dict[str, list] = {k: [] for k in METRICS}
    for s, t in zip(scores, thresholds):
        for k, f in metric_evaluators(y, s, t).items():
            ev[k].append(f)
    res, rs = bs.bootstrap_many(y, groups, ev, n_boot=cfg["n_boot"], seed=cfg["seed"])
    metrics = {k: {**res[k].as_dict(), "per_seed": res[k].seed_estimates} for k in METRICS}
    for k in METRICS:
        if res[k].n_nan:
            warnings.append(f"{k}: {res[k].n_nan} bootstrap replicate(s) undefined (nothing predicted fake) and dropped")

    progress("curves", 5, 6)
    confs = [confusion_at(y, s, t) for s, t in zip(scores, thresholds)]
    if len(confs) == 1:
        confusion = {**confs[0], "aggregation": "single run", "per_run": confs}
    else:
        confusion = {**{k: round(float(np.mean([c[k] for c in confs])), 2) for k in ("tp", "fp", "tn", "fn")},
                     "aggregation": f"mean over {len(confs)} runs (seeds), each at its own threshold", "per_run": confs}
    roc = roc_points(y, scores)
    breakdown = breakdown_table(cfg, extras, ids, y, scores, thresholds)
    if cfg.get("breakdown_column") and breakdown is None:
        warnings.append(f"breakdown column {cfg['breakdown_column']!r} is not in the score file")

    # data-quality warnings
    frac = n_fake / n
    if min(frac, 1 - frac) < 0.10:
        warnings.append(f"class imbalance beyond 90/10 ({n_fake} fake, {n_real} real): accuracy and precision are "
                        "dominated by the larger class")
    if rs.n_groups < 30:
        warnings.append(f"only {rs.n_groups} groups: the group bootstrap interval is unreliable below about 30 groups")
    if cfg["split"] in DEV_SPLITS:
        warnings.append("development slice: for ranking candidates, not a result")
    if cfg["split"] == "train":
        warnings.append("training split: scores on data the model was fitted on are optimistic")
    if cfg["split"] in FINAL_SPLITS:
        warnings.append("final held-out split: this is a re-analysis of a logged evaluation, not a new held-out look")
    if cfg["threshold_source"] == "fixed":
        warnings.append("fixed threshold: its provenance text is recorded as given and was not verified by the lab")
    if cfg["purpose"] == "validation" and cfg["split"] != "val":
        warnings.append(f"purpose 'validation' on split {cfg['split']!r}")
    ov = leak["group_overlap"]
    if ov["n_shared_groups"] and cfg["split"] not in FINAL_SPLITS:
        warnings.append(f"{ov['n_shared_groups']} group id(s) of the evaluated block also occur in "
                        f"{'/'.join(ov['checked_against'])} rows of the same runs: possible leakage")
    if leak["duplicate_sample_rows"]["n_samples"]:
        warnings.append(f"{leak['duplicate_sample_rows']['n_samples']} sample id(s) repeat in the block "
                        "(frame rows merged by the reader)")
    h = leak["hash"]
    if not h["checked"]:
        warnings.append("no sha256 or file_hash column in the score file: duplicate files across splits could not be checked")
    else:
        if h["n_shared_with_development_splits"] and cfg["split"] not in FINAL_SPLITS:
            warnings.append(f"{h['n_shared_with_development_splits']} file hash(es) of the block also occur in train/val rows")
        if h["n_shared_with_other_splits"]:
            warnings.append(f"{h['n_shared_with_other_splits']} file hash(es) of the block also occur in other splits")
        if h["n_duplicates_within_block"]:
            warnings.append(f"{h['n_duplicates_within_block']} duplicate file hash(es) inside the block")

    progress("done", 6, 6)
    return {
        "role": ROLES[cfg["split"]],
        "final_guard": {"required": cfg["split"] in FINAL_SPLITS, "ledger_entries": entries},
        "threshold": {"value": float(np.mean(thresholds)), "source": cfg["threshold_source"], "provenance": provenance,
                      "fitted_on_these_rows": fitted, "per_run": per_run,
                      "aggregation": ("single run" if len(run_ids) == 1
                                      else "mean of the per-run thresholds (each run is evaluated at its own)")},
        "data": {"n_samples": n, "n_fake": n_fake, "n_real": n_real, "n_groups": int(rs.n_groups),
                 "class_balance": {"fake_fraction": frac, "real_fraction": 1 - frac},
                 "skipped": [], "warnings": warnings, "leakage": leak,
                 "run_ids": run_ids, "seeds": seeds},
        "metrics": metrics,
        "protocol": {"n_boot": cfg["n_boot"], "level": bs.LEVEL, "seed": cfg["seed"], "n_redrawn": int(rs.n_redrawn),
                     "resampling": "cluster (group_id) percentile bootstrap, dfd.eval.bootstrap",
                     "seed_rule": "metric per run (seed) per resample, averaged over runs; the interval is that of the average"},
        "confusion": confusion,
        "roc": roc,
        "breakdown": breakdown,
    }
