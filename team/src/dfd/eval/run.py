"""Stage 2 evaluation entry point.

  python -m dfd.eval.run --scores scores.csv [--final --iteration I-0 [--repeat "<reason>"]]
                         [--ledger path] [--out report.json] [--n-boot 2000] [--splits val,test_x]

Steps: read and validate the score file (scorefile.py); apply the final-test
guard (guard.py); for every candidate and seed fit the threshold (max balanced
accuracy) and the temperature on its val rows; compute all metrics per
(modality, dataset, split) with group bootstrap intervals, before and after
temperature scaling; write the JSON report and print a short table.

Candidate: the runs of one (iteration, modality) in the file; they must differ
in `seed` and cover the same samples. A candidate with several seeds gets the
seed-averaged value and interval of PROTOCOL.md section 6; one seed is the
degenerate case. Val rows of all datasets in a run (normally the training
dataset only) are pooled for the threshold and the temperature. The metrics on
the val split itself use rows the threshold and temperature were fitted on and
are optimistic (flagged in the report). `--splits` limits the evaluated splits
(val is still read to fit); the guard looks only at the splits evaluated.

Splits: train, val, test_in, test_x, dev_x. test_in and test_x are final (guard
and ledger). dev_x (D-035) is a development slice: valid in the score file, not
final, evaluable any number of times without --final, reported as its own block
(role "development split", `development: true`, "(dev)" in the text table), never
pooled with test_x and never the headline.

Paired comparison:
  python -m dfd.eval.run --compare --scores A.csv [--scores-b B.csv]
        --run-a r1,r2,r3 --run-b q1,q2,q3 --modality visual --dataset celebdf --split dev_x
        --metric auc [--n-boot 2000] [--out compare.json]
Side A and side B are one or several run_ids (several = several seeds, the
section 6 rule). With one file both --run-a and --run-b are required; with two
files they default to all runs of the modality in each file. The same resampled
groups are used for both sides; the output is metric(A) - metric(B) with its
percentile interval (schema "dfd-eval-compare/1": block, metric, protocol, a, b,
difference{estimate, low, high, se, contains_zero}). Both sides must cover the
same samples, labels and groups. On test_in or test_x it needs --final and
--iteration (and --iteration-b if B has another tag), and every (iteration,
dataset) must already be in the ledger from a regular final evaluation: a
comparison re-analyses logged scores, it never makes a first look and writes no
ledger entry. dev_x, val and train comparisons need no flag.

Exit codes: 0 ok, 1 invalid input, 2 refused by the guard.

Report JSON (schema "dfd-eval-report/1"):
  schema, created (date), score_file {path, sha256, n_rows, n_frame_rows, warnings},
  protocol {n_boot, level, seed, ece_bins, clip_eps, threshold_rule, aggregation},
  final {flag, iteration, repeat, ledger_entries},
  candidates: [ {
    iteration, modality, run_ids, seeds,
    fits: {run_id: {seed, threshold, val_balanced_accuracy, temperature,
                    temperature_at_bound, val_nll, val_nll_at_T1, n_val, val_datasets}},
    blocks: [ {
      dataset, split, n_samples, n_groups, n_fake, n_real, fitted_on_these_rows (bool),
      n_redrawn, notes,
      metrics: {name: {estimate, low, high, se, seed_sd, n_nan, per_seed: [...]}}
    } ] } ]
Metric names: auc, eer, balanced_accuracy and precision and recall (at the val
threshold), accuracy_at_0.5, log_loss, brier, ece (top-label), ece_class1, and
the same four probability metrics after temperature scaling with suffix _cal
(log_loss_cal, brier_cal, ece_cal, ece_class1_cal). Metrics that need both
classes are omitted for a one-class block (noted).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from pathlib import Path

import numpy as np

from . import bootstrap as bs
from .calibration import apply_temperature, fit_temperature
from .guard import GuardError, check_compare_guard, check_guard
from .ledger import DEFAULT_LEDGER, append_entries
from .metrics import CLIP_EPS, ECE_BINS, OneClassError, make_evaluator
from .scorefile import DEV_SPLITS, SPLITS, ScoreFileError, ScoreTable, read_scores
from .threshold import choose_threshold_max_balanced_accuracy

SCHEMA = "dfd-eval-report/1"
RANK_METRICS = ("auc", "eer", "balanced_accuracy", "precision", "recall")
PROB_METRICS = ("accuracy_at_0.5", "log_loss", "brier", "ece", "ece_class1")
CAL_METRICS = ("log_loss_cal", "brier_cal", "ece_cal", "ece_class1_cal")


def _fit_run(t: ScoreTable, run_id: str) -> dict:
    m = (t.run_id == run_id) & (t.split == "val")
    if not m.any():
        raise ScoreFileError(f"run {run_id!r} has no val rows: the threshold and the "
                             "temperature are fitted on val")
    y, s, z = t.label[m], t.score[m], t.logit[m]
    try:
        ch = choose_threshold_max_balanced_accuracy(y, s)
        tf = fit_temperature(z, y)
    except (OneClassError, ValueError) as e:
        raise ScoreFileError(f"run {run_id!r}: cannot fit on val ({e})") from None
    return {
        "threshold": ch.threshold, "val_balanced_accuracy": ch.balanced_accuracy,
        "temperature": tf.temperature, "temperature_at_bound": tf.at_bound,
        "val_nll": tf.nll, "val_nll_at_T1": tf.nll_at_one,
        "n_val": int(m.sum()), "val_datasets": sorted({str(d) for d in t.dataset[m]}),
    }


def _block_rows(t: ScoreTable, run_ids, dataset, split):
    """Rows of one block per run, in a common sample order; check alignment."""
    out = []
    for r in run_ids:
        idx = np.flatnonzero((t.run_id == r) & (t.dataset == dataset) & (t.split == split))
        if idx.size == 0:
            raise ScoreFileError(f"run {r!r} has no rows for {dataset}/{split}")
        out.append(idx[np.argsort(t.sample_id[idx], kind="stable")])
    ref = out[0]
    for r, idx in zip(run_ids[1:], out[1:]):
        if (idx.size != ref.size or np.any(t.sample_id[idx] != t.sample_id[ref])
                or np.any(t.label[idx] != t.label[ref]) or np.any(t.group_id[idx] != t.group_id[ref])):
            raise ScoreFileError(f"runs {run_ids[0]!r} and {r!r} do not cover the same samples, "
                                 f"labels and groups in {dataset}/{split}")
    return out


def _evaluators(y, s, z, fit, both=True):
    """All named weight-evaluators for one run on one block.

    `fit` is the dict from _fit_run (threshold, temperature); it may be None when
    only metrics that need no fit are asked for (the evaluators needing it are
    then left out).
    """
    items = {
        "accuracy_at_0.5": make_evaluator("accuracy", y, s, threshold=0.5),
        "log_loss": make_evaluator("log_loss", y, s),
        "brier": make_evaluator("brier", y, s),
        "ece": make_evaluator("ece", y, s),
        "ece_class1": make_evaluator("ece_class1", y, s),
    }
    if fit is not None:
        pc = apply_temperature(z, fit["temperature"])
        items["log_loss_cal"] = make_evaluator("log_loss", y, pc)
        items["brier_cal"] = make_evaluator("brier", y, pc)
        items["ece_cal"] = make_evaluator("ece", y, pc)
        items["ece_class1_cal"] = make_evaluator("ece_class1", y, pc)
    if both:
        items["auc"] = make_evaluator("auc", y, s)
        items["eer"] = make_evaluator("eer", y, s)
        if fit is not None:
            for nm in ("balanced_accuracy", "precision", "recall"):
                items[nm] = make_evaluator(nm, y, s, threshold=fit["threshold"])
    return items


ROLES = {
    "train": "training data",
    "val": "validation: threshold and temperature fitted here",
    "test_in": "final, in-dataset (not the headline)",
    "test_x": "final, cross-dataset (headline)",
    "dev_x": "development slice: ranking only, never the headline",
}


def evaluate(t: ScoreTable, *, n_boot=bs.N_BOOT, level=bs.LEVEL, seed=bs.BASE_SEED,
             splits=None) -> list[dict]:
    """Return the list of candidate dicts of the report."""
    cands = []
    pairs = sorted({(str(i), str(m)) for i, m in zip(t.iteration, t.modality)})
    for iteration, modality in pairs:
        sub = t.select((t.iteration == iteration) & (t.modality == modality))
        run_ids = sorted({str(r) for r in sub.run_id}, key=lambda r: (int(sub.seed[sub.run_id == r][0]), r))
        seeds = [int(sub.seed[sub.run_id == r][0]) for r in run_ids]
        if len(set(seeds)) != len(seeds):
            raise ScoreFileError(f"{iteration}/{modality}: two runs share a seed ({seeds}); "
                                 "one file holds one candidate, split candidates into files")
        fits = {r: _fit_run(sub, r) for r in run_ids}
        blocks = []
        keys = sorted({(str(d), str(s)) for d, s in zip(sub.dataset, sub.split)})
        for dataset, split in keys:
            if splits is not None and split not in splits:
                continue
            rows = _block_rows(sub, run_ids, dataset, split)
            y = sub.label[rows[0]]
            groups = sub.group_id[rows[0]]
            both = bool(y.min() == 0 and y.max() == 1)
            ev: dict[str, list] = {}
            for r, idx in zip(run_ids, rows):
                items = _evaluators(y, sub.score[idx], sub.logit[idx], fits[r], both)
                for k, fn in items.items():
                    ev.setdefault(k, []).append(fn)
            res, rs = bs.bootstrap_many(y, groups, ev, n_boot=n_boot, level=level, seed=seed,
                                        require_both_classes=both)
            order = [k for k in (*RANK_METRICS, *PROB_METRICS, *CAL_METRICS) if k in res]
            notes = []
            if not both:
                notes.append("one class only: auc, eer, balanced_accuracy, precision, recall omitted; "
                             "resamples are not redrawn")
            fitted = split == "val"
            if fitted:
                notes.append("threshold and temperature were fitted on these rows: optimistic")
            if split in DEV_SPLITS:
                notes.append("development split (D-035): for ranking candidates, not a result; "
                             "never the headline and never pooled with test_x")
            blocks.append({
                "dataset": dataset, "split": split, "role": ROLES[split],
                "development": split in DEV_SPLITS, "n_samples": int(y.size),
                "n_groups": int(rs.n_groups), "n_fake": int(y.sum()), "n_real": int(y.size - y.sum()),
                "fitted_on_these_rows": fitted, "n_redrawn": int(rs.n_redrawn), "notes": notes,
                "metrics": {k: {**res[k].as_dict(), "per_seed": res[k].seed_estimates} for k in order},
            })
        cands.append({
            "iteration": iteration, "modality": modality, "run_ids": run_ids, "seeds": seeds,
            "fits": {r: {"seed": s, **fits[r]} for r, s in zip(run_ids, seeds)},
            "blocks": blocks,
        })
    return cands


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def _fmt(m, key, digits=3):
    v = m.get(key)
    if v is None or v["estimate"] is None:
        return "-"
    return f"{v['estimate']:.{digits}f}"


def format_table(report: dict) -> str:
    head = (f"{'iter':<5} {'mod':<7} {'dataset':<14} {'split':<12} {'n':>7} "
            f"{'AUC [95% CI]':<24} {'EER':>6} {'BA':>6} {'ECE':>6} {'ECEcal':>6} "
            f"{'Brier':>6} {'Brcal':>6}")
    lines = [head, "-" * len(head)]
    for c in report["candidates"]:
        for b in c["blocks"]:
            m = b["metrics"]
            a = m.get("auc")
            auc = "-" if a is None or a["estimate"] is None else (
                f"{a['estimate']:.3f} [{a['low']:.3f},{a['high']:.3f}]")
            flag = "*" if b["fitted_on_these_rows"] else " "
            split = b["split"] + (" (dev)" if b["development"] else "")
            lines.append(
                f"{c['iteration']:<5} {c['modality']:<7} {b['dataset'][:14]:<14} "
                f"{split:<12} {b['n_samples']:>7} {auc:<24} {_fmt(m, 'eer'):>6} "
                f"{_fmt(m, 'balanced_accuracy'):>6} {_fmt(m, 'ece'):>6} {_fmt(m, 'ece_cal'):>6} "
                f"{_fmt(m, 'brier'):>6} {_fmt(m, 'brier_cal'):>6}{flag}")
    lines.append("* fitted on these rows (optimistic). BA uses the val threshold.")
    if any(b["development"] for c in report["candidates"] for b in c["blocks"]):
        lines.append("(dev) = development split (D-035): used to rank candidates, never the headline.")
    return "\n".join(lines)


def build_parser():
    p = argparse.ArgumentParser(prog="python -m dfd.eval.run", description=__doc__.split("\n")[0])
    p.add_argument("--scores", required=True, help="score CSV (PROTOCOL.md section 2)")
    p.add_argument("--final", action="store_true", help="allow test_in and test_x rows")
    p.add_argument("--iteration", help="iteration tag, required with --final")
    p.add_argument("--repeat", help="reason for a second final evaluation of an iteration and dataset")
    p.add_argument("--ledger", default=str(DEFAULT_LEDGER), help="ledger path")
    p.add_argument("--out", help="write the JSON report here")
    p.add_argument("--n-boot", type=int, default=bs.N_BOOT)
    p.add_argument("--splits", help="comma list of splits to evaluate (default: all in the file)")
    c = p.add_argument_group("paired comparison (--compare)")
    c.add_argument("--compare", action="store_true",
                   help="paired bootstrap of metric(A) - metric(B) on one block; see module docstring")
    c.add_argument("--scores-b", help="score file of side B (default: the --scores file)")
    c.add_argument("--run-a", help="comma list of run_ids of side A (default with two files: all)")
    c.add_argument("--run-b", help="comma list of run_ids of side B (default with two files: all)")
    c.add_argument("--iteration-b", help="iteration tag of side B on a final split (default: --iteration)")
    c.add_argument("--modality")
    c.add_argument("--dataset")
    c.add_argument("--split", choices=SPLITS)
    c.add_argument("--metric", default="auc", choices=COMPARE_METRICS)
    return p


# D-062 (b): three confirmatory tests, so a robustness line at the Bonferroni level 1 - 0.05 / 3 (98.33 percent), read from the SAME bootstrap
# replicates as the declared 95 percent interval (no new draws). It never replaces the declared rule.
ROBUST_LEVEL = 1.0 - 0.05 / 3


def _robust_interval(reps, level) -> dict:
    ok = np.asarray(reps, dtype=float)
    ok = ok[~np.isnan(ok)]
    a = (1.0 - level) / 2.0
    low, high = (float(v) for v in np.quantile(ok, [a, 1.0 - a]))
    return {"level": level, "low": low, "high": high, "n_replicates": int(ok.size), "contains_zero": bool(low <= 0.0 <= high)}


COMPARE_METRICS = (*RANK_METRICS, *PROB_METRICS, *CAL_METRICS)
NEEDS_FIT = ("balanced_accuracy", "precision", "recall", *CAL_METRICS)


def _compare_side(table, spec, modality, dataset, split, metric, label):
    """Resolve one side: its run_ids, rows (common sample order) and evaluators."""
    tm = table.select(table.modality == modality)
    avail = sorted({str(r) for r in tm.run_id})
    if spec:
        run_ids = [r.strip() for r in spec.split(",") if r.strip()]
        missing = [r for r in run_ids if r not in avail]
        if missing:
            raise ScoreFileError(f"side {label}: run_id {missing} not found for modality {modality!r}")
    else:
        run_ids = avail
    if not run_ids:
        raise ScoreFileError(f"side {label}: no runs for modality {modality!r}")
    if len(set(run_ids)) != len(run_ids):
        raise ScoreFileError(f"side {label}: a run_id is listed twice")
    return tm, run_ids


def compare(table_a, table_b, *, run_a, run_b, modality, dataset, split, metric,
            n_boot=bs.N_BOOT, level=bs.LEVEL, seed=bs.BASE_SEED, iteration=None,
            iteration_b=None, final=False, ledger_path=DEFAULT_LEDGER,
            robust_levels=(ROBUST_LEVEL,)) -> dict:
    """Paired bootstrap of metric(A) - metric(B) on one (modality, dataset, split) block.

    Each side is one or several runs (several = several seeds, section 6 rule:
    metric per seed per resample, averaged). Both sides must cover the same
    samples, labels and groups of the block. On a final split the guard applies
    to each side (check_compare_guard).
    """
    sides = []
    for label, tab, spec, tag in (("A", table_a, run_a, iteration), ("B", table_b, run_b, iteration_b or iteration)):
        tm, ids = _compare_side(tab, spec, modality, dataset, split, metric, label)
        keep = np.isin(tm.run_id, ids) & np.isin(tm.split, [split, "val"])
        sub = tm.select(keep)
        check_compare_guard(sub, final=final, iteration=tag, ledger_path=ledger_path)
        sides.append((sub, ids))
    out = {}
    ref = None
    for label, (sub, ids) in zip("AB", sides):
        rows = _block_rows(sub, ids, dataset, split)
        y, g, sid = sub.label[rows[0]], sub.group_id[rows[0]], sub.sample_id[rows[0]]
        if ref is None:
            ref = (y, g, sid)
        elif (y.size != ref[0].size or np.any(y != ref[0]) or np.any(g != ref[1]) or np.any(sid != ref[2])):
            raise ScoreFileError("sides A and B do not cover the same samples, labels and groups "
                                 f"in {dataset}/{split}")
        if not (y.min() == 0 and y.max() == 1) and metric in RANK_METRICS:
            raise ScoreFileError(f"{dataset}/{split} has one class only; {metric} is undefined")
        fits = {r: (_fit_run(sub, r) if metric in NEEDS_FIT else None) for r in ids}
        evs = []
        for r, idx in zip(ids, rows):
            items = _evaluators(y, sub.score[idx], sub.logit[idx], fits[r])
            evs.append(items[metric])
        seeds = [int(sub.seed[sub.run_id == r][0]) for r in ids]
        if len(set(seeds)) != len(seeds):
            raise ScoreFileError(f"side {label}: two runs share a seed")
        out[label] = (evs, ids, seeds, sorted({str(i) for i in sub.iteration}), sub.sha256)
    both = bool(ref[0].min() == 0 and ref[0].max() == 1)
    diff, a, b = bs.paired_from_evaluators(ref[0], ref[1], out["A"][0], out["B"][0], n_boot=n_boot,
                                           level=level, seed=seed, require_both_classes=both)
    def side(label, res):
        _, ids, seeds, its, sha = out[label]
        return {"run_ids": ids, "seeds": seeds, "iterations": its, "score_file_sha256": sha,
                **res.as_dict(), "per_seed": res.seed_estimates}
    return {
        "schema": "dfd-eval-compare/1",
        "created": _dt.date.today().isoformat(),
        "block": {"modality": modality, "dataset": dataset, "split": split, "role": ROLES[split],
                  "development": split in DEV_SPLITS, "n_samples": int(ref[0].size),
                  "n_groups": int(np.unique(ref[1]).size)},
        "metric": metric,
        "protocol": {"n_boot": n_boot, "level": level, "seed": seed,
                     "rule": "same resampled groups for A and B; difference A - B per resample; "
                             "several seeds: metric per seed per resample, averaged"},
        "a": side("A", a), "b": side("B", b),
        "difference": {**diff.as_dict(), "contains_zero": bool(diff.low <= 0.0 <= diff.high)},
        "robustness": [_robust_interval(diff.replicates, lv) for lv in robust_levels],
    }


def format_compare(rep: dict) -> str:
    b, d = rep["block"], rep["difference"]
    dev = " (development split, not the headline)" if b["development"] else ""
    lines = [
        f"paired comparison on {b['modality']} / {b['dataset']} / {b['split']}{dev}, "
        f"n={b['n_samples']}, groups={b['n_groups']}, metric={rep['metric']}",
        f"  A runs {rep['a']['run_ids']}: {rep['a']['estimate']:.4f} "
        f"[{rep['a']['low']:.4f}, {rep['a']['high']:.4f}]",
        f"  B runs {rep['b']['run_ids']}: {rep['b']['estimate']:.4f} "
        f"[{rep['b']['low']:.4f}, {rep['b']['high']:.4f}]",
        f"  A - B = {d['estimate']:+.4f}  {int(rep['protocol']['level'] * 100)}% interval "
        f"[{d['low']:+.4f}, {d['high']:+.4f}]  "
        + ("contains 0: the data do not separate A and B" if d["contains_zero"] else "excludes 0"),
    ]
    for r in rep.get("robustness", []):
        lines.append(f"  robustness (same replicates, level {r['level'] * 100:.2f}%, Bonferroni over three, D-062 b): "
                     f"[{r['low']:+.4f}, {r['high']:+.4f}]  " + ("contains 0" if r["contains_zero"] else "excludes 0"))
    return "\n".join(lines)


def _main_compare(args) -> int:
    for need in ("modality", "dataset", "split"):
        if not getattr(args, need):
            raise ScoreFileError(f"--compare needs --{need}")
    ta = read_scores(args.scores)
    tb = read_scores(args.scores_b) if args.scores_b else ta
    if not args.scores_b and not (args.run_a and args.run_b):
        raise ScoreFileError("--compare on one file needs --run-a and --run-b")
    rep = compare(ta, tb, run_a=args.run_a, run_b=args.run_b, modality=args.modality,
                  dataset=args.dataset, split=args.split, metric=args.metric, n_boot=args.n_boot,
                  iteration=args.iteration, iteration_b=args.iteration_b, final=args.final,
                  ledger_path=args.ledger)
    rep = _clean(rep)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1)
    print(format_compare(rep))
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.compare:
        try:
            return _main_compare(args)
        except GuardError as e:
            print(str(e), file=sys.stderr)
            return 2
        except (ScoreFileError, ValueError, OSError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    try:
        table = read_scores(args.scores)
        wanted = None
        if args.splits:
            wanted = {s.strip() for s in args.splits.split(",") if s.strip()}
            bad = wanted - set(SPLITS)
            if bad:
                raise ScoreFileError(f"unknown splits {sorted(bad)}")
            table = table.select(np.isin(table.split, sorted(wanted | {"val"})))
        decision = check_guard(table, final=args.final, iteration=args.iteration,
                               repeat=args.repeat, ledger_path=args.ledger)
        cands = evaluate(table, n_boot=args.n_boot, splits=wanted)
    except GuardError as e:
        print(str(e), file=sys.stderr)
        return 2
    except (ScoreFileError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    report = {
        "schema": SCHEMA,
        "created": _dt.date.today().isoformat(),
        "score_file": {"path": table.path, "sha256": table.sha256, "n_rows": len(table),
                       "n_frame_rows": table.n_frame_rows, "warnings": table.warnings},
        "protocol": {"n_boot": args.n_boot, "level": bs.LEVEL, "seed": bs.BASE_SEED,
                     "ece_bins": ECE_BINS, "clip_eps": CLIP_EPS,
                     "threshold_rule": "max balanced accuracy on val, midpoints, tie closest to 0.5",
                     "aggregation": "mean of per-frame probabilities"},
        "final": {"flag": bool(args.final), "iteration": args.iteration,
                  "repeat": args.repeat if decision.is_repeat else None,
                  "ledger_entries": decision.entries},
        "candidates": cands,
    }
    report = _clean(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    if decision.entries:
        append_entries(decision.entries, args.ledger)
    print(format_table(report))
    if decision.is_repeat:
        print("repeat recorded in the ledger: list it under 'Mistakes and corrections' in STATUS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
