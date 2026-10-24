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
from .guard import GuardError, check_guard
from .ledger import DEFAULT_LEDGER, append_entries
from .metrics import CLIP_EPS, ECE_BINS, OneClassError, make_evaluator
from .scorefile import ScoreFileError, ScoreTable, read_scores
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
                f = fits[r]
                s, z = sub.score[idx], sub.logit[idx]
                pc = apply_temperature(z, f["temperature"])
                items = {
                    "accuracy_at_0.5": make_evaluator("accuracy", y, s, threshold=0.5),
                    "log_loss": make_evaluator("log_loss", y, s),
                    "brier": make_evaluator("brier", y, s),
                    "ece": make_evaluator("ece", y, s),
                    "ece_class1": make_evaluator("ece_class1", y, s),
                    "log_loss_cal": make_evaluator("log_loss", y, pc),
                    "brier_cal": make_evaluator("brier", y, pc),
                    "ece_cal": make_evaluator("ece", y, pc),
                    "ece_class1_cal": make_evaluator("ece_class1", y, pc),
                }
                if both:
                    items["auc"] = make_evaluator("auc", y, s)
                    items["eer"] = make_evaluator("eer", y, s)
                    for nm in ("balanced_accuracy", "precision", "recall"):
                        items[nm] = make_evaluator(nm, y, s, threshold=f["threshold"])
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
            blocks.append({
                "dataset": dataset, "split": split, "n_samples": int(y.size),
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
    head = (f"{'iter':<5} {'mod':<7} {'dataset':<14} {'split':<8} {'n':>7} "
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
            lines.append(
                f"{c['iteration']:<5} {c['modality']:<7} {b['dataset'][:14]:<14} "
                f"{b['split']:<8} {b['n_samples']:>7} {auc:<24} {_fmt(m, 'eer'):>6} "
                f"{_fmt(m, 'balanced_accuracy'):>6} {_fmt(m, 'ece'):>6} {_fmt(m, 'ece_cal'):>6} "
                f"{_fmt(m, 'brier'):>6} {_fmt(m, 'brier_cal'):>6}{flag}")
    lines.append("* fitted on these rows (optimistic). BA uses the val threshold.")
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
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        table = read_scores(args.scores)
        wanted = None
        if args.splits:
            wanted = {s.strip() for s in args.splits.split(",") if s.strip()}
            bad = wanted - {"train", "val", "test_in", "test_x"}
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
