"""Paired comparison of two finished lab runs (dfd.eval.bootstrap.paired_from_evaluators).

Available only when both runs cover the same samples, labels and groups; otherwise the answer is
{available: false, reason}. No baseline is ever invented: both sides are existing lab runs.
"""

from __future__ import annotations

import numpy as np

from dfd.eval import bootstrap as bs
from dfd.eval.run import _block_rows
from dfd.eval.scorefile import ScoreFileError

from . import analysis
from .common import METRICS, LabError


def _side(runner, rec: dict, label: str):
    """(y, groups, sample_ids, evaluators-by-metric list per run, info) or raises LabError(unavailable reason)."""
    if rec["state"] != "done":
        raise LabError("UNAVAILABLE", f"run {rec['id']} is in state {rec['state']!r}; only finished runs can be compared")
    cfg = rec["config"]
    try:
        table = runner.tables.get(runner.abs_path(rec))
    except (OSError, ScoreFileError) as e:
        raise LabError("UNAVAILABLE", f"side {label}: the score file cannot be read ({e})") from None
    if table.sha256 != rec["score_file"]["sha256"]:
        raise LabError("UNAVAILABLE", f"side {label}: the score file changed since the run was recorded")
    sub = table.select((table.iteration == cfg["iteration"]) & (table.modality == cfg["modality"])
                       & np.isin(table.run_id, cfg["run_ids"]))
    if cfg["split"] in ("test_in", "test_x"):                       # same guard as the run itself
        try:
            analysis.check_final_guard(cfg, sub, table.sha256, runner.ledger_path)
        except analysis.Refused as e:
            raise LabError("UNAVAILABLE", f"side {label}: {e.message}") from None
    try:
        rows = _block_rows(sub, cfg["run_ids"], cfg["dataset"], cfg["split"])
    except ScoreFileError as e:
        raise LabError("UNAVAILABLE", f"side {label}: {e}") from None
    y, g, sid = sub.label[rows[0]], sub.group_id[rows[0]], sub.sample_id[rows[0]]
    per_run = rec["threshold"]["per_run"]
    evs = [analysis.metric_evaluators(y, sub.score[idx].astype(float), per_run[r]) for r, idx in zip(cfg["run_ids"], rows)]
    return y, g, sid, evs


def compare_runs(runner, rec_a: dict, rec_b: dict, metric: str, n_boot: int | None = None, seed: int | None = None) -> dict:
    if metric not in METRICS:
        raise LabError("INVALID_METRIC", f"metric must be one of {list(METRICS)}")
    n_boot = n_boot or rec_a["config"]["n_boot"]
    seed = rec_a["config"]["seed"] if seed is None else seed
    if rec_a["id"] == rec_b["id"]:
        return {"available": False, "reason": "run_a and run_b are the same run"}
    try:
        ya, ga, sa, eva = _side(runner, rec_a, "A")
        yb, gb, sb, evb = _side(runner, rec_b, "B")
    except LabError as e:
        return {"available": False, "reason": e.message}
    if ya.size != yb.size or np.any(sa != sb):
        return {"available": False, "reason": "the two runs do not cover the same samples (different sample ids): "
                "a paired comparison needs identical samples"}
    if np.any(ya != yb):
        return {"available": False, "reason": "the two runs have different labels for the same sample ids"}
    if np.any(ga != gb):
        return {"available": False, "reason": "the two runs have different group ids for the same samples"}
    if ya.min() == ya.max():
        return {"available": False, "reason": "the block has one class only"}
    diff, a, b = bs.paired_from_evaluators(ya, ga, [e[metric] for e in eva], [e[metric] for e in evb],
                                           n_boot=n_boot, seed=seed)
    ca, cb = rec_a["config"], rec_b["config"]
    note = []
    if ca["threshold_source"] != cb["threshold_source"] or rec_a["threshold"]["value"] != rec_b["threshold"]["value"]:
        note.append("the two runs use different thresholds; each side is evaluated at its own recorded threshold")
    return {
        "available": True, "metric": metric,
        "run_a": rec_a["id"], "run_b": rec_b["id"],
        "block": {"dataset": ca["dataset"], "split": ca["split"], "n_samples": int(ya.size),
                  "n_groups": int(np.unique(ga).size)},
        "a": {"run": rec_a["id"], "iteration": ca["iteration"], "run_ids": ca["run_ids"], **{k: a.as_dict()[k] for k in ("estimate", "low", "high")}},
        "b": {"run": rec_b["id"], "iteration": cb["iteration"], "run_ids": cb["run_ids"], **{k: b.as_dict()[k] for k in ("estimate", "low", "high")}},
        "difference": {"estimate": diff.estimate, "low": diff.low, "high": diff.high,
                       "contains_zero": bool(diff.low <= 0.0 <= diff.high)},
        "protocol": {"n_boot": n_boot, "level": bs.LEVEL, "seed": seed,
                     "rule": "paired cluster bootstrap over group_id: the same resampled groups for A and B, "
                             "difference A - B per resample"},
        "notes": note,
    }
