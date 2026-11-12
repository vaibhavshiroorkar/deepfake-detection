"""Export of a run record: the JSON record itself, or a Markdown report with the same numbers."""

from __future__ import annotations

import json

from .common import LIMITATIONS, METRICS, jsonable


def _f(v, digits=4):
    return "n/a" if v is None else f"{v:.{digits}f}"


def to_json(rec: dict) -> str:
    return json.dumps(jsonable(rec), indent=1, sort_keys=False)


def to_markdown(rec: dict) -> str:
    cfg, sf = rec["config"], rec["score_file"]
    L = [f"# Evaluation Lab run {rec['id']}", ""]
    if cfg.get("label"):
        L += [f"Label: {cfg['label']}", ""]
    if rec["final_guard"]["required"] and rec["state"] == "done":
        L += ["**Final held-out, re-analysis of a logged evaluation.** This run is not a new held-out look.", ""]
    L += [f"State: {rec['state']}", f"Created: {rec['created']}  Started: {rec.get('started')}  "
          f"Finished: {rec.get('finished')}  Runtime: {rec.get('runtime_s')} s", ""]
    if rec.get("error"):
        L += [f"Error ({rec['error']['code']}): {rec['error']['message']}", ""]
    L += ["## Configuration (frozen before computing)", "",
          f"- Score file: `{sf['path']}`", f"- Score file SHA-256: `{sf['sha256']}`", f"- Rows: {sf['n_rows']}",
          f"- Configuration SHA-256: `{rec['config_sha256']}`",
          f"- Iteration / modality: {cfg['iteration']} / {cfg['modality']}",
          f"- Block: dataset `{cfg['dataset']}`, split `{cfg['split']}`",
          f"- Role of the split: {rec['role']}",
          f"- Runs (seeds): {', '.join(cfg['run_ids'])}",
          f"- Purpose: {rec['purpose']}",
          f"- Bootstrap: {cfg['n_boot']} group resamples, seed {cfg['seed']}, 95 percent percentile interval",
          f"- Code state: git HEAD `{rec['environment'].get('git_head')}`, "
          f"{rec['environment'].get('dirty_entries')} dirty entries, Python {rec['environment'].get('python')}", ""]
    th = rec["threshold"]
    L += ["## Threshold", "", f"- Source: {th['source']}", f"- Value: {_f(th['value'], 6)}",
          f"- Provenance: {th['provenance']}",
          f"- Fitted on the evaluated rows: {'yes (optimistic)' if th['fitted_on_these_rows'] else 'no'}"]
    if th.get("per_run"):
        L += ["- Per run: " + ", ".join(f"{r} = {v:.6f}" for r, v in th["per_run"].items())]
    L += [""]
    g = rec.get("final_guard") or {}
    if g.get("required"):
        L += ["## Final-split guard", "", f"- Ledger entries that authorise this re-analysis: {len(g['ledger_entries'])}"]
        for e in g["ledger_entries"]:
            L += [f"  - {e.get('date')} {e.get('iteration')} {e.get('dataset')} {e.get('split')}"]
        L += ["- The lab wrote no ledger entry.", ""]
    d = rec.get("data")
    if d:
        L += ["## Data", "", f"- Samples: {d['n_samples']} ({d['n_fake']} fake, {d['n_real']} real), groups: {d['n_groups']}",
              f"- Skipped rows: {len(d['skipped'])}"]
        lk = d["leakage"]
        L += [f"- Groups shared with train/val rows of the same runs: {lk['group_overlap']['n_shared_groups']}",
              f"- Repeated sample ids (frame rows merged): {lk['duplicate_sample_rows']['n_samples']}",
              "- File hashes: " + ("not checked (no sha256 or file_hash column)" if not lk["hash"]["checked"] else
                                   f"column `{lk['hash']['column']}`, shared with train/val: "
                                   f"{lk['hash']['n_shared_with_development_splits']}, with other splits: "
                                   f"{lk['hash']['n_shared_with_other_splits']}, duplicates inside the block: "
                                   f"{lk['hash']['n_duplicates_within_block']}"), ""]
    m = rec.get("metrics")
    if m:
        L += ["## Metrics (95 percent group-bootstrap interval)", "", "| Metric | Estimate | Low | High |", "|---|---|---|---|"]
        for k in METRICS:
            L += [f"| {k} | {_f(m[k]['estimate'])} | {_f(m[k]['low'])} | {_f(m[k]['high'])} |"]
        L += [""]
    c = rec.get("confusion")
    if c:
        L += ["## Confusion matrix", "", f"({c['aggregation']})", "", "| | predicted fake | predicted real |", "|---|---|---|",
              f"| actual fake | TP {c['tp']} | FN {c['fn']} |", f"| actual real | FP {c['fp']} | TN {c['tn']} |", ""]
    b = rec.get("breakdown")
    if b:
        L += [f"## Breakdown by `{b['column']}`", "", "| Value | n fake | n real | AUC | FPR | FNR |", "|---|---|---|---|---|---|"]
        for r in b["rows"]:
            L += [f"| {r['value']} | {r['n_fake']} | {r['n_real']} | {_f(r['auc'])} | {_f(r['fpr'])} | {_f(r['fnr'])} |"]
        L += [""]
    if d and d["warnings"]:
        L += ["## Warnings", ""] + [f"- {w}" for w in d["warnings"]] + [""]
    L += ["## Limitations", "", LIMITATIONS, ""]
    return "\n".join(L)
