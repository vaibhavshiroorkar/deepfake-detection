"""Write the sweep results into docs/EXPERIMENTS.md after a run finished, and generate the model comparison tables.

    PYTHONPATH=src .venv/Scripts/python.exe -m dfd.sweep.report --task T-V

Reads data/_results/<task>/rank.json (written by dfd.sweep.rank), the extraction sidecars, data/_logs/encoder_weights.json and
data/_logs/step_time_*.json. Nothing here chooses a winner: it formats what rank.py computed. Generated blocks are delimited by
HTML comments and replaced on a rerun; the hypothesis and criteria text of an experiment entry is never touched (rule 2).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from dfd.sweep.tasks import DROPPED, RESULTS, TASKS

EXPERIMENTS = Path("docs/EXPERIMENTS.md")
WEIGHTS_LOG = Path("data/_logs/encoder_weights.json")

LICENCE = {
    "IMG-01": "Apache-2.0 tag; ImageNet-trained, treated as research-only",
    "IMG-04": "apache-2.0 declared by timm; upstream Keras licence NR; research-only",
    "IMG-07": "Apache-2.0; commercial allowed by tag",
    "IMG-08": "Apache-2.0; commercial allowed by tag",
    "IMG-13": "NR in its repo; research-only (accepted, D-038)",
    "IMG-15": "MIT tag; card says research use",
    "IMG-17": "Apache-2.0 tag; commercial allowed by tag",
    "AU-1": "Apache-2.0 tag; commercial allowed by tag",
    "AU-4": "no tag, CC BY-SA 3.0 upstream; research-only (accepted, D-038)",
    "AU-6": "Apache-2.0 tag, MIT upstream",
    "AU-8": "Apache-2.0 tag; commercial allowed by tag",
}


def _replace_block(text: str, tag: str, body: str, anchor_before: str) -> str:
    begin, end = f"<!-- AUTO {tag} begin -->", f"<!-- AUTO {tag} end -->"
    block = f"{begin}\n{body.strip()}\n{end}"
    if begin in text:
        return re.sub(re.escape(begin) + r".*?" + re.escape(end), lambda m: block, text, flags=re.S)
    assert anchor_before in text, anchor_before
    return text.replace(anchor_before, block + "\n\n" + anchor_before, 1)


def _entry_span(text: str, exp: str) -> tuple[int, int]:
    s = text.index(f"## {exp}:")
    e = text.find("\n## ", s + 5)
    return s, (len(text) if e < 0 else e + 1)


def write_experiment_entry(res: dict, md: str, path: Path = EXPERIMENTS) -> None:
    t = path.read_text(encoding="utf-8")
    crlf = "\r\n" in t
    t = t.replace("\r\n", "\n")
    s, e = _entry_span(t, res["experiment"])
    entry = t[s:e]
    day = res["date"].split(" ")[0]
    first = "none yet" in entry or "to be filled" in entry
    if first:  # fill the placeholders once; the hypothesis and criteria lines are not touched
        entry = entry.replace("- Status: PLANNED.", f"- Status: RUN on {day} (automatic ranking, dfd.sweep.rank).", 1)
        entry = re.sub(r"(- Date / Environment / Git commit / Seed\(s\): )to be filled at the run / ENV-DESKTOP-GPU / to be filled / head seeds 0, 1, 2\.",
                       lambda m: f"{m.group(1)}{day} / ENV-DESKTOP-GPU / {res['commit']} / head seeds {', '.join(map(str, res['seeds']))}.", entry)
        entry = entry.replace("- Command and config: to be filled at the run.", f"- Command and config: `{res['command']}`; config in data/_results/{res['task']}/rank.json.", 1)
        entry = entry.replace("- Result (numbers, with seeds and intervals): none yet.", f"- Result (numbers, with seeds and intervals): see the generated block below (data/_results/{res['task']}/rank.md).", 1)
        entry = entry.replace("- Was the hypothesis supported? Why or why not: none yet.",
                              f"- Was the hypothesis supported? Why or why not: mechanical verdict on the pre-declared criteria: {res['criteria']['verdict']} (details in the generated block; exploratory dev_x result, D-041).", 1)
    block = f"<!-- AUTO {res['experiment']} begin -->\n{md.strip()}\n<!-- AUTO {res['experiment']} end -->"
    if f"<!-- AUTO {res['experiment']} begin -->" in entry:
        entry = re.sub(r"<!-- AUTO %s begin -->.*?<!-- AUTO %s end -->" % (res["experiment"], res["experiment"]), lambda m: block, entry, flags=re.S)
    else:
        entry = entry.rstrip("\n") + "\n\n" + block + "\n\n"
    t = t[:s] + entry + t[e:]
    path.write_text(t.replace("\n", "\r\n") if crlf else t, encoding="utf-8", newline="")


def comparison_tables(res: dict) -> str:
    task = TASKS[res["task"]]
    wl = {r["id"]: r for r in json.loads(WEIGHTS_LOG.read_text())} if WEIGHTS_LOG.is_file() else {}
    st = {}
    for f in sorted(Path("data/_logs").glob("step_time_*.json")):
        d = json.loads(f.read_text())
        for r in d["results"]:
            st[(r["cid"], r["batch"])] = (d["tag"], r)
    A = ["| Candidate | Checkpoint and revision | Weights source | Licence and label | SHA-256 (local, verified) | Parameters | Feature size | Input size | FLOPs | Peak VRAM, extraction (GiB) | Extraction s per 1,000 inputs, end-to-end | Model-only s per 1,000 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    Bt = ["| Candidate | Validation AUC | dev_x AUC [95%] | Gap | Rank (I-1 order) | Tied with leader | Entered I-1 candidates | I-1 fit (D-042, video only) |", "|---|---|---|---|---|---|---|---|"]
    for c in res["i1_order"]:
        p, r = res["candidates"][c], wl.get(c, {})
        cost = p["cost"]
        A.append(f"| {c} | {r.get('repo', 'NR')} @ {str(r.get('revision', 'NR'))[:12]} | {r.get('repo', 'NR')} | {LICENCE.get(c, 'NR')} | {str(r.get('sha256_local', 'NR'))[:16]}... | {cost['params']:,} | {cost.get('dim', 'NR')} | {cost.get('input_size', 'NR')} | NR (not measured) | {cost.get('peak_vram_gb', 'NR')} | {cost['s_per_1000']:.1f} | NR (section 7 cost record not run) |")
        d = p["dev_auc"]
        rank = res["i1_order"].index(c) + 1
        fit = "n/a"
        if res["task"] == "T-V":
            s32 = st.get((c, 32))
            if s32 is None:
                fit = "not measured"
            else:
                tag, m = s32
                vram, fps = m.get("peak_vram_gb"), m.get("frames_per_s")
                ok = (not m.get("oom")) and vram is not None and vram <= 10 and 57600 / fps <= 600
                fit = f"{'fits' if ok else 'does not fit'} ({vram} GiB at batch 32, {57600 / fps / 60:.1f} min per budget; step-time run '{tag}')" if not m.get("oom") else "OOM at batch 32"
        Bt.append(f"| {c} | {p['val_auc_mean']:.4f} | {d['estimate']:.4f} [{d['low']:.4f}, {d['high']:.4f}] | {p['gap']:+.4f} | {rank} | {'yes' if c in res['tie_set'] else 'no'} | {'yes' if rank <= 3 else 'no'} | {fit} |")
    dropped = "\n".join(f"- {k}: {v}" for k, v in DROPPED.items())
    ref = f"- From-scratch reference ({'IMG-20' if res['task'] != 'T-A' else 'AU-REF'}): {res['reference']}."
    return (f"#### {res['task']} ({task.title}), generated {res['date']} from run {res['commit']}\n\nWeights and cost (one end-to-end run each; FLOPs and the model-only cost are not measured yet):\n\n"
            + "\n".join(A) + "\n\nResults (exploratory, dev_x; step-time file tags are in the fit column):\n\n" + "\n".join(Bt)
            + "\n\nConsidered and not in the sweep:\n" + dropped + "\n" + ref + "\n")


def write_comparison(res: dict, path: Path = EXPERIMENTS) -> None:
    t = path.read_text(encoding="utf-8")
    crlf = "\r\n" in t
    t = t.replace("\r\n", "\n")
    head = "## Model comparison tables (generated, do not edit by hand)"
    if head not in t:
        t = t.replace("## Experiment log\n", head + "\n\nOne block per task, written by src/dfd/sweep/report.py after each ranking run.\n\n## Experiment log\n", 1)
    t = _replace_block(t, f"TABLE {res['task']}", comparison_tables(res), "## Experiment log\n")
    path.write_text(t.replace("\n", "\r\n") if crlf else t, encoding="utf-8", newline="")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    a = ap.parse_args(argv)
    res = json.loads((RESULTS / a.task / "rank.json").read_text())
    md = (RESULTS / a.task / "rank.md").read_text()
    write_experiment_entry(res, md)
    write_comparison(res)
    (RESULTS / a.task / "model_comparison.md").write_text(comparison_tables(res), encoding="utf-8")
    print("wrote", EXPERIMENTS, "and", RESULTS / a.task / "model_comparison.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
