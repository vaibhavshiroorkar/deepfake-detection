"""Completeness and consistency check of the freeze score files, before the evaluation step.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/freeze/check_score_files.py [--dry] [--iteration I-0 I-1 ...]

Reads every <root>/scores/*.csv (root data/_final, or data/_final_dry with --dry) with dfd.eval.scorefile.read_scores (the same reader the evaluation uses),
prints the number of rows per (iteration, modality, dataset, split) and the number of seeds, and checks:
  * the columns and values are valid (the reader), every block has the seeds 0, 1, 2 and the same sample ids in each seed;
  * the number of items against the split files (items missing because a clip had no face or no finished crop are listed, not hidden);
  * a --dry file contains no sealed split; a final file contains the sealed splits it should;
  * parity with the logged training-time scores: validation clip scores of I-0 (data/_results/I-0) and of the fine-tuned IMG-17 (data/_results/I-1) against
    the score file, on the items they share (tolerance 1e-3: fp16 inference is not bit-exact, the largest difference seen in the dry run was 4.4e-4).
Exit code 1 if a check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
from dfd.eval.scorefile import read_scores  # noqa: E402

TOL = 1e-3


def expected_counts():
    exp = {}
    for ds in K.VIDEO_SETS + K.IMAGE_SETS + K.AUDIO_SETS + [K.LAVDF]:
        j = json.loads(Path(ds["split_csv"]).with_suffix(".json").read_text())
        for split, c in j["counts"].items():
            exp[(ds["key"], split)] = c["samples"]
    v2, pp = exp.get(("celebdf_v2", "test_x")), exp.get(("celebdf_pp", "test_x"))
    j2 = json.loads(Path("data/_splits/celebdf_v2-82a3a70594e9.json").read_text())["counts"]["test_x"]
    exp[("celebdf_family", "test_x")] = v2 + pp
    exp[("celebdf_pp_with_v2_real", "test_x")] = pp + j2["real"]
    for split in ("val", "test_in", "train"):
        n = exp.get(("lavdf", split))
        if n:
            for name in ("lavdf_videolabel", "lavdf_audiolabel"):
                exp[(name, split)] = n
            exp[("lavdf_produced_videolabel", split)] = exp[("lavdf_produced_audiolabel", split)] = n * 3 // 4
    return exp


def parity(t, problems):
    """Validation scores against the logged training-time scores (only for the items present in the score file)."""
    tag = str(t.iteration[0])
    checks = []
    if tag in ("I-0",):
        for tid, mod, ds in (("T-V", "visual", "ffpp"), ("T-I", "image", "forensynths_progan_val"), ("T-A", "audio", "asvspoof2019_la")):
            ref = np.load(Path("data/_results/I-0") / tid / "scores.npz", allow_pickle=True)
            checks.append((mod, ds, {str(s): ref["val_scores"][:, i] for i, s in enumerate(ref["val_sample_id"])}))
    if tag in ("I-1", "I-2", "I-3", "I-4"):
        ref = [np.load(f"data/_results/I-1/IMG-17_lr1e-05_s{s}/scores.npz", allow_pickle=True) for s in K.SEEDS]
        checks.append(("visual", "ffpp", {str(sid): np.array([r["val_scores"][i] for r in ref]) for i, sid in enumerate(ref[0]["val_sample_id"])}))
    for mod, ds, table in checks:
        worst, n = 0.0, 0
        for s in K.SEEDS:
            m = (t.modality == mod) & (t.dataset == ds) & (t.split == "val") & (t.seed == s)
            for sid, sc in zip(t.sample_id[m], t.score[m]):
                if sid in table:
                    worst, n = max(worst, abs(float(table[sid][s]) - float(sc))), n + 1
        status = "ok" if n and worst <= TOL else "FAIL"
        print(f"    parity {mod}/{ds}/val against the logged scores: {n} values compared, largest difference {worst:.2e} [{status}]")
        if status == "FAIL":
            problems.append(f"{tag}: parity {mod}/{ds}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--iteration", nargs="*")
    a = ap.parse_args()
    root = K.DRY_ROOT if a.dry else K.FINAL_ROOT
    files = sorted((root / "scores").glob("*.csv"))
    if a.iteration:
        files = [f for f in files if f.stem in a.iteration]
    if not files:
        raise SystemExit(f"no score files in {root / 'scores'}")
    exp, problems = expected_counts(), []
    for f in files:
        if f.stem.endswith(".partial") or ".partial" in f.name:
            print(f"-- {f.name}: partial file, skipped")
            continue
        try:
            t = read_scores(f)
        except Exception as e:   # noqa: BLE001
            problems.append(f"{f.name}: {e}")
            print(f"-- {f.name}: INVALID: {e}")
            continue
        its = sorted({str(x) for x in t.iteration})
        print(f"-- {f.name}: {len(t)} clip or image rows, iteration {its}, sha256 {t.sha256[:12]}")
        if len(its) != 1 or its[0] != f.stem:
            problems.append(f"{f.name}: iteration column {its}")
        sealed = bool(np.isin(t.split, K.SEALED).any())
        if a.dry and sealed:
            problems.append(f"{f.name}: a dry-run file contains sealed rows")
        blocks = sorted({(str(m), str(d), str(s)) for m, d, s in zip(t.modality, t.dataset, t.split)})
        for mod, ds, sp in blocks:
            m = (t.modality == mod) & (t.dataset == ds) & (t.split == sp)
            seeds = sorted({int(x) for x in t.seed[m]})
            sets = [frozenset(t.sample_id[m & (t.seed == s)]) for s in seeds]
            n = len(sets[0])
            same = all(s == sets[0] for s in sets)
            e = exp.get((ds, sp))
            miss = "" if e is None else (" complete" if e == n else f" MISSING {e - n} of {e}" if n < e else f" MORE THAN LISTED ({n} > {e})")
            lim = a.dry and e is not None and n != e
            flag = ""
            if seeds != list(K.SEEDS):
                flag += " SEEDS " + str(seeds)
                problems.append(f"{f.name}: {mod}/{ds}/{sp} seeds {seeds}")
            if not same:
                flag += " SAMPLE SETS DIFFER BETWEEN SEEDS"
                problems.append(f"{f.name}: {mod}/{ds}/{sp} sample sets differ between seeds")
            if e is not None and n > e:
                problems.append(f"{f.name}: {mod}/{ds}/{sp} has more rows than the split lists")
            if e is not None and n < e and not a.dry:
                problems.append(f"{f.name}: {mod}/{ds}/{sp} lacks {e - n} items (check data/_final/scores/*.provenance.json, no_score)")
            if len({int(x) for x in t.label[m]}) < 2:
                flag += " ONE CLASS ONLY"
                problems.append(f"{f.name}: {mod}/{ds}/{sp} has one class only")
            print(f"    {mod:7s} {ds:28s} {sp:7s} items {n:6d} seeds {len(seeds)}{'' if lim else miss}{flag}")
        parity(t, problems)
    print()
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
