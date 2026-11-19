"""How to combine the website's checks into one score: three rules, each tested on a dataset it never saw.

    .venv/Scripts/python.exe scripts/fusion_test.py

Rules, all on the checks' percentiles among real calibration clips (face, voice, lip sync, generated):
- max: the most unusual check decides (the website's rule until 11-20);
- sum: an untrained evidence sum, -2 sum ln(1 - p) divided by its degrees, so that several mildly unusual
  checks add up (Fisher's method); nothing is learned from fakes;
- trained: logistic regression on the four percentiles (as logits) with a missing-check flag each.

Leave one corpus out: the line (and the trained rule's weights) come from the other three corpora, whose
real clips set the line at 5 in 100; the held-out corpus gives catches and false alarms. Writes
results/fusion/report.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CHECKS = ("face", "voice", "sync", "gen")
OUT = ROOT / "results" / "fusion" / "report.json"


def evidence_sum(percentiles: list[float]) -> float:
    """Fisher's statistic over the checks that ran, divided by its degrees of freedom."""
    stat = -2 * sum(math.log(max(1 - p, 1e-6)) for p in percentiles)
    return stat / (2 * len(percentiles))


def features(pcts: dict) -> list[float]:
    out = []
    for c in CHECKS:
        p = pcts.get(c)
        q = min(max(p, 1e-4), 1 - 1e-4) if p is not None else 0.5
        out += [math.log(q / (1 - q)), 0.0 if p is not None else 1.0]
    return out


def scorer(rule: str, train: list[dict], features, corpus_weights):
    """The function that gives one clip its combined score under `rule`; the trained rule fits on `train`."""
    if rule == "max":
        return lambda r: max(r["seen"])
    if rule == "sum":
        return lambda r: evidence_sum(r["seen"])
    from sklearn.linear_model import LogisticRegression

    x = np.array([features(r["pcts"]) for r in train])
    y = np.array([r["label"] for r in train])
    w = np.array(corpus_weights([r["corpus"] for r in train]))
    model = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(
        x, y, sample_weight=w
    )

    def trained(r: dict) -> float:
        return float(model.decision_function([features(r["pcts"])])[0])

    return trained


def main() -> int:
    from build_web_calibration import CORPORA, replace_scores, rows_for

    from deepfake_detection.webapp.calibration import (
        Calibration,
        corpus_weights,
        weighted_quantile,
    )

    choice = json.loads(
        (ROOT / "results" / "web-models.json").read_text(encoding="utf-8")
    )
    sources: dict = {}
    rows = [r for c, (b, s) in CORPORA.items() for r in rows_for(c, b, s, sources)]
    replace_scores(rows, choice, sources)
    cal = Calibration.from_json(
        (ROOT / "results" / "web-calibration-v2.json").read_text(encoding="utf-8")
    )
    for r in rows:
        r["pcts"] = cal.percentiles(r["scores"])
        r["seen"] = [p for p in r["pcts"].values() if p is not None]
    rows = [r for r in rows if r["seen"]]
    report: dict = {"rules": {}}
    for rule in ("max", "sum", "trained"):
        per = {}
        caught_all = fakes_all = 0
        for held in CORPORA:
            train = [r for r in rows if r["corpus"] != held]
            test = [r for r in rows if r["corpus"] == held]
            score = scorer(rule, train, features, corpus_weights)
            reals = [r for r in train if r["label"] == 0]
            line = weighted_quantile(
                [score(r) for r in reals],
                corpus_weights([r["corpus"] for r in reals]),
                0.95,
            )
            fakes = [score(r) >= line for r in test if r["label"] == 1]
            real_flags = [score(r) >= line for r in test if r["label"] == 0]
            per[held] = {
                "caught": float(np.mean(fakes)),
                "reals_flagged": float(np.mean(real_flags)),
            }
            caught_all += sum(fakes)
            fakes_all += len(fakes)
        report["rules"][rule] = {
            "held_out": per,
            "all_fakes_caught": caught_all / fakes_all,
            "mean_reals_flagged": float(
                np.mean([v["reals_flagged"] for v in per.values()])
            ),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for rule, block in report["rules"].items():
        cells = " | ".join(
            f"{c} {v['caught']:.1%} ({v['reals_flagged']:.1%})"
            for c, v in block["held_out"].items()
        )
        print(
            f"{rule:8s} all {block['all_fakes_caught']:.1%}, reals {block['mean_reals_flagged']:.1%} | {cells}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
