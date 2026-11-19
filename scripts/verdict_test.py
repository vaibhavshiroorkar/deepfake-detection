"""Can the site say "Real" or "Fake" where it is reliable, and "can't tell" only elsewhere? Two measurements.

    .venv/Scripts/python.exe scripts/verdict_test.py

1. Familiarity: a logistic regression on the CLIP face embeddings (runs/gen-probe) tells clips of the corpora
   where the checks are reliable (FakeAVCeleb, HiFi-AVDF) from the others (DFDC, LAV-DF), scored by 5-fold
   cross-validation.
2. A "Real" line: within the reliable corpora, the overall score (the evidence sum) below which at most 5 in
   100 fakes fall, and the share of real clips below it; set on half the clips, measured on the other half.

Writes results/verdict/report.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
RELIABLE = ("FakeAVCeleb", "HiFi-AVDF")
OUT = ROOT / "results" / "verdict" / "report.json"


def familiarity() -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    xs, ys, groups = [], [], []
    for corpus in ("FakeAVCeleb", "DFDC", "LAV-DF", "HiFi-AVDF"):
        data = np.load(ROOT / "runs" / "gen-probe" / f"embeddings-{corpus}.npz")
        xs.append(data["embedding"])
        ys += [corpus in RELIABLE] * len(data["embedding"])
        groups += [corpus] * len(data["embedding"])
    x, y = np.concatenate(xs), np.array(ys, int)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced"),
    )
    p = cross_val_predict(
        model,
        x,
        y,
        cv=StratifiedKFold(5, shuffle=True, random_state=41),
        method="predict_proba",
    )[:, 1]
    groups = np.array(groups)
    return {
        "accuracy": float(np.mean((p >= 0.5) == y)),
        "called_reliable_per_corpus": {
            c: float(np.mean(p[groups == c] >= 0.5)) for c in sorted(set(groups))
        },
    }


def real_line() -> dict:
    from build_web_calibration import CORPORA, replace_scores, rows_for

    from deepfake_detection.webapp.calibration import Calibration, evidence_sum

    choice = json.loads(
        (ROOT / "results" / "web-models.json").read_text(encoding="utf-8")
    )
    sources: dict = {}
    rows = [
        r
        for c, (b, s) in CORPORA.items()
        for r in rows_for(c, b, s, sources)
        if c in RELIABLE
    ]
    replace_scores(rows, choice, sources)
    cal = Calibration.from_json(
        (ROOT / "results" / "web-calibration-v2.json").read_text(encoding="utf-8")
    )
    scored = []
    for r in rows:
        seen = [p for p in cal.percentiles(r["scores"]).values() if p is not None]
        if seen:
            scored.append((r["corpus"], r["label"], evidence_sum(seen)))
    rng = np.random.default_rng(41)
    order = rng.permutation(len(scored))
    half = len(order) // 2
    fit = [scored[i] for i in order[:half]]
    test = [scored[i] for i in order[half:]]
    fakes_fit = np.array([s for _, lab, s in fit if lab == 1])
    line = float(np.quantile(fakes_fit, 0.05))  # at most 5 in 100 fakes below
    out = {"line": line, "held_out": {}}
    for corpus in RELIABLE:
        fakes = np.array([s for c, lab, s in test if c == corpus and lab == 1])
        reals = np.array([s for c, lab, s in test if c == corpus and lab == 0])
        out["held_out"][corpus] = {
            "fakes_called_real": float(np.mean(fakes < line)),
            "reals_called_real": float(np.mean(reals < line)),
        }
    return out


def main() -> int:
    report = {"familiarity": familiarity(), "real_line": real_line()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
