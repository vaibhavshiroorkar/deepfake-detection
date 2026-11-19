"""Compare the website's current face and voice models with the v2 candidates on the held-out sets.

    .venv/Scripts/python.exe scripts/report_v2_eval.py

Reads results/v2-eval/scores*.jsonl (written by score_v2_eval.py) and, when present, cache-scores.csv
(score_v2_cache.py, the four calibration corpora as cleaned video). For each set and model: AUC with a
95% interval from a bootstrap over groups (speakers, source videos or methods), and the share of fakes
caught when 5 in 100 of that set's real items are flagged. Models are compared only on the items every
model of that check scored, so a model is not helped by skipping hard items. Refused items are counted.
Writes results/v2-eval/report.json and prints a table.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "results" / "v2-eval"
RESAMPLES = 1000


def catch_at(
    labels: np.ndarray, scores: np.ndarray, false_alarm: float = 0.05
) -> float:
    line = np.quantile(scores[labels == 0], 1 - false_alarm)
    return float(np.mean(scores[labels == 1] > line))


def interval(labels, scores, groups, seed: int = 19) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    values = []
    for _ in range(RESAMPLES):
        pick = np.concatenate([members[g] for g in rng.choice(unique, len(unique))])
        if len(np.unique(labels[pick])) == 2:
            values.append(roc_auc_score(labels[pick], scores[pick]))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _group(set_name: str, item: dict) -> str:
    """The unit the interval resamples: the person or source video, so near-copies stay together.

    FF++ fakes are named target_source and Celeb-DF fakes idA_idB_n, so the first id is the person
    whose video it is; a real video's stem starts with the same id.
    """
    if set_name in ("ffpp", "celebdf"):
        return Path(item["path"].replace("\\", "/")).stem.split("_")[0]
    return str(item.get("group", ""))


def summarize(name: str, items: list[dict]) -> dict:
    models = sorted({m for i in items for m in i.get("scores", {})})
    scored = [
        i
        for i in items
        if "scores" in i and all(i["scores"].get(m) is not None for m in models)
    ]
    out = {
        "items": len(items),
        "refused": sum("refused" in i for i in items),
        "compared": len(scored),
        "models": {},
    }
    if not scored:
        return out
    labels = np.array([int(i["label"]) for i in scored])
    groups = np.array([_group(name, i) for i in scored])
    out["real"], out["fake"] = int((labels == 0).sum()), int((labels == 1).sum())
    if len(np.unique(labels)) < 2:
        return out
    for m in models:
        scores = np.array([i["scores"][m] for i in scored], dtype=float)
        low, high = (
            interval(labels, scores, groups)
            if len(np.unique(groups)) > 1
            else (None, None)
        )
        out["models"][m] = {
            "auc": round(float(roc_auc_score(labels, scores)), 3),
            "auc_low": None if low is None else round(low, 3),
            "auc_high": None if high is None else round(high, 3),
            "caught_at_5pct": round(catch_at(labels, scores), 3),
        }
    return out


def load_items() -> dict[str, list[dict]]:
    sets: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(FOLDER.glob("scores*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            sets[item["set"]].append(item)
    cache = FOLDER / "cache-scores.csv"
    if cache.exists():
        with open(cache, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                for check in ("face", "voice"):
                    scores = {
                        k: float(v)
                        for k, v in r.items()
                        if k.startswith(f"{check}_") and v
                    }
                    if scores:
                        sets[f"cache-{r['corpus']}-{check}"].append(
                            {
                                "label": r["label"],
                                "group": r["clip_id"],
                                "scores": scores,
                            }
                        )
    return sets


def main() -> int:
    report = {
        name: summarize(name, items) for name, items in sorted(load_items().items())
    }
    (FOLDER / "report.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="utf-8"
    )
    print(
        f"{'set':28} {'model':20} {'AUC':>6} {'95% interval':>15} {'caught@5%':>10}  items"
    )
    for name, s in report.items():
        for m, r in s["models"].items():
            span = (
                f"[{r['auc_low']}, {r['auc_high']}]" if r["auc_low"] is not None else ""
            )
            print(
                f"{name:28} {m:20} {r['auc']:6.3f} {span:>15} {r['caught_at_5pct']:10.3f}  "
                f"{s['compared']} of {s['items']} ({s['refused']} refused)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
