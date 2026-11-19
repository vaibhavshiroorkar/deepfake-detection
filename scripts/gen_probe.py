"""A generated-video probe on CLIP face embeddings: how well it finds generators it never saw.

    .venv/Scripts/python.exe scripts/gen_probe.py

Trains a logistic regression on HiFi-AVDF pairs (each generated clip has a real twin of the same person
and scene) and tests it three ways:

1. Leave one generator out: train on five HiFi-AVDF generators, test on the sixth.
2. Real clips of the other calibration corpora (FakeAVCeleb, DFDC, LAV-DF), never trained on: the line
   that flags 5 in 100 of them, and how many fakes of the held-out generator reach it.
3. External sets, never trained on: our own Veo 3 clips, DF26 (evaluation only) and uploads, per video
   (the most unusual part), at the same line.

Writes results/gen-probe/report.json.

With --export it also writes the website's model, models/website/gen-probe.json (trained on every
HiFi-AVDF generator), and results/gen-probe/cache-scores.csv: the check's logit for every calibration
clip. A HiFi-AVDF clip is scored by the model trained without its generator, so the site's lines and
catch rates for it describe a generator the model did not see.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "gen-probe"
OUT = ROOT / "results" / "gen-probe" / "report.json"
C = 0.005  # on standardized embeddings; chosen by the mean leave-one-generator-out result, before any external set was read


def load(corpus: str):
    data = np.load(RUNS / f"embeddings-{corpus}.npz")
    return data["clip_id"], data["label"].astype(int), data["embedding"]


def generator(clip_id: str) -> str:
    return clip_id.split("__")[1]


def fit(x, y):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=3000, class_weight="balanced"),
    )
    return model.fit(x, y)


def line(scores, q=0.95) -> float:
    return float(np.quantile(scores, q))


def external(model, cut: float) -> dict:
    out = {}
    for path in sorted(RUNS.glob("videos-*.jsonl")):
        per_video = defaultdict(lambda: {"label": None, "group": None, "scores": []})
        for row in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(row)
            v = per_video[r["path"]]
            v["label"], v["group"] = r["label"], r["group"]
            if r["embedding"] is not None:
                v["scores"].append(
                    float(model.decision_function(np.array([r["embedding"]]))[0])
                )
        videos = [
            v | {"score": max(v["scores"])} for v in per_video.values() if v["scores"]
        ]
        groups = defaultdict(list)
        for v in videos:
            groups[v["group"]].append(v)
        reals = [v["score"] for v in videos if v["label"] == 0]
        block = {}
        for g, vs in sorted(groups.items()):
            scores = [v["score"] for v in vs]
            entry = {
                "videos": len(vs),
                "reach_line": float(np.mean([s >= cut for s in scores])),
            }
            if vs[0]["label"] == 1 and reals:
                entry["auc_vs_set_reals"] = float(
                    roc_auc_score([0] * len(reals) + [1] * len(scores), reals + scores)
                )
            if g == "vaibhav":
                entry["scores"] = {
                    Path(v_path).name: round(max(v["scores"]), 3)
                    for v_path, v in per_video.items()
                    if v["group"] == g and v["scores"]
                }
            block[g] = entry
        out[path.stem.removeprefix("videos-")] = block
    return out


def head(model) -> dict:
    scaler, logistic = (
        model.named_steps["standardscaler"],
        model.named_steps["logisticregression"],
    )
    return {
        "note": "Generated-video check: logistic head on mean CLIP ViT-L/14 face-crop embeddings (scripts/gen_probe.py).",
        "C": C,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": logistic.coef_[0].tolist(),
        "intercept": float(logistic.intercept_[0]),
    }


def export(x, y, ids, gens) -> None:
    import csv

    final = fit(x, y)
    target = ROOT / "models" / "website" / "gen-probe.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(head(final)) + "\n", encoding="utf-8")
    held_out = {g: fit(x[gens != g], y[gens != g]) for g in sorted(set(gens))}
    out = ROOT / "results" / "gen-probe" / "cache-scores.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["corpus", "clip_id", "label", "face_gen"])
        for corpus in ("FakeAVCeleb", "DFDC", "LAV-DF"):
            cids, labels, emb = load(corpus)
            for c, lab, s in zip(
                cids, labels, final.decision_function(emb), strict=True
            ):
                writer.writerow([corpus, c, int(lab), f"{s:.6f}"])
        for c, lab, g, e in zip(ids, y, gens, x, strict=True):
            writer.writerow(
                [
                    "HiFi-AVDF",
                    c,
                    int(lab),
                    f"{held_out[g].decision_function(e[None])[0]:.6f}",
                ]
            )
    print(f"wrote {target.relative_to(ROOT)} and {out.relative_to(ROOT)}")


def main() -> int:
    ids, y, x = load("HiFi-AVDF")
    gens = np.array([generator(c) for c in ids])
    others = {c: load(c) for c in ("FakeAVCeleb", "DFDC", "LAV-DF")}
    other_reals = np.concatenate([e[y_ == 0] for _, y_, e in others.values()])
    report = {"C": C, "leave_one_generator_out": {}}
    for held in sorted(set(gens)):
        train = gens != held
        model = fit(x[train], y[train])
        test = ~train
        s = model.decision_function(x[test])
        cut = line(model.decision_function(other_reals))
        fakes = s[y[test] == 1]
        report["leave_one_generator_out"][held] = {
            "auc_vs_twins": float(roc_auc_score(y[test], s)),
            "caught_at_5pct_other_reals": float(np.mean(fakes >= cut)),
            "twins_flagged": float(np.mean(s[y[test] == 0] >= cut)),
        }
    model = fit(x, y)
    cut = line(model.decision_function(other_reals))
    report["all_generators"] = {
        "line_5pct_other_reals": cut,
        "other_reals_flagged_per_corpus": {
            c: float(np.mean(model.decision_function(e[y_ == 0]) >= cut))
            for c, (_, y_, e) in others.items()
        },
        "other_fakes_reaching_line": {
            c: float(np.mean(model.decision_function(e[y_ == 1]) >= cut))
            for c, (_, y_, e) in others.items()
        },
        "external": external(model, cut),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    if "--export" in sys.argv:
        export(x, y, ids, gens)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
