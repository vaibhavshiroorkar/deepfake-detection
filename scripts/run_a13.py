"""P-001 amendment A13 (exploratory): what does the visual path read on LAV-DF?

Six statistics of the cached face frames the visual path sees, compared between LAV-DF real clips
and LAV-DF fakes whose video was never edited (RealVideo-FakeAudio). FakeAVCeleb (real against
FakeVideo clips) is measured the same way, to show the direction the path learned. Writes
`{output}/a13.json` and `{output}/statistics.csv`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import laplace
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

STATISTICS = ("sharpness", "high_frequency", "blockiness", "brightness", "contrast", "temporal")
MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
LUMA = np.array([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1)
REAL = "RealVideo-RealAudio"
SEEDS = (17, 29, 43)


def luminance(view: np.ndarray) -> np.ndarray:
    """(frames, 3, H, W) ImageNet-normalized RGB -> (frames, H, W) luminance in [0, 1]."""
    return ((np.asarray(view, dtype=np.float64) * STD + MEAN) * LUMA).sum(axis=1)


def _high_frequency(frame: np.ndarray) -> float:
    power = np.abs(np.fft.fft2(frame - frame.mean())) ** 2
    fy, fx = np.meshgrid(np.fft.fftfreq(frame.shape[0]), np.fft.fftfreq(frame.shape[1]), indexing="ij")
    total = power.sum()
    return float(power[np.hypot(fx, fy) > 0.25].sum() / total) if total > 0 else 0.0


def _blockiness(frame: np.ndarray) -> float:
    ratios = []
    for diff in (np.abs(np.diff(frame, axis=1)), np.abs(np.diff(frame, axis=0)).T):
        boundary = (np.arange(diff.shape[1]) + 1) % 8 == 0
        # The epsilon keeps a flat block interior (no jump inside) reading as very blocky, not neutral.
        ratios.append((diff[:, boundary].mean() + 1e-6) / (diff[:, ~boundary].mean() + 1e-6))
    return float(np.mean(ratios))


def clip_statistics(view: np.ndarray) -> dict[str, float]:
    y = luminance(view)
    return {
        "sharpness": float(np.mean([laplace(f).var() for f in y])),
        "high_frequency": float(np.mean([_high_frequency(f) for f in y])),
        "blockiness": float(np.mean([_blockiness(f) for f in y])),
        "brightness": float(y.mean()),
        "contrast": float(np.mean([f.std() for f in y])),
        "temporal": float(np.abs(np.diff(y, axis=0)).mean()) if len(y) > 1 else 0.0,
    }


def _bootstrap_auc(labels, values, groups, samples=1000, seed=17):
    ids = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in ids}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        index = np.concatenate([rows[g] for g in rng.choice(ids, size=len(ids))])
        if labels[index].min() != labels[index].max():
            draws.append(roc_auc_score(labels[index], values[index]))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def contrast(table: list[dict], positive, score_key: str | None) -> dict:
    """Each statistic's AUC for `positive` clips against the reals, and the A13 reading."""
    rows = [r for r in table if r["manipulation_type"] == REAL or positive(r["manipulation_type"])]
    labels = np.array([int(r["manipulation_type"] != REAL) for r in rows])
    groups = np.array([r["source"] for r in rows])
    x = np.array([[r[s] for s in STATISTICS] for r in rows])
    out = {"clips": len(rows), "positives": int(labels.sum()), "statistics": {}}
    score = np.array([r[score_key] for r in rows]) if score_key else None
    for i, name in enumerate(STATISTICS):
        auc = float(roc_auc_score(labels, x[:, i]))
        entry = {"auc": auc, "interval": _bootstrap_auc(labels, x[:, i], groups)}
        if score is not None:
            entry["spearman_with_visual"] = float(spearmanr(x[:, i], score).statistic)
            entry["candidate"] = bool((auc < 0.4 or auc > 0.6) and abs(entry["spearman_with_visual"]) >= 0.3)
        out["statistics"][name] = entry
    folds = min(5, len(np.unique(groups)))
    oof = np.empty(len(rows))
    for train, held in GroupKFold(n_splits=folds).split(x, labels, groups):
        mu, sd = x[train].mean(axis=0), x[train].std(axis=0)
        sd[sd == 0] = 1.0
        model = LogisticRegression(max_iter=1000, class_weight="balanced").fit((x[train] - mu) / sd, labels[train])
        oof[held] = model.predict_proba((x[held] - mu) / sd)[:, 1]
    out["all_six_cross_fit_auc"] = float(roc_auc_score(labels, oof))
    if score is not None:
        out["visual_path_auc"] = float(roc_auc_score(labels, score))
    return out


def measure(predictions: Path, index_csv: Path, cache_root: Path, keep) -> list[dict]:
    """Statistics for each scored clip whose manipulation type passes `keep`, with the visual score."""
    from deepfake_detection.data.cache_index import read_cache_index
    from deepfake_detection.views.cache_store import CacheStore

    index, store = read_cache_index(index_csv), CacheStore(cache_root)
    scores: dict[str, list[float]] = {}
    meta: dict[str, dict] = {}
    for seed in SEEDS:
        with Path(str(predictions).format(seed=seed)).open(encoding="utf-8", newline="") as handle:
            for r in csv.DictReader(handle):
                if keep(r["manipulation_type"]):
                    scores.setdefault(r["clip_id"], []).append(float(r["probability"]))
                    meta[r["clip_id"]] = r
    table = []
    for clip_id in sorted(meta):
        view = store.load(index[clip_id], views=("visual_view",)).visual_view
        table.append({"clip_id": clip_id, "manipulation_type": meta[clip_id]["manipulation_type"],
                      "source": meta[clip_id]["source"], "visual": float(np.mean(scores[clip_id])),
                      **clip_statistics(view)})
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, default=Path("runs/p001"))
    parser.add_argument("--branches", type=Path, default=Path("results/p001-branches"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lavdf = measure(args.branches / "visual-seed{seed}-lavdf-test-predictions.csv",
                    args.run / "cache-lavdf-test-index.csv", args.run / "cache-targets",
                    lambda t: t in (REAL, "RealVideo-FakeAudio"))
    favc = measure(args.branches / "visual-seed{seed}-test-identity-strict-predictions.csv",
                   args.run / "cache-index.csv", args.run / "cache",
                   lambda t: t == REAL or t.startswith("FakeVideo"))
    report = {
        "amendment": "A13", "exploratory": True,
        "lavdf_unedited_video_fakes_vs_reals": contrast(lavdf, lambda t: t == "RealVideo-FakeAudio", "visual"),
        "fakeavceleb_video_fakes_vs_reals": contrast(favc, lambda t: t.startswith("FakeVideo"), "visual"),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "a13.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    with (args.output / "statistics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", *lavdf[0].keys()])
        writer.writeheader()
        writer.writerows([{"dataset": "LAV-DF", **r} for r in lavdf] + [{"dataset": "FakeAVCeleb", **r} for r in favc])
    for name, block in report.items():
        if isinstance(block, dict):
            print(name, "all six:", round(block["all_six_cross_fit_auc"], 3), "visual:", round(block.get("visual_path_auc", float("nan")), 3))
            for stat, e in block["statistics"].items():
                print(f"   {stat:15s} AUC {e['auc']:.3f} [{e['interval'][0]:.3f}, {e['interval'][1]:.3f}]"
                      f" rho {e.get('spearman_with_visual', float('nan')):+.2f} candidate {e.get('candidate')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
