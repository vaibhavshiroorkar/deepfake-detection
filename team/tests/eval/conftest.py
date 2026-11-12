"""Shared helper: build a synthetic score file for pipeline tests."""

import numpy as np
import pytest

from dfd.eval.scorefile import write_scores


def build_scores(path, *, seed=0, seeds=(1, 2), scorer="random", iteration="I-0",
                 modality="visual", groups_per_split=80, per_group=4, train_ds="synthA",
                 x_ds="synthB", run_prefix="r", layout=None, strength=2.0):
    """Write a score CSV with train, val, test_in (train_ds) and test_x (x_ds).

    Groups never cross splits. scorer 'random' gives uniform scores unrelated to
    the label; 'signal' gives a real separation.
    """
    rng = np.random.default_rng(seed)
    layout = layout or [("train", train_ds), ("val", train_ds), ("test_in", train_ds), ("test_x", x_ds)]
    cols = {c: [] for c in ("run_id", "iteration", "seed", "modality", "dataset", "split",
                            "sample_id", "group_id", "label", "score")}
    for sd in seeds:
        for split, ds in layout:
            for g in range(groups_per_split):
                gid = f"{ds}_{split}_g{g}"
                for k in range(per_group):
                    y = int(rng.integers(0, 2)) if sd == seeds[0] else None
                    cols["run_id"].append(f"{run_prefix}{sd}")
                    cols["iteration"].append(iteration)
                    cols["seed"].append(sd)
                    cols["modality"].append(modality)
                    cols["dataset"].append(ds)
                    cols["split"].append(split)
                    cols["sample_id"].append(f"{gid}_s{k}")
                    cols["group_id"].append(gid)
                    cols["label"].append(y)
                    cols["score"].append(None)
    # labels are shared across seeds: fill from the first seed's rows
    n_first = len(layout) * groups_per_split * per_group
    labels = cols["label"][:n_first]
    for i in range(len(cols["label"])):
        cols["label"][i] = labels[i % n_first]
    y = np.array(cols["label"])
    r = np.random.default_rng([seed, 999])
    if scorer == "random":
        sc = r.random(y.size)
    else:
        sc = 1.0 / (1.0 + np.exp(-(strength * (2 * y - 1) + r.normal(0, 1.5, y.size))))
    cols["score"] = [f"{v:.12g}" for v in np.clip(sc, 0, 1)]
    write_scores(path, cols)
    return path


@pytest.fixture
def make_scores(tmp_path):
    def _make(name="scores.csv", **kw):
        return build_scores(tmp_path / name, **kw)
    return _make
