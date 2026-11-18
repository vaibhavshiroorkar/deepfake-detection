"""run_a12.py: leave-one-corpus-out fusion and the in-target upper bound (P-001 amendment A12)."""

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_a12.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_a12", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _corpus(rng, n, informative):
    labels = np.tile([0, 1], n // 2)
    x = rng.normal(size=(n, 3))
    x[:, informative] += 2.0 * labels
    groups = np.array([f"id{i // 4}" for i in range(n)])
    return module_target(labels, groups, x)


def module_target(labels, groups, x):
    return {"labels": labels, "groups": groups, "x": x, "a8": np.full(len(labels), 0.5)}


def test_leave_one_out_scores_never_see_the_held_out_labels():
    module = _module()
    rng = np.random.default_rng(0)
    data = {
        "a": _corpus(rng, 40, 0),
        "b": _corpus(rng, 40, 1),
        "c": _corpus(rng, 40, 2),
    }
    first = module.scores_for_seed(data)
    shuffled = dict(data)
    shuffled["c"] = dict(data["c"], labels=rng.permutation(data["c"]["labels"]))
    second = module.scores_for_seed(shuffled)
    assert np.array_equal(first["c"]["leave_one_out"], second["c"]["leave_one_out"])
    # The upper bound uses the target's own labels, so it must change.
    assert not np.array_equal(first["c"]["upper_bound"], second["c"]["upper_bound"])


def test_upper_bound_finds_the_path_that_only_this_corpus_rewards():
    module = _module()
    rng = np.random.default_rng(1)
    data = {
        "a": _corpus(rng, 200, 0),
        "b": _corpus(rng, 200, 0),
        "c": _corpus(rng, 200, 2),
    }
    scores = module.scores_for_seed(data)
    labels = data["c"]["labels"]
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(labels, scores["c"]["upper_bound"]) > 0.85
    assert roc_auc_score(labels, scores["c"]["leave_one_out"]) < 0.7
