import json
from pathlib import Path

import numpy as np
import pytest

from dfd.sweep import heads, rank, report, tasks
from dfd.sweep.tasks import Task


def make_features(root: Path, task: Task, cand: str, strength: float, seconds: float, params: int, seed: int):
    """Synthetic frame features: class mean shift of `strength` on the first coordinate, 12 groups, masks with some missing frames."""
    rng = np.random.default_rng(seed)
    for ds, n, ngroups in ((task.train, 240, 12), (task.val, 120, 6), (task.devx, 240, 12)):
        y = (np.arange(n) % 2).astype(int)
        groups = np.array([f"g{(i // 2) % ngroups}" for i in range(n)])
        # the same clips and groups for every candidate: labels and groups are deterministic, only the features differ
        t, d = 4, 6
        f = rng.normal(size=(n, t, d)).astype(np.float16)
        f[:, :, 0] += (strength * (2 * y - 1))[:, None] if np.isscalar(strength) else 0
        mask = rng.random((n, t)) > 0.15
        mask[:, 0] = True
        p = root / cand / task.id / f"{ds}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, feats=f, mask=mask, label=y, group_id=groups, sample_id=np.array([f"{ds}:{i}" for i in range(n)]))
        p.with_suffix(".json").write_text(json.dumps(dict(seconds=seconds, n_frames=int(mask.sum()), params=params, dim=d, input_size=224, peak_vram_gb=1.0)))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    task = Task("T-V", "E-001", "synthetic", ("A", "B", "C", "D"), "tr", "va", "dx", True, ("A", "B"), ("C",), 0.55, 0.60)
    feats = tmp_path / "features"
    monkeypatch.setattr(tasks, "FEATURES", feats)
    monkeypatch.setattr(rank, "RESULTS", tmp_path / "results")
    # A and B carry the same strong signal; B is cheaper. C is weak, D is noise.
    make_features(feats, task, "A", 1.5, 100.0, 90_000_000, 1)
    make_features(feats, task, "B", 1.5, 40.0, 20_000_000, 2)
    make_features(feats, task, "C", 0.4, 10.0, 5_000_000, 3)
    make_features(feats, task, "D", 0.0, 5.0, 1_000_000, 4)
    return task, tmp_path


def test_head_learns_a_separable_signal_and_is_seeded(setup):
    task, _ = setup
    tr, _ = rank.load_split("A", task, task.train)
    va, _ = rank.load_split("A", task, task.val)
    h1, h2 = heads.train_head(tr, va, 0), heads.train_head(tr, va, 0)
    assert h1.val_auc > 0.9
    assert np.allclose(h1.w, h2.w) and h1.best_epoch == h2.best_epoch
    assert not np.allclose(h1.w, heads.train_head(tr, va, 1).w)


def test_ranking_applies_the_tie_rule_and_cost(setup):
    task, tmp = setup
    res = rank.run_task(task, n_boot=300, out_dir=tmp / "out")
    assert res["leader"] in ("A", "B")
    assert set(["A", "B"]) <= set(res["tie_set"])            # the two equal candidates are tied with the leader
    assert "C" not in res["tie_set"] and "D" not in res["tie_set"]
    assert res["winner"] == "B" and res["cost_decided"]       # cheapest of the tie set wins, not the highest AUC
    assert res["i1_order"][:2] == ["B", "A"] and res["i1_order"][2:] == ["C", "D"]
    assert res["candidates"]["D"]["dev_auc"]["estimate"] < 0.65
    assert res["data"]["n_dev_groups"] == 12 and res["data"]["n_dev_excluded_no_score"] == 0
    for f in ("rank.json", "rank.md", "scores.npz"):
        assert (tmp / "out" / "T-V" / f).is_file()


def test_criteria_are_evaluated_mechanically(setup):
    task, tmp = setup
    cr = rank.run_task(task, n_boot=300, out_dir=tmp / "out")["criteria"]
    assert cr["a_family_vs_baseline"]["a"] in ("A", "B") and cr["a_family_vs_baseline"]["b"] == "C"
    assert cr["a_family_vs_baseline"]["met"] is True and cr["b_lower_above"]["met"] is True
    assert cr["verdict"] == "SUPPORTED" and cr["inconclusive"] is False


def test_partial_run_needs_a_separate_output_directory(setup, tmp_path):
    with pytest.raises(SystemExit):
        rank.main(["--task", "T-V", "--candidates", "A,B"])


def test_sign_flip_permutation():
    rng = np.random.default_rng(0)
    y = np.tile([0, 1], 150)
    g = np.array([f"g{i // 2 % 10}" for i in range(300)])
    good = (y + rng.normal(0, 0.6, 300))[None, :]
    noise = rng.normal(0, 1, 300)[None, :]
    far = rank.sign_flip_p(y, g, good, noise)
    assert far["exact"] and far["p"] < 0.05
    same = rank.sign_flip_p(y, g, good, good)
    assert same["p"] == 1.0
    few = rank.sign_flip_p(y, np.array([f"g{i % 2}" for i in range(300)]), good, noise)
    assert few["p"] is None and "fewer than 6" in few["note"]


def test_report_fills_placeholders_once_and_replaces_its_block(setup, tmp_path):
    task, tmp = setup
    res = rank.run_task(task, n_boot=200, out_dir=tmp / "out")
    md = (tmp / "out" / "T-V" / "rank.md").read_text()
    doc = tmp_path / "EXPERIMENTS.md"
    doc.write_text("# E\n\n## E-001: synthetic   (Iteration I-0)\n- Status: PLANNED. The hypothesis was written before.\n"
                   "- Date / Environment / Git commit / Seed(s): to be filled at the run / ENV-DESKTOP-GPU / to be filled / head seeds 0, 1, 2.\n"
                   "- Hypothesis (written BEFORE the run): H stays as written.\n- Command and config: to be filled at the run.\n"
                   "- Result (numbers, with seeds and intervals): none yet.\n- Was the hypothesis supported? Why or why not: none yet.\n\n## Experiment log\n", encoding="utf-8")
    report.write_experiment_entry(res, md, doc)
    report.write_comparison(res, doc)
    t1 = doc.read_text(encoding="utf-8")
    assert "RUN on" in t1 and "PLANNED" not in t1 and "none yet" not in t1 and "to be filled" not in t1
    assert "H stays as written." in t1
    assert "mechanical verdict" in t1 and t1.count("<!-- AUTO E-001 begin -->") == 1 and "<!-- AUTO TABLE T-V begin -->" in t1
    assert "DINOv3" in t1 and "IMG-05" in t1                  # dropped candidates stay in the table with the reason
    report.write_experiment_entry(res, md, doc)
    report.write_comparison(res, doc)
    t2 = doc.read_text(encoding="utf-8")
    assert t2.count("<!-- AUTO E-001 begin -->") == 1 and t2.count("<!-- AUTO TABLE T-V begin -->") == 1
    assert t2.count("## Experiment log") == 1


def test_task_readiness_follows_the_feature_files(setup):
    task, tmp = setup
    assert not tasks.is_ready(tasks.TASKS["T-V"])          # the real tasks expect real candidates and files
    assert all(p.is_file() for c in task.candidates for p in [tasks.feature_path(c, task, task.train)])
    t3 = Task("T-X", "E-009", "x", ("A",), "tr", "va", "dx", True, ("A",), ("A",))
    assert not tasks.is_ready(t3)


def test_reference_is_reported_but_never_selected(setup):
    task, tmp = setup
    out = tmp / "out" / "T-V"
    out.mkdir(parents=True)
    # a reference with a better dev AUC than every candidate must not change the selection
    ds = np.load(tmp / "features" / "A" / "T-V" / "dx.npz")
    y = ds["label"]
    perfect = np.tile((y * 0.8 + 0.1).astype(float), (3, 1))
    np.savez(out / "reference_scores.npz", sample_id=ds["sample_id"], scores=perfect)
    res = rank.run_task(task, n_boot=200, out_dir=tmp / "out")
    assert res["reference"].startswith("reported") and res["reference_result"]["dev_auc"]["estimate"] > 0.99
    assert res["winner"] == "B" and "reference" not in res["i1_order"]
    assert "From-scratch reference" in (out / "rank.md").read_text()
    assert res["reference_result"]["diff_vs_leader"]["estimate"] > 0
