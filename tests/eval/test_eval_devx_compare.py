"""Split dev_x (D-035) and the --compare option."""

import json

import numpy as np
import pytest

from dfd.eval import ledger as L
from dfd.eval.guard import GuardError, check_guard
from dfd.eval.run import main
from dfd.eval.scorefile import read_scores, write_scores

DEV_LAYOUT = [("val", "synthA"), ("dev_x", "synthB")]


def test_dev_x_is_valid_and_not_final(make_scores, tmp_path):
    t = read_scores(make_scores(seeds=(1,), groups_per_split=10, layout=DEV_LAYOUT))
    assert set(t.split) == {"val", "dev_x"}
    d = check_guard(t, final=False, ledger_path=tmp_path / "l.jsonl")  # no flag needed
    assert d.entries == [] and d.final_blocks == []


def test_dev_x_only_file_runs_without_final_and_repeatedly(make_scores, tmp_path, capsys):
    f = str(make_scores(seeds=(1, 2), groups_per_split=20, layout=DEV_LAYOUT, scorer="signal"))
    lg = tmp_path / "ledger.jsonl"
    out = tmp_path / "r.json"
    for _ in range(3):  # repeated evaluation is allowed
        assert main(["--scores", f, "--ledger", str(lg), "--n-boot", "50", "--out", str(out)]) == 0
    assert not lg.exists()
    shown = capsys.readouterr().out
    assert "dev_x (dev)" in shown and "development split" in shown
    blocks = json.loads(out.read_text(encoding="utf-8"))["candidates"][0]["blocks"]
    dev = [b for b in blocks if b["split"] == "dev_x"][0]
    assert dev["development"] is True and "development" in dev["role"]
    assert any("never the headline" in n for n in dev["notes"])
    val = [b for b in blocks if b["split"] == "val"][0]
    assert val["development"] is False


def test_test_x_rows_still_need_final(make_scores, tmp_path, capsys):
    layout = [("val", "synthA"), ("dev_x", "synthB"), ("test_x", "synthB")]
    f = str(make_scores(seeds=(1,), groups_per_split=10, layout=layout))
    lg = str(tmp_path / "l.jsonl")
    assert main(["--scores", f, "--ledger", lg, "--n-boot", "20"]) == 2
    assert "refused" in capsys.readouterr().err
    assert main(["--scores", f, "--ledger", lg, "--n-boot", "20", "--splits", "val,dev_x"]) == 0
    assert main(["--scores", f, "--ledger", lg, "--n-boot", "20", "--final", "--iteration", "I-0"]) == 0
    entries = L.read_ledger(lg)
    assert {(e["dataset"], e["split"]) for e in entries} == {("synthB", "test_x")}  # dev_x not logged


def test_dev_x_not_pooled_into_test_x(make_scores, tmp_path):
    layout = [("val", "synthA"), ("dev_x", "synthB"), ("test_x", "synthB")]
    f = make_scores(seeds=(1,), groups_per_split=10, layout=layout)
    out = tmp_path / "r.json"
    assert main(["--scores", str(f), "--ledger", str(tmp_path / "l.jsonl"), "--n-boot", "20",
                 "--final", "--iteration", "I-0", "--out", str(out)]) == 0
    blocks = json.loads(out.read_text(encoding="utf-8"))["candidates"][0]["blocks"]
    by = {b["split"]: b for b in blocks}
    assert by["dev_x"]["n_samples"] == by["test_x"]["n_samples"] == 40  # separate, not 80
    assert by["dev_x"]["development"] and not by["test_x"]["development"]
    assert by["test_x"]["role"] != by["dev_x"]["role"] and "final" in by["test_x"]["role"]


# ------------------------------------------------------------------ --compare

def compare_args(f, extra=(), split="dev_x", **kw):
    a = ["--compare", "--scores", str(f), "--modality", "visual", "--dataset", "synthB",
         "--split", split, "--n-boot", "300"]
    return a + list(extra)


def test_compare_identical_runs_difference_zero(make_scores, tmp_path, capsys):
    f = make_scores(seeds=(1, 2), groups_per_split=40, layout=DEV_LAYOUT, scorer="signal")
    out = tmp_path / "c.json"
    for metric in ("auc", "eer", "brier", "ece_cal"):
        code = main(compare_args(f, ["--run-a", "r1,r2", "--run-b", "r1,r2", "--metric", metric,
                                     "--out", str(out)]))
        assert code == 0
        d = json.loads(out.read_text(encoding="utf-8"))["difference"]
        assert d["estimate"] == 0.0 and d["low"] == 0.0 and d["high"] == 0.0
        assert d["contains_zero"] is True
    assert "contains 0" in capsys.readouterr().out


def test_compare_clearly_better_run_has_interval_above_zero(make_scores, tmp_path, capsys):
    strong = make_scores("a.csv", seeds=(1,), groups_per_split=60, layout=DEV_LAYOUT,
                         scorer="signal", strength=2.5, run_prefix="good")
    weak = make_scores("b.csv", seeds=(1,), groups_per_split=60, layout=DEV_LAYOUT,
                       scorer="signal", strength=0.2, run_prefix="bad")
    out = tmp_path / "c.json"
    code = main(compare_args(strong, ["--scores-b", str(weak), "--out", str(out)]))
    assert code == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    d = rep["difference"]
    assert d["low"] > 0 and d["estimate"] > 0 and not d["contains_zero"]
    assert rep["a"]["estimate"] > rep["b"]["estimate"]
    assert rep["block"]["development"] is True and rep["schema"] == "dfd-eval-compare/1"
    # swapping the sides flips the sign of the interval
    main(compare_args(weak, ["--scores-b", str(strong), "--out", str(out)]))
    d2 = json.loads(out.read_text(encoding="utf-8"))["difference"]
    assert d2["high"] < 0
    assert d2["low"] == pytest.approx(-d["high"], abs=1e-12)
    assert "excludes 0" in capsys.readouterr().out


def test_compare_one_file_two_runs_matches_library_paired_bootstrap(make_scores):
    from dfd.eval.bootstrap import paired_bootstrap
    from dfd.eval.run import compare
    f = make_scores(seeds=(1, 2), groups_per_split=30, layout=DEV_LAYOUT, scorer="signal")
    t = read_scores(f)
    rep = compare(t, t, run_a="r1", run_b="r2", modality="visual", dataset="synthB",
                  split="dev_x", metric="auc", n_boot=200)
    m = (t.dataset == "synthB") & (t.split == "dev_x")
    sa = t.score[m & (t.run_id == "r1")]
    sb = t.score[m & (t.run_id == "r2")]
    order = np.argsort(t.sample_id[m & (t.run_id == "r1")])  # compare sorts by sample_id
    y = t.label[m & (t.run_id == "r1")][order]
    g = t.group_id[m & (t.run_id == "r1")][order]
    diff, _, _ = paired_bootstrap(y, sa[order], sb[order], g, "auc", n_boot=200)
    assert rep["difference"]["low"] == pytest.approx(diff.low, abs=1e-12)
    assert rep["difference"]["high"] == pytest.approx(diff.high, abs=1e-12)


def test_compare_input_errors(make_scores, tmp_path):
    f = make_scores(seeds=(1, 2), groups_per_split=10, layout=DEV_LAYOUT)
    base = compare_args(f)
    assert main(base) == 1  # one file needs --run-a and --run-b
    assert main(base + ["--run-a", "r1", "--run-b", "nope"]) == 1
    # misaligned samples between the two sides
    t = read_scores(f)
    drop = t.sample_id[(t.run_id == "r2") & (t.split == "dev_x")][0]
    keep = ~((t.run_id == "r2") & (t.sample_id == drop))
    cols = {c: getattr(t, c)[keep] for c in t.COLUMNS if c != "logit"}
    short = tmp_path / "short.csv"
    write_scores(short, cols)
    assert main(compare_args(short) + ["--run-a", "r1", "--run-b", "r2"]) == 1


def test_compare_on_final_split_needs_flag_and_ledger(make_scores, tmp_path, capsys):
    layout = [("val", "synthA"), ("test_x", "synthB")]
    f = make_scores(seeds=(1,), groups_per_split=15, layout=layout, scorer="signal")
    lg = str(tmp_path / "l.jsonl")
    args = compare_args(f, ["--run-a", "r1", "--run-b", "r1", "--ledger", lg], split="test_x")
    assert main(args) == 2  # no --final
    assert main(args + ["--final", "--iteration", "I-0"]) == 2  # not in the ledger yet
    assert "regular final evaluation" in capsys.readouterr().err
    assert main(["--scores", str(f), "--ledger", lg, "--n-boot", "20", "--final", "--iteration", "I-0"]) == 0
    n = len(L.read_ledger(lg))
    assert main(args + ["--final", "--iteration", "I-0"]) == 0
    assert len(L.read_ledger(lg)) == n  # a comparison writes no ledger entry
    with pytest.raises(GuardError):
        from dfd.eval.guard import check_compare_guard
        check_compare_guard(read_scores(f), final=True, iteration="I-9", ledger_path=lg)
