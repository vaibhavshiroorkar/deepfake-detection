"""A8: end to end through dfd.eval.run with synthetic score files."""

import json

import pytest

from dfd.eval.run import main


def run(tmp_path, csv, extra=()):
    out = tmp_path / "report.json"
    ledger = tmp_path / "ledger.jsonl"
    code = main(["--scores", str(csv), "--final", "--iteration", "I-0", "--ledger", str(ledger),
                 "--out", str(out), *extra])
    assert code == 0
    return json.loads(out.read_text(encoding="utf-8"))


def test_a8_random_scorer_gives_chance_auc(make_scores, tmp_path, capsys):
    csv = make_scores(seed=20261024, seeds=(1, 2, 3), scorer="random", groups_per_split=100)
    rep = run(tmp_path, csv)
    assert rep["schema"] == "dfd-eval-report/1"
    cand = rep["candidates"][0]
    assert cand["seeds"] == [1, 2, 3]
    splits = {(b["dataset"], b["split"]) for b in cand["blocks"]}
    assert splits == {("synthA", "train"), ("synthA", "val"), ("synthA", "test_in"), ("synthB", "test_x")}
    for b in cand["blocks"]:
        a = b["metrics"]["auc"]
        assert abs(a["estimate"] - 0.5) < 0.06, (b["split"], a)
        assert a["low"] <= 0.5 <= a["high"], (b["split"], a)
        assert a["seed_sd"] is not None
        # a random scorer is badly calibrated only through its spread; ECE after fit is small
        assert b["metrics"]["ece_cal"]["estimate"] < 0.12
    out = capsys.readouterr().out
    assert "synthB" in out and "AUC" in out


def test_report_content_and_calibration_properties(make_scores, tmp_path):
    csv = make_scores(seed=3, seeds=(1, 2), scorer="signal", groups_per_split=100)
    rep = run(tmp_path, csv, ["--n-boot", "200"])
    cand = rep["candidates"][0]
    for f in cand["fits"].values():
        assert 0 < f["threshold"] < 1
        assert f["val_nll"] <= f["val_nll_at_T1"] + 1e-12
        assert f["val_datasets"] == ["synthA"]
    by = {(b["dataset"], b["split"]): b for b in cand["blocks"]}
    assert by[("synthB", "test_x")]["metrics"]["auc"]["estimate"] > 0.8
    assert by[("synthA", "val")]["fitted_on_these_rows"] is True
    assert by[("synthB", "test_x")]["fitted_on_these_rows"] is False
    # temperature scaling on the fitting rows cannot raise the log loss
    v = by[("synthA", "val")]["metrics"]
    assert v["log_loss_cal"]["estimate"] <= v["log_loss"]["estimate"] + 1e-9
    # ranking metrics are unchanged by calibration, so they are not reported twice
    assert "auc_cal" not in v
    assert rep["final"]["ledger_entries"][0]["iteration"] == "I-0"
    assert rep["protocol"]["n_boot"] == 200 and rep["protocol"]["seed"] == 20261024


def test_run_without_final_splits_needs_no_flag(make_scores, tmp_path):
    csv = make_scores(seeds=(1,), groups_per_split=20)
    out = tmp_path / "r.json"
    assert main(["--scores", str(csv), "--ledger", str(tmp_path / "l.jsonl"), "--splits",
                 "train,val", "--n-boot", "50", "--out", str(out)]) == 0
    blocks = json.loads(out.read_text(encoding="utf-8"))["candidates"][0]["blocks"]
    assert {b["split"] for b in blocks} == {"train", "val"}
    assert not (tmp_path / "l.jsonl").exists()


def test_bad_inputs_exit_one(tmp_path, make_scores):
    assert main(["--scores", str(tmp_path / "missing.csv")]) == 1
    csv = make_scores(seeds=(1, 2), groups_per_split=5)
    text = csv.read_text(encoding="utf-8").replace("r2", "r1", 1)
    bad = tmp_path / "bad.csv"
    bad.write_text(text, encoding="utf-8")
    assert main(["--scores", str(bad), "--final", "--iteration", "I-0",
                 "--ledger", str(tmp_path / "l.jsonl")]) == 1
