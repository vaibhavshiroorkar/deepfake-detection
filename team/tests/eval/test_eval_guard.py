"""A5: the final-test guard and the ledger. Only temporary ledgers are used."""

import datetime as dt
import json

import pytest

from dfd.eval import ledger as L
from dfd.eval.guard import GuardError, check_guard
from dfd.eval.run import main
from dfd.eval.scorefile import read_scores


@pytest.fixture
def table(make_scores):
    return read_scores(make_scores(seeds=(1,), groups_per_split=10))


def test_a5_refuses_final_split_without_flag(table, tmp_path):
    with pytest.raises(GuardError, match="--final"):
        check_guard(table, final=False, iteration="I-0", ledger_path=tmp_path / "l.jsonl")
    with pytest.raises(GuardError, match="--iteration"):
        check_guard(table, final=True, ledger_path=tmp_path / "l.jsonl")
    with pytest.raises(GuardError, match="does not match"):
        check_guard(table, final=True, iteration="I-9", ledger_path=tmp_path / "l.jsonl")


def test_a5_first_accepted_second_refused_without_repeat(table, tmp_path):
    lg = tmp_path / "l.jsonl"
    d = check_guard(table, final=True, iteration="I-0", ledger_path=lg, today=dt.date(2026, 10, 24))
    assert not d.is_repeat
    assert {(e["dataset"], e["split"]) for e in d.entries} == {("synthA", "test_in"), ("synthB", "test_x")}
    L.append_entries(d.entries, lg)
    e = L.read_ledger(lg)[0]
    assert e["date"] == "2026-10-24" and e["iteration"] == "I-0"
    assert e["run_ids"] == ["r1"] and len(e["score_file_sha256"]) == 64
    with pytest.raises(GuardError, match="--repeat"):
        check_guard(table, final=True, iteration="I-0", ledger_path=lg)
    with pytest.raises(GuardError, match="--repeat"):
        check_guard(table, final=True, iteration="I-0", repeat="   ", ledger_path=lg)
    d2 = check_guard(table, final=True, iteration="I-0", repeat="seed bug fixed", ledger_path=lg)
    assert d2.is_repeat and all(e["repeat_reason"] == "seed bug fixed" for e in d2.entries)


def test_guard_other_iteration_is_a_first_evaluation(make_scores, tmp_path):
    lg = tmp_path / "l.jsonl"
    t0 = read_scores(make_scores("a.csv", seeds=(1,), groups_per_split=5, iteration="I-0"))
    t1 = read_scores(make_scores("b.csv", seeds=(1,), groups_per_split=5, iteration="I-1"))
    L.append_entries(check_guard(t0, final=True, iteration="I-0", ledger_path=lg).entries, lg)
    assert not check_guard(t1, final=True, iteration="I-1", ledger_path=lg).is_repeat


def test_no_final_rows_needs_no_flag(table, tmp_path):
    val_only = table.select(table.split == "val")
    d = check_guard(val_only, ledger_path=tmp_path / "l.jsonl")
    assert d.entries == []


def test_a5_through_the_command_line(make_scores, tmp_path, capsys):
    f = str(make_scores(seeds=(1,), groups_per_split=10))
    lg = str(tmp_path / "ledger.jsonl")
    base = ["--scores", f, "--ledger", lg, "--n-boot", "20"]
    assert main(base) == 2  # refused, no flag
    assert "refused" in capsys.readouterr().err
    assert not (tmp_path / "ledger.jsonl").exists()
    assert main(base + ["--final"]) == 2  # no iteration
    assert main(base + ["--final", "--iteration", "I-0"]) == 0  # first one accepted
    assert len(L.read_ledger(lg)) == 2
    assert main(base + ["--final", "--iteration", "I-0"]) == 2  # second refused
    assert len(L.read_ledger(lg)) == 2
    assert main(base + ["--final", "--iteration", "I-0", "--repeat", "checking"]) == 0
    rows = L.read_ledger(lg)
    assert len(rows) == 4 and rows[-1]["repeat_reason"] == "checking"
    # restricting to val needs no flag and writes nothing
    assert main(base + ["--splits", "val"]) == 0
    assert len(L.read_ledger(lg)) == 4


def test_default_ledger_is_the_repo_one_and_untouched_by_tests():
    assert L.DEFAULT_LEDGER.name == "final_ledger.jsonl"
    assert L.DEFAULT_LEDGER.parent.name == "ledger"


def test_corrupt_ledger_is_an_error(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text("not json\n", encoding="utf-8")
    with pytest.raises(L.LedgerError):
        L.read_ledger(p)
    p.write_text(json.dumps({"x": 1}) + "\n", encoding="utf-8")
    with pytest.raises(L.LedgerError):
        L.read_ledger(p)
