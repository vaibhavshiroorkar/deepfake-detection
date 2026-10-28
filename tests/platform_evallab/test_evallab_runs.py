"""Metrics, threshold provenance, integrity rules, freezing and reproducibility of lab runs."""

import json

import numpy as np
import pytest
from evallab_helpers import block, cfg_for, log_entry, make_lab, synth, write_rows

from dfd.eval.metrics import roc_auc
from dfd.eval.scorefile import ScoreFileError, file_sha256
from dfd.eval.threshold import choose_threshold_max_balanced_accuracy
from dfd.platform.evallab import analysis
from dfd.platform.evallab.common import LabError
from dfd.platform.evallab.runner import config_hash

VAL_FAKE, VAL_REAL = [0.6, 0.7, 0.8, 0.9], [0.1, 0.2, 0.3, 0.4]          # best val threshold: 0.5
DEV_FAKE, DEV_REAL = [0.85, 0.9, 0.95, 0.97], [0.5, 0.6, 0.7, 0.8]       # best threshold on these rows: 0.825


def threshold_file(scores_dir, name="t.csv"):
    rows = block("val", "dsA", VAL_FAKE, VAL_REAL) + block("dev_x", "dsA", DEV_FAKE, DEV_REAL, gprefix="dev", sprefix="dev")
    return write_rows(scores_dir / name, rows)


def test_metrics_on_a_hand_computable_fixture(tmp_path, scores_dir):
    # fake 0.9, 0.6; real 0.4, 0.7. AUC = 3/4. At t = 0.5: TP 2, FN 0, FP 1 (0.7), TN 1.
    p = write_rows(scores_dir / "hand.csv", block("dev_x", "dsA", [0.9, 0.6], [0.4, 0.7]))
    lab = make_lab(tmp_path)
    rec = lab.runner.submit(cfg_for(lab, p, threshold_source="fixed", threshold=0.5, threshold_provenance="hand fixture"))
    assert rec["state"] == "done", rec["error"]
    m = rec["metrics"]
    assert m["auc"]["estimate"] == pytest.approx(0.75)
    assert m["recall"]["estimate"] == pytest.approx(1.0)
    assert m["fnr"]["estimate"] == pytest.approx(0.0)
    assert m["fpr"]["estimate"] == pytest.approx(0.5)
    assert m["specificity"]["estimate"] == pytest.approx(0.5)
    assert m["precision"]["estimate"] == pytest.approx(2 / 3)
    assert m["f1"]["estimate"] == pytest.approx(0.8)
    assert m["accuracy"]["estimate"] == pytest.approx(0.75)
    assert m["balanced_accuracy"]["estimate"] == pytest.approx(0.75)
    assert m["eer"]["estimate"] == pytest.approx(0.5)      # t = 0.7: FPR 1/2 = FNR 1/2
    assert {k: rec["confusion"][k] for k in ("tp", "fp", "tn", "fn")} == {"tp": 2, "fp": 1, "tn": 1, "fn": 0}
    assert len(rec["roc"]["fpr"]) == len(rec["roc"]["tpr"]) <= 200
    assert rec["roc"]["fpr"][0] == 0 and rec["roc"]["tpr"][-1] == 1
    assert rec["data"]["n_samples"] == 4 and rec["data"]["n_groups"] == 4
    assert rec["protocol"]["seed"] == 20261027 and rec["protocol"]["n_boot"] == 200
    assert rec["threshold"]["source"] == "fixed" and rec["threshold"]["provenance"] == "hand fixture"


def test_threshold_is_fitted_on_val_only(tmp_path, scores_dir):
    p = threshold_file(scores_dir)
    lab = make_lab(tmp_path)
    rec = lab.runner.submit(cfg_for(lab, p))
    assert rec["state"] == "done", rec["error"]
    t_val = choose_threshold_max_balanced_accuracy([1] * 4 + [0] * 4, VAL_FAKE + VAL_REAL).threshold
    t_eval = choose_threshold_max_balanced_accuracy([1] * 4 + [0] * 4, DEV_FAKE + DEV_REAL).threshold
    assert t_val != pytest.approx(t_eval)
    assert rec["threshold"]["value"] == pytest.approx(t_val)
    assert rec["threshold"]["fitted_on_these_rows"] is False
    # the val threshold 0.5 sits below every dev real score: FPR 1, recall 1 (a dev-fitted one would give FPR 0.25)
    assert rec["metrics"]["fpr"]["estimate"] == pytest.approx(1.0)
    assert rec["metrics"]["recall"]["estimate"] == pytest.approx(1.0)


def test_val_split_is_flagged_as_fitted_on_these_rows(tmp_path, scores_dir):
    p = threshold_file(scores_dir)
    lab = make_lab(tmp_path)
    rec = lab.runner.submit(cfg_for(lab, p, split="val", purpose="validation"))
    assert rec["threshold"]["fitted_on_these_rows"] is True
    assert any("fitted on these rows" in w for w in rec["data"]["warnings"])


def test_multi_seed_follows_the_eval_run_convention(tmp_path, scores_dir):
    rng = np.random.default_rng(1)
    rows = []
    for sd in (0, 1, 2):
        fv, rv = synth(rng, 30, 30)
        fd, rd = synth(rng, 40, 40, strength=1.5)
        run = f"R-s{sd}"
        rows += block("val", "dsA", fv, rv, run=run, seed=sd, gprefix="val", sprefix="val")
        rows += block("dev_x", "dsB", fd, rd, run=run, seed=sd, gprefix="dev", sprefix="dev")
    p = write_rows(scores_dir / "ms.csv", rows)
    lab = make_lab(tmp_path)
    rec = lab.runner.submit(cfg_for(lab, p, dataset="dsB", n_boot=300))
    assert rec["state"] == "done", rec["error"]
    assert rec["data"]["run_ids"] == ["R-s0", "R-s1", "R-s2"] and rec["data"]["seeds"] == [0, 1, 2]
    per_run = rec["threshold"]["per_run"]
    assert set(per_run) == {"R-s0", "R-s1", "R-s2"}
    assert rec["threshold"]["value"] == pytest.approx(np.mean(list(per_run.values())))
    per_seed = rec["metrics"]["auc"]["per_seed"]
    assert len(per_seed) == 3
    assert rec["metrics"]["auc"]["estimate"] == pytest.approx(np.mean(per_seed))
    assert rec["metrics"]["auc"]["seed_sd"] == pytest.approx(np.std(per_seed, ddof=1))
    y = np.array([1] * 40 + [0] * 40)
    s0 = np.array([float(r["score"]) for r in rows if r["run_id"] == "R-s0" and r["split"] == "dev_x"])
    assert per_seed[0] == pytest.approx(roc_auc(y, s0))
    assert len(rec["confusion"]["per_run"]) == 3 and len(rec["roc"]["fpr"]) == 200


def test_subset_of_runs_and_unknown_run_rejected(tmp_path, scores_dir):
    rng = np.random.default_rng(2)
    rows = []
    for sd in (0, 1):
        fv, rv = synth(rng, 20, 20)
        rows += block("val", "dsA", fv, rv, run=f"R-s{sd}", seed=sd, gprefix="val", sprefix="val")
        rows += block("dev_x", "dsA", fv, rv, run=f"R-s{sd}", seed=sd, gprefix="dev", sprefix="dev")
    p = write_rows(scores_dir / "two.csv", rows)
    lab = make_lab(tmp_path)
    rec = lab.runner.submit(cfg_for(lab, p, run_ids=["R-s1"]))
    assert rec["state"] == "done" and rec["data"]["run_ids"] == ["R-s1"]
    with pytest.raises(LabError) as e:
        lab.runner.submit(cfg_for(lab, p, run_ids=["R-s9"]))
    assert e.value.code == "INVALID_CONFIG"


@pytest.mark.parametrize("bad", [
    {"threshold_source": "fixed"},                                                 # no value
    {"threshold_source": "fixed", "threshold": 0.5},                               # no provenance
    {"threshold_source": "fixed", "threshold": 1.5, "threshold_provenance": "x"},  # out of range
    {"threshold": 0.5},                                                            # value with val_fit
    {"n_boot": 10}, {"n_boot": 99999}, {"n_boot": "many"}, {"seed": 1.5},
    {"split": "nonsense"}, {"purpose": "tuning"}, {"purpose": "final_reanalysis"},  # final purpose on dev_x
    {"dataset": "nope"}, {"iteration": "I-77"}, {"bogus_key": 1}, {"label": "x" * 500},
])
def test_invalid_configs_raise_invalid_config(tmp_path, scores_dir, bad):
    p = threshold_file(scores_dir)
    lab = make_lab(tmp_path)
    with pytest.raises(LabError) as e:
        lab.runner.submit(cfg_for(lab, p, **bad))
    assert e.value.code == "INVALID_CONFIG"
    assert not lab.store.dir.exists()                          # nothing was recorded for a malformed request


# ---------------------------------------------------------------- final splits

def final_file(scores_dir, name="f.csv"):
    rng = np.random.default_rng(3)
    fv, rv = synth(rng, 40, 40)
    ft, rt = synth(rng, 40, 40, strength=1.0)
    rows = (block("val", "dsA", fv, rv, gprefix="val", sprefix="val")
            + block("test_x", "dsB", ft, rt, gprefix="tx", sprefix="tx"))
    return write_rows(scores_dir / name, rows)


def final_cfg(lab, p, **kw):
    return cfg_for(lab, p, dataset="dsB", split="test_x", purpose="final_reanalysis", **kw)


def test_unlogged_final_split_is_refused_and_ledger_untouched(tmp_path, scores_dir):
    p = final_file(scores_dir)
    lab = make_lab(tmp_path)
    rec = lab.runner.submit(final_cfg(lab, p))
    assert rec["state"] == "refused"
    assert rec["error"]["code"] == "FINAL_GUARD" and "ledger" in rec["error"]["message"]
    assert rec["metrics"] is None
    assert not lab.ledger_path.exists()


def test_final_split_needs_the_reanalysis_purpose(tmp_path, scores_dir, ledger):
    p = final_file(scores_dir)
    log_entry(ledger, iteration="I-0", dataset="dsB", split="test_x", sha=file_sha256(p))
    lab = make_lab(tmp_path, ledger_path=ledger)
    rec = lab.runner.submit(cfg_for(lab, p, dataset="dsB", split="test_x", purpose="development"))
    assert rec["state"] == "refused" and rec["error"]["code"] == "FINAL_PURPOSE"


def test_logged_final_split_is_accepted_without_a_ledger_write(tmp_path, scores_dir, ledger):
    p = final_file(scores_dir)
    log_entry(ledger, iteration="I-0", dataset="dsB", split="test_x", sha=file_sha256(p))
    before = ledger.read_bytes()
    lab = make_lab(tmp_path, ledger_path=ledger)
    rec = lab.runner.submit(final_cfg(lab, p))
    assert rec["state"] == "done", rec["error"]
    assert rec["final_guard"]["required"] is True and len(rec["final_guard"]["ledger_entries"]) == 1
    assert any("not a new held-out look" in w for w in rec["data"]["warnings"])
    assert ledger.read_bytes() == before


def test_final_split_with_another_files_hash_is_refused(tmp_path, scores_dir, ledger):
    p = final_file(scores_dir)
    log_entry(ledger, iteration="I-0", dataset="dsB", split="test_x", sha="0" * 64)
    lab = make_lab(tmp_path, ledger_path=ledger)
    rec = lab.runner.submit(final_cfg(lab, p))
    assert rec["state"] == "refused" and "not the one logged" in rec["error"]["message"]


def test_final_split_logged_for_another_dataset_is_refused(tmp_path, scores_dir, ledger):
    p = final_file(scores_dir)
    log_entry(ledger, iteration="I-0", dataset="dsOther", split="test_x", sha=file_sha256(p))
    lab = make_lab(tmp_path, ledger_path=ledger)
    rec = lab.runner.submit(final_cfg(lab, p))
    assert rec["state"] == "refused" and rec["error"]["code"] == "FINAL_GUARD"


def test_final_split_refused_on_group_leakage(tmp_path, scores_dir, ledger):
    rng = np.random.default_rng(4)
    fv, rv = synth(rng, 40, 40)
    rows = (block("val", "dsA", fv, rv, gprefix="shared", sprefix="val")
            + block("test_x", "dsB", fv, rv, gprefix="shared", sprefix="tx"))
    p = write_rows(scores_dir / "leak.csv", rows)
    log_entry(ledger, iteration="I-0", dataset="dsB", split="test_x", sha=file_sha256(p))
    lab = make_lab(tmp_path, ledger_path=ledger)
    rec = lab.runner.submit(final_cfg(lab, p))
    assert rec["state"] == "refused" and rec["error"]["code"] == "LEAKAGE"


def test_final_split_refused_on_hash_leakage(tmp_path, scores_dir, ledger):
    rng = np.random.default_rng(9)
    fv, rv = synth(rng, 40, 40)
    val = block("val", "dsA", fv, rv, gprefix="val", sprefix="val")
    fin = block("test_x", "dsB", fv, rv, gprefix="tx", sprefix="tx")
    for i, r in enumerate(val):
        r["file_hash"] = f"h{i}"
    for i, r in enumerate(fin):
        r["file_hash"] = "h0" if i == 0 else f"t{i}"
    p = write_rows(scores_dir / "hleak.csv", val + fin, extra=("file_hash",))
    log_entry(ledger, iteration="I-0", dataset="dsB", split="test_x", sha=file_sha256(p))
    lab = make_lab(tmp_path, ledger_path=ledger)
    rec = lab.runner.submit(final_cfg(lab, p))
    assert rec["state"] == "refused" and rec["error"]["code"] == "LEAKAGE" and "hash" in rec["error"]["message"]


# --------------------------------------------------------------------- leakage

def test_group_overlap_with_val_is_a_warning_on_a_dev_split(tmp_path, scores_dir):
    rng = np.random.default_rng(5)
    fv, rv = synth(rng, 20, 20)
    rows = (block("val", "dsA", fv, rv, gprefix="g", sprefix="val")
            + block("dev_x", "dsA", fv, rv, gprefix="g", sprefix="dev"))
    p = write_rows(scores_dir / "ov.csv", rows)
    lab = make_lab(tmp_path)
    rec = lab.runner.submit(cfg_for(lab, p))
    assert rec["state"] == "done"
    assert rec["data"]["leakage"]["group_overlap"]["n_shared_groups"] == 40
    assert any("40 group id" in w for w in rec["data"]["warnings"])


def test_duplicate_sample_ids_are_reported(tmp_path, scores_dir):
    rng = np.random.default_rng(6)
    fv, rv = synth(rng, 20, 20)
    dev = block("dev_x", "dsA", fv, rv, gprefix="dev", sprefix="dev")
    dup = [dict(r, score=round(float(r["score"]) * 0.9, 6)) for r in dev[:3]]       # frame rows of 3 samples
    rows = block("val", "dsA", fv, rv, gprefix="val", sprefix="val") + dev + dup
    p = write_rows(scores_dir / "dup.csv", rows)
    lab = make_lab(tmp_path)
    rec = lab.runner.submit(cfg_for(lab, p))
    assert rec["state"] == "done"
    assert rec["data"]["leakage"]["duplicate_sample_rows"]["n_samples"] == 3
    assert rec["data"]["n_samples"] == 40
    assert any("repeat in the block" in w for w in rec["data"]["warnings"])


def test_duplicate_hash_across_splits_is_reported(tmp_path, scores_dir):
    rng = np.random.default_rng(7)
    fv, rv = synth(rng, 20, 20)
    val = block("val", "dsA", fv, rv, gprefix="val", sprefix="val")
    dev = block("dev_x", "dsA", fv, rv, gprefix="dev", sprefix="dev")
    for i, r in enumerate(val):
        r["sha256"] = f"hv{i}"
    for i, r in enumerate(dev):
        r["sha256"] = f"hv{i}" if i < 5 else f"hd{i}"           # 5 dev files are byte copies of val files
    p = write_rows(scores_dir / "hash.csv", val + dev, extra=("sha256",))
    lab = make_lab(tmp_path)
    rec = lab.runner.submit(cfg_for(lab, p))
    assert rec["state"] == "done"
    assert rec["data"]["leakage"]["hash"]["n_shared_with_development_splits"] == 5
    assert any("5 file hash" in w for w in rec["data"]["warnings"])


def test_no_hash_column_is_stated(tmp_path, scores_dir):
    p = threshold_file(scores_dir)
    lab = make_lab(tmp_path)
    rec = lab.runner.submit(cfg_for(lab, p))
    assert rec["data"]["leakage"]["hash"]["checked"] is False
    assert any("could not be checked" in w for w in rec["data"]["warnings"])


def test_data_quality_warnings(tmp_path, scores_dir):
    rows = (block("val", "dsA", VAL_FAKE, VAL_REAL, gprefix="val", sprefix="val")
            + block("dev_x", "dsA", [0.9, 0.8], [0.1, 0.2, 0.3, 0.4] * 6, gprefix="dev", sprefix="dev"))
    p = write_rows(scores_dir / "q.csv", rows)
    lab = make_lab(tmp_path)
    rec = lab.runner.submit(cfg_for(lab, p))
    w = " ".join(rec["data"]["warnings"])
    assert "class imbalance" in w and "groups" in w and "development slice" in w


def test_one_class_block_is_refused(tmp_path, scores_dir):
    rows = (block("val", "dsA", VAL_FAKE, VAL_REAL, gprefix="val", sprefix="val")
            + block("dev_x", "dsA", [0.9, 0.8], [], gprefix="d", sprefix="d"))
    p = write_rows(scores_dir / "one.csv", rows)
    lab = make_lab(tmp_path)
    rec = lab.runner.submit(cfg_for(lab, p))
    assert rec["state"] == "refused" and rec["error"]["code"] == "ONE_CLASS"


def test_missing_val_rows_refuse_val_fit(tmp_path, scores_dir):
    p = write_rows(scores_dir / "noval.csv", block("dev_x", "dsA", DEV_FAKE, DEV_REAL))
    lab = make_lab(tmp_path)
    rec = lab.runner.submit(cfg_for(lab, p))
    assert rec["state"] == "refused" and rec["error"]["code"] == "NO_VAL_ROWS"


def test_breakdown_by_generator_column(tmp_path, scores_dir):
    rng = np.random.default_rng(8)
    fv, rv = synth(rng, 20, 20)
    val = block("val", "dsA", fv, rv, gprefix="val", sprefix="val")
    dev = block("dev_x", "dsA", fv, rv, gprefix="dev", sprefix="dev")
    for i, r in enumerate(dev):
        r["generator"] = ("gan" if i % 2 == 0 else "diffusion") if r["label"] == 1 else "camera"
    for r in val:
        r["generator"] = "x"
    p = write_rows(scores_dir / "bd.csv", val + dev, extra=("generator",))
    lab = make_lab(tmp_path)
    rec = lab.runner.submit(cfg_for(lab, p))
    b = rec["breakdown"]
    assert b["column"] == "generator"
    by = {r["value"]: r for r in b["rows"]}
    assert set(by) == {"gan", "diffusion", "camera"}
    assert by["camera"]["n_fake"] == 0 and by["camera"]["auc"] is None and by["camera"]["fnr"] is None
    assert by["gan"]["n_real"] == 0 and by["gan"]["fpr"] is None and by["gan"]["fnr"] is not None
    assert sum(r["n_fake"] + r["n_real"] for r in b["rows"]) == 40


def test_no_breakdown_without_a_column(tmp_path, scores_dir):
    p = threshold_file(scores_dir)
    lab = make_lab(tmp_path)
    assert lab.runner.submit(cfg_for(lab, p))["breakdown"] is None


def test_score_file_rejected_by_the_reader_at_run_time_is_refused(tmp_path, scores_dir, monkeypatch):
    p = threshold_file(scores_dir)
    lab = make_lab(tmp_path)
    cfg = cfg_for(lab, p)

    def boom(path):
        raise ScoreFileError("column 'score' must be finite and within [0, 1]")

    monkeypatch.setattr(lab.tables, "get", boom)
    rec = lab.runner.submit(cfg)
    assert rec["state"] == "refused" and rec["error"]["code"] == "SCORE_FILE_INVALID"
    assert "within [0, 1]" in rec["error"]["message"]


# ------------------------------------------------------- freezing, reproducibility

def test_config_is_frozen_before_compute_and_hash_is_stable(tmp_path, scores_dir, monkeypatch):
    p = threshold_file(scores_dir)
    lab = make_lab(tmp_path)
    seen = {}
    real_compute = analysis.compute

    def spy(cfg, table, extras, **kw):
        run_json = next(lab.store.dir.iterdir()) / "run.json"          # what is on disk while computing
        seen.update(json.loads(run_json.read_text(encoding="utf-8")))
        return real_compute(cfg, table, extras, **kw)

    monkeypatch.setattr(analysis, "compute", spy)
    cfg = cfg_for(lab, p)
    rec = lab.runner.submit(cfg)
    assert rec["state"] == "done"
    assert seen["state"] == "running" and seen["metrics"] is None
    assert seen["config"] == rec["config"] and seen["config_sha256"] == rec["config_sha256"] == config_hash(rec["config"])
    assert seen["score_file"]["sha256"] == file_sha256(p)
    assert "git_head" in seen["environment"] and "dirty_entries" in seen["environment"]
    monkeypatch.undo()
    again = lab.runner.submit(cfg)
    assert again["config_sha256"] == rec["config_sha256"] and again["id"] != rec["id"]


def test_same_config_twice_gives_identical_metrics(tmp_path, scores_dir):
    p = threshold_file(scores_dir)
    lab = make_lab(tmp_path)
    a = lab.runner.submit(cfg_for(lab, p))
    b = lab.runner.submit(cfg_for(lab, p))
    assert a["metrics"] == b["metrics"] and a["confusion"] == b["confusion"] and a["roc"] == b["roc"]
    c = lab.runner.submit(cfg_for(lab, p, seed=7))
    assert c["config_sha256"] != a["config_sha256"]


def test_file_changed_after_freeze_is_refused(tmp_path, scores_dir, monkeypatch):
    p = threshold_file(scores_dir)
    lab = make_lab(tmp_path)
    real = analysis.compute

    def altered(cfg, table, extras, **kw):
        kw["frozen_sha256"] = "f" * 64
        return real(cfg, table, extras, **kw)

    monkeypatch.setattr(analysis, "compute", altered)
    rec = lab.runner.submit(cfg_for(lab, p))
    assert rec["state"] == "refused" and rec["error"]["code"] == "FILE_CHANGED"


def test_runs_survive_a_restart_and_unfinished_ones_are_marked_failed(tmp_path, scores_dir):
    p = threshold_file(scores_dir)
    lab = make_lab(tmp_path)
    rec = lab.runner.submit(cfg_for(lab, p))
    stale = dict(rec, id="20260101T000000-deadbeef", state="running")
    lab.store.save(stale)
    lab2 = make_lab(tmp_path)
    assert lab2.store.get(rec["id"])["metrics"] == rec["metrics"]
    assert lab2.store.get(stale["id"])["state"] == "failed"
    assert lab2.store.get(stale["id"])["error"]["code"] == "INTERRUPTED"
