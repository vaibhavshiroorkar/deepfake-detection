"""Catalogue, HTTP API, comparison, export and the forbidden-operations check of the Evaluation Lab."""

import json
import re
import time
from pathlib import Path

import numpy as np
import pytest
from evallab_helpers import block, cfg_for, log_entry, make_lab, synth, write_rows
from flask import Flask

import dfd.platform.evallab as evallab_pkg
from dfd.eval.scorefile import file_sha256
from dfd.platform.dashboard.serve import create_app
from dfd.platform.evallab import register
from dfd.platform.evallab.common import ROLES, file_id

VAL_FAKE, VAL_REAL = [0.6, 0.7, 0.8, 0.9], [0.1, 0.2, 0.3, 0.4]


def two_iteration_file(scores_dir, name="two_it.csv", dev_sample_prefix="dev", n=30, seed=11):
    """Iterations I-0 and I-1 over identical samples (so they can be compared), different scores."""
    rng = np.random.default_rng(seed)
    rows = []
    for it, strength in (("I-0", 1.0), ("I-1", 2.5)):
        fv, rv = synth(rng, n, n)
        fd, rd = synth(rng, n, n, strength=strength)
        rows += block("val", "dsA", fv, rv, run=f"{it}-s0", it=it, gprefix="val", sprefix="val")
        rows += block("dev_x", "dsB", fd, rd, run=f"{it}-s0", it=it, gprefix="dev", sprefix=dev_sample_prefix)
    return write_rows(scores_dir / name, rows)


def client_for(lab):
    app = Flask(__name__)
    register(app, lab)
    return app.test_client()


# ------------------------------------------------------------------- catalogue

def test_catalogue_lists_score_files_blocks_and_skips_other_csvs(tmp_path, scores_dir):
    p = two_iteration_file(scores_dir)
    (scores_dir / "notes.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    lab = make_lab(tmp_path)
    cat = lab.catalogue.scan()
    assert [sf["path"] for sf in cat["score_files"]] == ["scores/two_it.csv"]
    sf = cat["score_files"][0]
    assert sf["id"] == file_id("scores/two_it.csv") and sf["sha256"] == file_sha256(p)
    assert sf["n_rows"] == 4 * 30 * 2 and sf["iterations"] == ["I-0", "I-1"] and sf["modalities"] == ["visual"]
    blk = {(b["dataset"], b["split"]): b for b in sf["blocks"]}
    dev = blk[("dsB", "dev_x")]
    assert (dev["n_samples"], dev["n_fake"], dev["n_real"], dev["n_groups"]) == (60, 30, 30, 60)
    assert dev["runs"] == ["I-0-s0", "I-1-s0"] and dev["seeds"] == [0] and dev["role"] == ROLES["dev_x"]
    assert cat["roles"]["test_x"] == "final held-out (sealed)"
    assert cat["default_dirs"] == ["data/_final/scores", "data/_evallab/imports"]
    assert [s["path"] for s in cat["skipped_files"]] == ["scores/notes.csv"]
    assert {c["iteration"] for c in sf["candidates"]} == {"I-0", "I-1"}


def test_catalogue_is_cached_and_invalidated_by_change(tmp_path, scores_dir, monkeypatch):
    p = two_iteration_file(scores_dir)
    lab = make_lab(tmp_path)
    lab.catalogue.scan()
    cache = tmp_path / "data" / "_evallab" / "cache" / "catalogue.json"
    assert cache.is_file()
    import dfd.platform.evallab.catalogue as cat_mod

    parsed = []
    real = cat_mod.read_scores
    monkeypatch.setattr(cat_mod, "read_scores", lambda path: parsed.append(str(path)) or real(path))
    again = make_lab(tmp_path).catalogue.scan()                  # new object: reads the cache file only
    assert len(again["score_files"]) == 1 and parsed == []
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")            # same rows, new size
    changed = make_lab(tmp_path).catalogue.scan()
    assert len(parsed) == 1 and len(changed["score_files"]) == 1      # a changed file is parsed again


def test_catalogue_directories_come_from_the_environment(tmp_path, scores_dir, monkeypatch):
    two_iteration_file(scores_dir)
    other = tmp_path / "elsewhere"
    other.mkdir()
    write_rows(other / "o.csv", block("val", "dsA", VAL_FAKE, VAL_REAL))
    monkeypatch.setenv("DFD_EVALLAB_DIRS", "elsewhere;scores")
    from dfd.platform.evallab import EvalLab
    lab = EvalLab(root=tmp_path, data_dir=tmp_path / "d", ledger_path=tmp_path / "l.jsonl")
    assert sorted(s["path"] for s in lab.catalogue.scan()["score_files"]) == ["elsewhere/o.csv", "scores/two_it.csv"]
    assert lab.catalogue.scan()["dirs"] == ["elsewhere", "scores"]


def test_missing_directory_gives_an_empty_catalogue(tmp_path):
    cat = make_lab(tmp_path, dirs=["nowhere"]).catalogue.scan()
    assert cat["score_files"] == [] and cat["skipped_files"] == []


# ------------------------------------------------------------------------- API

def test_api_catalogue_run_list_get_export(tmp_path, scores_dir):
    two_iteration_file(scores_dir)
    lab = make_lab(tmp_path)
    c = client_for(lab)
    cat = c.get("/api/evallab/catalogue").get_json()
    fid = cat["score_files"][0]["id"]
    body = {"score_file_id": fid, "iteration": "I-1", "modality": "visual", "dataset": "dsB", "split": "dev_x",
            "n_boot": 200, "purpose": "development", "label": "api test"}
    r = c.post("/api/evallab/runs", json=body)
    assert r.status_code == 202
    rid = r.get_json()["id"]
    rec = c.get(f"/api/evallab/runs/{rid}").get_json()
    assert rec["state"] == "done" and rec["config"]["label"] == "api test"
    for key in ("created", "started", "finished", "progress", "config_sha256", "score_file", "role", "purpose",
                "final_guard", "threshold", "data", "metrics", "confusion", "roc", "breakdown", "runtime_s", "error",
                "environment"):
        assert key in rec
    assert set(rec["metrics"]) == {"auc", "eer", "precision", "recall", "specificity", "fpr", "fnr", "f1", "accuracy",
                                   "balanced_accuracy"}
    assert all({"estimate", "low", "high"} <= set(v) for v in rec["metrics"].values())
    lst = c.get("/api/evallab/runs").get_json()["runs"]
    assert [x["id"] for x in lst] == [rid] and lst[0]["state"] == "done" and lst[0]["auc"] == rec["metrics"]["auc"]["estimate"]
    j = c.get(f"/api/evallab/runs/{rid}/export?format=json")
    assert j.status_code == 200 and json.loads(j.data) == rec and "attachment" in j.headers["Content-Disposition"]
    md = c.get(f"/api/evallab/runs/{rid}/export?format=md").get_data(as_text=True)
    for needle in (rid, rec["config_sha256"], rec["score_file"]["sha256"], "Role of the split", "Threshold",
                   "| auc |", "Confusion matrix", "TP ", "Limitations", "cannot verify", "not a new held-out look",
                   "max balanced accuracy on the val rows"):
        assert needle in md
    assert "—" not in md                                       # no em-dashes
    assert c.get(f"/api/evallab/runs/{rid}/export?format=pdf").status_code == 400


def test_api_errors(tmp_path, scores_dir):
    two_iteration_file(scores_dir)
    lab = make_lab(tmp_path)
    c = client_for(lab)
    fid = c.get("/api/evallab/catalogue").get_json()["score_files"][0]["id"]
    ok = {"score_file_id": fid, "iteration": "I-1", "modality": "visual", "dataset": "dsB", "split": "dev_x"}
    for bad, code in (({}, "INVALID_CONFIG"), ({**ok, "n_boot": 5}, "INVALID_CONFIG"), ({**ok, "split": "test_x"}, "INVALID_CONFIG"),
                      ({**ok, "threshold_source": "fixed"}, "INVALID_CONFIG"), ({**ok, "extra": 1}, "INVALID_CONFIG")):
        r = c.post("/api/evallab/runs", json=bad)
        assert r.status_code == 400 and r.get_json()["error"]["code"] == code and r.get_json()["error"]["message"], bad
    r = c.post("/api/evallab/runs", json={**ok, "score_file_id": "0" * 16})
    assert r.status_code == 404 and r.get_json()["error"]["code"] == "UNKNOWN_SCORE_FILE"
    assert c.post("/api/evallab/runs", data="not json", content_type="text/plain").status_code == 400
    assert c.get("/api/evallab/runs/nope").status_code == 404
    assert c.get("/api/evallab/runs/..%2F..%2Fx").status_code == 404
    assert c.get("/api/evallab/runs").get_json() == {"runs": []}


def test_api_refused_final_run_is_recorded_with_the_guard_reason(tmp_path, scores_dir):
    rng = np.random.default_rng(3)
    fv, rv = synth(rng, 40, 40)
    p = write_rows(scores_dir / "f.csv", block("val", "dsA", fv, rv, gprefix="v", sprefix="v")
                   + block("test_x", "dsB", fv, rv, gprefix="t", sprefix="t"))
    lab = make_lab(tmp_path)
    c = client_for(lab)
    fid = c.get("/api/evallab/catalogue").get_json()["score_files"][0]["id"]
    r = c.post("/api/evallab/runs", json={"score_file_id": fid, "iteration": "I-0", "modality": "visual", "dataset": "dsB",
                                          "split": "test_x", "purpose": "final_reanalysis"})
    rec = r.get_json()
    assert r.status_code == 202 and rec["state"] == "refused" and "ledger" in rec["error"]["message"]
    assert c.get(f"/api/evallab/runs/{rec['id']}").get_json()["state"] == "refused"
    assert not lab.ledger_path.exists()
    assert "Error (FINAL_GUARD)" in c.get(f"/api/evallab/runs/{rec['id']}/export?format=md").get_data(as_text=True)


def test_background_run_reports_progress_and_finishes(tmp_path, scores_dir):
    two_iteration_file(scores_dir)
    lab = make_lab(tmp_path, sync=False)
    c = client_for(lab)
    fid = c.get("/api/evallab/catalogue").get_json()["score_files"][0]["id"]
    rec = c.post("/api/evallab/runs", json={"score_file_id": fid, "iteration": "I-1", "modality": "visual",
                                            "dataset": "dsB", "split": "dev_x", "n_boot": 200}).get_json()
    assert rec["state"] in ("queued", "running", "done") and rec["progress"]["total"] == 6
    for _ in range(200):
        cur = c.get(f"/api/evallab/runs/{rec['id']}").get_json()
        if cur["state"] in ("done", "failed", "refused"):
            break
        time.sleep(0.05)
    assert cur["state"] == "done" and cur["progress"] == {"stage": "done", "done": 6, "total": 6}
    assert cur["runtime_s"] is not None and cur["finished"]


# --------------------------------------------------------------------- compare

def _two_runs(tmp_path, scores_dir, c, **second):
    fid = c.get("/api/evallab/catalogue").get_json()["score_files"][0]["id"]
    base = {"score_file_id": fid, "modality": "visual", "dataset": "dsB", "split": "dev_x", "n_boot": 300}
    a = c.post("/api/evallab/runs", json={**base, "iteration": "I-0"}).get_json()
    b = c.post("/api/evallab/runs", json={**base, "iteration": "I-1", **second}).get_json()
    return a, b


def test_compare_available_on_identical_samples(tmp_path, scores_dir):
    two_iteration_file(scores_dir)
    c = client_for(make_lab(tmp_path))
    a, b = _two_runs(tmp_path, scores_dir, c)
    r = c.post("/api/evallab/compare", json={"run_a": b["id"], "run_b": a["id"], "metric": "auc"}).get_json()
    assert r["available"] is True and r["metric"] == "auc"
    d = r["difference"]
    assert d["estimate"] == pytest.approx(r["a"]["estimate"] - r["b"]["estimate"])
    assert d["estimate"] > 0 and d["low"] <= d["estimate"] <= d["high"] and d["contains_zero"] is False
    assert r["block"]["n_samples"] == 60 and r["protocol"]["n_boot"] == 300
    again = c.post("/api/evallab/compare", json={"run_a": b["id"], "run_b": a["id"], "metric": "auc"}).get_json()
    assert again == r                                                          # reproducible


def test_compare_unavailable_on_different_samples(tmp_path, scores_dir):
    two_iteration_file(scores_dir, "a.csv")
    lab = make_lab(tmp_path)
    c = client_for(lab)
    fid_a = next(s["id"] for s in lab.catalogue.scan()["score_files"])
    ra = c.post("/api/evallab/runs", json={"score_file_id": fid_a, "iteration": "I-0", "modality": "visual",
                                           "dataset": "dsB", "split": "dev_x", "n_boot": 200}).get_json()
    two_iteration_file(scores_dir, "b.csv", dev_sample_prefix="other", seed=12)         # other sample ids
    fid_b = next(s["id"] for s in lab.catalogue.scan()["score_files"] if s["path"].endswith("b.csv"))
    rb = c.post("/api/evallab/runs", json={"score_file_id": fid_b, "iteration": "I-0", "modality": "visual",
                                           "dataset": "dsB", "split": "dev_x", "n_boot": 200}).get_json()
    r = c.post("/api/evallab/compare", json={"run_a": ra["id"], "run_b": rb["id"], "metric": "auc"}).get_json()
    assert r["available"] is False and "same samples" in r["reason"] and "difference" not in r


def test_compare_unavailable_for_a_refused_run_and_for_same_run(tmp_path, scores_dir):
    two_iteration_file(scores_dir)
    lab = make_lab(tmp_path)
    c = client_for(lab)
    a, _ = _two_runs(tmp_path, scores_dir, c)
    fid = a["score_file"]["path"]
    refused = lab.runner.submit(cfg_for(lab, Path(fid), iteration="I-1", dataset="dsB", split="dev_x", threshold_source="val_fit"))
    lab.store.save(dict(refused, state="refused"))
    r = c.post("/api/evallab/compare", json={"run_a": a["id"], "run_b": refused["id"], "metric": "auc"}).get_json()
    assert r["available"] is False and "refused" in r["reason"]
    r = c.post("/api/evallab/compare", json={"run_a": a["id"], "run_b": a["id"], "metric": "auc"}).get_json()
    assert r["available"] is False and "same run" in r["reason"]


def test_compare_request_validation(tmp_path, scores_dir):
    two_iteration_file(scores_dir)
    c = client_for(make_lab(tmp_path))
    a, b = _two_runs(tmp_path, scores_dir, c)
    assert c.post("/api/evallab/compare", json={"run_a": a["id"]}).status_code == 400
    assert c.post("/api/evallab/compare", json={"run_a": a["id"], "run_b": "nope-nope-1"}).status_code == 404
    r = c.post("/api/evallab/compare", json={"run_a": a["id"], "run_b": b["id"], "metric": "bogus"})
    assert r.status_code == 400 and r.get_json()["error"]["code"] == "INVALID_METRIC"


# --------------------------------------------------------------- serve.create_app

def test_dashboard_app_registers_the_lab_and_keeps_its_routes(tmp_path, scores_dir):
    two_iteration_file(scores_dir)
    app = create_app(evallab_options={"root": tmp_path, "dirs": ["scores"], "data_dir": tmp_path / "d",
                                      "ledger_path": tmp_path / "l.jsonl", "synchronous": True})
    c = app.test_client()
    assert c.get("/api/evallab/catalogue").get_json()["score_files"][0]["path"] == "scores/two_it.csv"
    assert c.get("/api/mode").status_code == 200 and c.get("/api/vocabulary").status_code == 200
    assert c.get("/").status_code == 200 and c.get("/dashboard.js").status_code == 200
    assert c.get("/api/fixtures").status_code == 200


# --------------------------------------------------------------- forbidden operations

FORBIDDEN = {
    "APPROVED_HEADS": r"APPROVED_HEADS",
    "checkpoints folder": r"checkpoints[/\\]",
    "ledger writer": r"append_entries|DEFAULT_LEDGER\s*\)?\s*\.\s*(write|open)",
    "final data folder write": r"_final[^\n]*(write_text|write_bytes|open\([^)]*['\"]w|shutil\.(copy|move|rmtree)|os\.remove|unlink)",
    "model or inference import": r"from\s+dfd\.(infer|web)|import\s+dfd\.(infer|web)|\btorch\b|\btransformers\b",
    "ledger file mode": r"ledger[^\n]*open\([^)]*['\"][wa]",
    "git state change": r"git[^\n]*['\"](add|commit|stash|checkout|reset|clean|push)['\"]",
}


def test_package_never_touches_product_state():
    pkg = Path(evallab_pkg.__file__).parent
    files = sorted(pkg.glob("*.py"))
    assert len(files) >= 8
    hits = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        for name, pat in FORBIDDEN.items():
            for m in re.finditer(pat, text):
                hits.append(f"{f.name}: {name}: {m.group(0)[:60]}")
    assert not hits, hits


def test_package_writes_only_under_its_data_directory():
    pkg = Path(evallab_pkg.__file__).parent
    writes = []
    for f in pkg.glob("*.py"):
        for ln, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"write_text|write_bytes|os\.replace|\.mkdir\(|open\([^)]*['\"]w", line):
                writes.append(f"{f.name}:{ln}")
    # every write site is in the catalogue cache, the run store or the tests' own data dir logic
    assert {w.split(":")[0] for w in writes} <= {"catalogue.py", "store.py"}, writes


def test_no_em_dashes_in_the_package():
    pkg = Path(evallab_pkg.__file__).parent
    for f in pkg.glob("*.py"):
        assert "—" not in f.read_text(encoding="utf-8"), f.name
