"""Recommended demonstrations: logged results read from their result files, displayed as logged, nothing run or written."""

import json

import pytest
from flask import Flask

from dfd.eval.scorefile import file_sha256
from dfd.platform.evallab import register
from dfd.platform.evallab.lab import EvalLab
from dfd.platform.evallab.showcase import ENTRIES, Showcase

REPO = Showcase().root


def _m(per_seed, low=0.1, high=0.9):
    return {"estimate": sum(per_seed) / len(per_seed), "low": low, "high": high, "per_seed": per_seed}


def write_sealed(root):
    """A synthetic dfd-eval-report/1 file with one visual block; every number is test data."""
    doc = {"protocol": {"n_boot": 2000, "threshold_rule": "max balanced accuracy on val"},
           "candidates": [{"modality": "visual",
                           "fits": {"X-s0": {"seed": 0, "threshold": 0.6}, "X-s1": {"seed": 1, "threshold": 0.55},
                                    "X-s2": {"seed": 2, "threshold": 0.5}},
                           "blocks": [{"dataset": "dsV", "split": "test_x", "n_samples": 100, "n_fake": 60, "n_real": 40,
                                       "n_groups": 20,
                                       "metrics": {"auc": _m([0.90, 0.80, 0.85]), "eer": _m([0.2, 0.2, 0.2]),
                                                   "balanced_accuracy": _m([0.80, 0.70, 0.75]),
                                                   "precision": _m([0.90, 0.80, 0.85]), "recall": _m([0.70, 0.60, 0.65])}}]}]}
    p = root / "rep.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def write_post(root):
    doc = {"experiment": "E-900", "n_boot": 2000, "datasets": {"dsP": {"groups": {"G1": {
        "lab_run_id": "run-1", "confusion": {"tp": 5, "fn": 5, "fp": 2, "tn": 8},
        "data": {"n_samples": 20, "n_fake": 10, "n_real": 10, "n_groups": 20},
        "metrics": {"auc": {"estimate": 0.7, "low": 0.6, "high": 0.8}, "specificity": {"estimate": 0.8, "low": 0.7, "high": 0.9}}}}}}}
    p = root / "post.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


ENTRIES_T = [
    {"id": "s", "section": "recommended", "modality": "face_video", "task": "T-V", "title": "t", "setting": "x", "why": "w", "caveats": ["c"],
     "source": {"kind": "stored_sealed", "file": "rep.json", "modality": "visual", "dataset": "dsV", "split": "test_x"}},
    {"id": "p", "section": "limitation", "modality": "speech", "task": "T-A", "title": "t", "setting": "x", "why": "w", "caveats": [],
     "lab": {"dataset": "dsP", "group": "G1"},
     "source": {"kind": "stored_post_freeze", "file": "post.json", "dataset": "dsP", "group": "G1"}},
    {"id": "missing", "section": "other", "modality": "image", "task": "T-I", "title": "t", "setting": "x", "why": "w", "caveats": [],
     "source": {"kind": "stored_sealed", "file": "nope.json", "modality": "image", "dataset": "d", "split": "test_x"}},
]


def test_listing_reads_files_and_derives_exactly(tmp_path):
    rep, post = write_sealed(tmp_path), write_post(tmp_path)
    out = {e["id"]: e for e in Showcase(tmp_path, ENTRIES_T).listing()["entries"]}
    s = out["s"]
    assert s["available"] and s["source_sha256"] == file_sha256(rep) and s["kind"] == "stored_sealed"
    assert (s["n_real"], s["n_fake"], s["n_samples"], s["n_groups"]) == (40, 60, 100, 20)
    assert s["metrics"]["auc"] == {"estimate": pytest.approx(0.85), "low": 0.1, "high": 0.9}
    assert s["seed0_auc"] == 0.90 and s["n_seeds"] == 3
    spec = [2 * 0.80 - 0.70, 2 * 0.70 - 0.60, 2 * 0.75 - 0.65]
    assert s["metrics"]["specificity"]["estimate"] == pytest.approx(sum(spec) / 3)
    assert s["metrics"]["fpr"]["estimate"] == pytest.approx(1 - sum(spec) / 3)
    f1 = [2 * p * r / (p + r) for p, r in ((0.9, 0.7), (0.8, 0.6), (0.85, 0.65))]
    assert s["metrics"]["f1"]["estimate"] == pytest.approx(sum(f1) / 3)
    acc = [(r * 60 + sp * 40) / 100 for r, sp in zip((0.7, 0.6, 0.65), spec)]
    assert s["metrics"]["accuracy"]["estimate"] == pytest.approx(sum(acc) / 3)
    for k in ("specificity", "fpr", "f1", "accuracy"):
        assert s["metrics"][k]["derived"] is True and s["metrics"][k]["low"] is None
    assert s["threshold"]["values"] == {"X-s0": 0.6, "X-s1": 0.55, "X-s2": 0.5}
    assert "c" in s["caveats"] and any("seed 0" in c for c in s["caveats"])
    p = out["p"]
    assert p["available"] and p["source_sha256"] == file_sha256(post) and p["lab"] == {"dataset": "dsP", "group": "G1"}
    assert p["metrics"]["auc"]["estimate"] == 0.7 and p["metrics"]["specificity"]["low"] == 0.7 and p["metrics"]["f1"] is None
    assert p["lab_run_id"] == "run-1" and p["confusion"]["tp"] == 5 and p["seed0_auc"] is None
    m = out["missing"]
    assert m["available"] is False and "nope.json" in m["reason"] and "metrics" not in m


def test_endpoint_is_read_only(tmp_path):
    write_sealed(tmp_path)
    write_post(tmp_path)
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    lab = EvalLab(root=tmp_path, synchronous=True)
    lab.showcase.entries = ENTRIES_T
    app = Flask(__name__)
    register(app, lab)
    r = app.test_client().get("/api/evallab/recommended")
    assert r.status_code == 200
    j = r.get_json()
    assert [e["id"] for e in j["entries"]] == ["s", "p", "missing"] and j["not_shown"] and "approval" in j["status"]
    assert lab.store.list() == []
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


def test_entries_are_well_formed():
    ids = [e["id"] for e in ENTRIES]
    assert len(ids) == len(set(ids))
    for e in ENTRIES:
        assert e["section"] in ("recommended", "other", "limitation")
        assert e["modality"] in ("face_video", "speech", "image")
        src = e["source"]
        assert src["kind"] in ("stored_sealed", "stored_post_freeze")
        assert "fused" not in json.dumps(e).lower() or e["section"] != "recommended"
        if src["kind"] == "stored_sealed":
            assert src["file"] == "data/_final/eval/I-4.json"
            # a sealed block links only to its declared re-analysis group of the same block (D-096), never to a new scoring run
            assert "lab" not in e or (e["lab"] == {"dataset": src["dataset"], "group": "G4"} and src["split"] == "test_x")
        else:
            assert e["lab"] == {"dataset": src["dataset"], "group": src["group"]}
        assert chr(0x2014) not in json.dumps(e)


@pytest.mark.skipif(not (REPO / "data/_final/eval/I-4.json").is_file(), reason="logged result files are not on this computer")
def test_real_entries_match_the_logged_files():
    """Every displayed AUROC and count equals the logged file; recommended entries have both classes and enough groups."""
    out = {e["id"]: e for e in Showcase().listing()["entries"]}
    rep = json.loads((REPO / "data/_final/eval/I-4.json").read_text(encoding="utf-8"))
    for e in ENTRIES:
        got = out[e["id"]]
        assert got["available"], (e["id"], got.get("reason"))
        src = e["source"]
        if src["kind"] == "stored_sealed":
            cand = next(c for c in rep["candidates"] if c["modality"] == src["modality"])
            blk = next(b for b in cand["blocks"] if b["dataset"] == src["dataset"] and b["split"] == src["split"])
            assert got["metrics"]["auc"]["estimate"] == blk["metrics"]["auc"]["estimate"]
            assert (got["n_real"], got["n_fake"]) == (blk["n_real"], blk["n_fake"])
        else:
            res = json.loads((REPO / src["file"]).read_text(encoding="utf-8"))
            g = res["datasets"][src["dataset"]]["groups"][src["group"]]
            assert got["metrics"]["auc"]["estimate"] == g["metrics"]["auc"]["estimate"]
            assert (got["n_real"], got["n_fake"]) == (g["data"]["n_real"], g["data"]["n_fake"])
        if e["section"] == "recommended":
            assert got["n_real"] > 0 and got["n_fake"] > 0 and got["n_groups"] >= 16
