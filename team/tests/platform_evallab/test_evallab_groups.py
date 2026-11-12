"""Evaluation groups (D-094): dataset -> group selection, guided runs, refusals and protection of sealed splits."""

import json

import numpy as np
import pytest
from evallab_helpers import block, log_entry, make_lab, synth, write_rows
from flask import Flask

from dfd.eval.scorefile import file_sha256
from dfd.platform.evallab import register
from dfd.platform.evallab.common import POST_FREEZE_STATUS

THR = 0.5


def setup_post(tmp_path, *, n_fake=40, n_real=40, split="post_x", groups=None, dataset="dsP", sha_override=None,
               with_full=True, name="E-900"):
    """A post-freeze score file, its groups file and its descriptor under <root>/data/_evallab/imports/."""
    rng = np.random.default_rng(5)
    f, r = synth(rng, n_fake, n_real)
    imports = tmp_path / "data" / "_evallab" / "imports"
    imports.mkdir(parents=True, exist_ok=True)
    rows = block(split, dataset, f, r, run=f"{name}-X", it=name, gprefix="p", sprefix="p")
    write_rows(imports / f"{name}.csv", rows)
    ids = [x["sample_id"] for x in rows]
    fakes = [x["sample_id"] for x in rows if x["label"] == 1]
    reals = [x["sample_id"] for x in rows if x["label"] == 0]
    if groups is None:
        groups = [
            {"id": "G1", "title": "Small sample", "kind": "general", "description": "small", "overlap": "nested",
             "group_unit": "sample", "n_groups": 20, "sample_ids": fakes[:10] + reals[:10]},
            {"id": "G2", "title": "Medium sample", "kind": "general", "description": "medium", "overlap": "nested",
             "group_unit": "sample", "n_groups": 60, "sample_ids": fakes[:30] + reals[:30] + ["p_missing"]},
            {"id": "G5", "title": "Diagnostic", "kind": "diagnostic", "description": "diag", "overlap": "subset",
             "group_unit": "sample", "n_groups": 20, "sample_ids": fakes[30:] + reals[30:]},
        ]
    doc = {"experiment": name, "datasets": [{"dataset": dataset, "split": split, "groups": groups,
                                             "full": ({"id": "FULL", "title": "Full eligible pool", "kind": "full",
                                                       "description": "all", "overlap": "all", "group_unit": "sample",
                                                       "n_groups": len(ids), "sample_ids": ids} if with_full else None)}]}
    gpath = tmp_path / "data" / "_manifests" / "posteval" / f"{name}.groups.json"
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text(json.dumps(doc), encoding="utf-8")
    desc = {"schema": "evallab-groups/1", "dataset": dataset, "title": f"{dataset} title", "experiment": name,
            "iteration": name, "modality": "visual", "head": "T-X", "split": split, "run_ids": [f"{name}-X"],
            "score_file": f"{name}.csv", "groups_file": f"data/_manifests/posteval/{name}.groups.json",
            "groups_sha256": sha_override or file_sha256(gpath),
            "threshold": {"value": THR, "provenance": "fixed in the checkpoint (test)"},
            "score_kind": "ranking score", "eligibility": "eligible (test)", "caveats": ["test caveat"],
            "provenance": {"experiment": name}}
    (imports / f"{name}.lab.json").write_text(json.dumps(desc), encoding="utf-8")
    return imports, gpath, rows


def lab_and_client(tmp_path):
    lab = make_lab(tmp_path, dirs=["data/_evallab/imports"])
    app = Flask(__name__)
    register(app, lab)
    return lab, app.test_client()


def test_groups_listing_filters_by_dataset_with_counts(tmp_path):
    setup_post(tmp_path)
    setup_post(tmp_path, dataset="dsQ", name="E-901")
    lab, c = lab_and_client(tmp_path)
    body = c.get("/api/evallab/groups").get_json()
    keys = sorted(d["dataset"] for d in body["datasets"])
    assert keys == ["dsP", "dsQ"]
    d = next(d for d in body["datasets"] if d["dataset"] == "dsP")
    assert [g["id"] for g in d["groups"]] == ["G1", "G2", "G5"]
    g1, g2 = d["groups"][0], d["groups"][1]
    assert (g1["n_samples"], g1["n_fake"], g1["n_real"]) == (20, 10, 10)
    assert g2["n_declared"] == 61 and g2["n_missing_scores"] == 1 and g2["n_samples"] == 60
    assert d["full"]["id"] == "FULL" and d["full"]["n_samples"] == 80 and "warning" in d["full"]
    assert d["status"] == POST_FREEZE_STATUS and "not an approval" in d["status"]
    assert "sample_ids" not in json.dumps(body)             # the listing never ships sample ids


def test_group_run_restricts_rows_and_uses_descriptor_threshold(tmp_path):
    setup_post(tmp_path)
    lab, c = lab_and_client(tmp_path)
    r = c.post("/api/evallab/group-runs", json={"dataset": "dsP", "group": "G1", "n_boot": 200})
    assert r.status_code == 202
    rec = lab.store.get(r.get_json()["id"])
    assert rec["state"] == "done", rec.get("error")
    assert rec["data"]["n_samples"] == 20 and rec["data"]["n_fake"] == 10
    assert rec["threshold"]["value"] == THR and rec["threshold"]["source"] == "fixed"
    assert rec["config"]["purpose"] == "post_freeze_evaluation" and rec["config"]["eval_group"] == "G1"
    assert rec["eval_group"]["title"] == "Small sample" and rec["eval_group"]["groups_sha256"]
    assert any("not an approval" in w for w in rec["data"]["warnings"])
    for k in ("auc", "f1", "precision", "recall", "specificity", "accuracy"):
        assert rec["metrics"][k]["estimate"] is not None


def test_group_run_counts_missing_scores(tmp_path):
    setup_post(tmp_path)
    lab, c = lab_and_client(tmp_path)
    rec = lab.store.get(c.post("/api/evallab/group-runs", json={"dataset": "dsP", "group": "G2", "n_boot": 200}).get_json()["id"])
    assert rec["data"]["n_samples"] == 60 and rec["data"]["group_missing_scores"] == 1
    assert any("no stored score" in w for w in rec["data"]["warnings"])


def test_full_pool_needs_explicit_confirmation(tmp_path):
    setup_post(tmp_path)
    lab, c = lab_and_client(tmp_path)
    r = c.post("/api/evallab/group-runs", json={"dataset": "dsP", "group": "FULL"})
    assert r.status_code == 400 and r.get_json()["error"]["code"] == "FULL_NOT_CONFIRMED"
    r = c.post("/api/evallab/group-runs", json={"dataset": "dsP", "group": "FULL", "confirm_full": True, "n_boot": 200})
    assert r.status_code == 202
    assert lab.store.get(r.get_json()["id"])["data"]["n_samples"] == 80


@pytest.mark.parametrize("body, status, code", [
    ({"dataset": "nope", "group": "G1"}, 404, "UNKNOWN_DATASET"),
    ({"dataset": "dsP", "group": "G9"}, 404, "UNKNOWN_GROUP"),
    ({"dataset": "dsP"}, 400, "INVALID_REQUEST"),
    ({"dataset": "dsP", "group": "G1", "threshold": 0.1}, 400, "INVALID_REQUEST"),   # the client cannot set a threshold
    ({"dataset": "dsP", "group": "G1", "n_boot": 5}, 400, "INVALID_CONFIG"),
])
def test_invalid_group_requests(tmp_path, body, status, code):
    setup_post(tmp_path)
    _, c = lab_and_client(tmp_path)
    r = c.post("/api/evallab/group-runs", json=body)
    assert r.status_code == status and r.get_json()["error"]["code"] == code


def test_changed_groups_file_is_refused(tmp_path):
    setup_post(tmp_path, sha_override="0" * 64)
    _, c = lab_and_client(tmp_path)
    body = c.get("/api/evallab/groups").get_json()
    assert body["datasets"] == [] and "groups file changed" in body["skipped"][0]["reason"]


def test_groups_file_changed_after_freeze_is_refused(tmp_path):
    imports, gpath, _ = setup_post(tmp_path)
    lab = make_lab(tmp_path, dirs=["data/_evallab/imports"], sync=True)
    ent, _ = lab.groups.resolve("dsP", "G1")
    cfg = {"score_file_id": ent["score_file_id"], "iteration": "E-900", "modality": "visual", "dataset": "dsP",
           "split": "post_x", "threshold_source": "fixed", "threshold": THR, "threshold_provenance": "t",
           "purpose": "post_freeze_evaluation", "eval_group": "G1", "n_boot": 200}
    cfg_frozen = dict(cfg)
    rec = lab.runner.submit(cfg)
    assert rec["state"] == "done"
    # same config but with a recorded hash that no longer matches the registry
    from dfd.platform.evallab import analysis
    rec2 = dict(rec)
    rec2["config"] = {**rec["config"], "eval_group_sha256": "f" * 64}
    with pytest.raises(analysis.Refused) as e:
        lab.runner._group_ids(rec2["config"])
    assert e.value.code == "GROUPS_CHANGED"
    assert cfg_frozen["eval_group"] == "G1"


@pytest.mark.parametrize("split", ["test_x", "test_in"])
def test_descriptor_for_sealed_split_is_refused(tmp_path, split):
    setup_post(tmp_path, split=split)
    _, c = lab_and_client(tmp_path)
    body = c.get("/api/evallab/groups").get_json()
    assert body["datasets"] == []
    assert "never for a sealed split" in body["skipped"][0]["reason"]
    r = c.post("/api/evallab/group-runs", json={"dataset": "dsP", "group": "G1"})
    assert r.status_code == 404


def test_eval_group_on_sealed_split_refused_in_config(tmp_path, scores_dir, ledger):
    rng = np.random.default_rng(1)
    fv, rv = synth(rng, 10, 10)
    ft, rt = synth(rng, 10, 10)
    p = write_rows(scores_dir / "f.csv", block("val", "dsA", fv, rv) + block("test_x", "dsT", ft, rt))
    log_entry(ledger, iteration="I-0", dataset="dsT", split="test_x", sha=file_sha256(p))
    lab = make_lab(tmp_path, ledger_path=ledger)
    entry = lab.catalogue.scan()["score_files"][0]
    from dfd.platform.evallab.common import LabError
    with pytest.raises(LabError) as e:
        lab.runner.submit({"score_file_id": entry["id"], "iteration": "I-0", "modality": "visual", "dataset": "dsT",
                           "split": "test_x", "purpose": "final_reanalysis", "eval_group": "G1", "n_boot": 200})
    assert e.value.code == "GROUP_ON_FINAL"


def test_post_split_requires_post_freeze_purpose(tmp_path):
    setup_post(tmp_path)
    lab, c = lab_and_client(tmp_path)
    ent, _ = lab.groups.resolve("dsP", "G1")
    r = c.post("/api/evallab/runs", json={"score_file_id": ent["score_file_id"], "iteration": "E-900", "modality": "visual",
                                         "dataset": "dsP", "split": "post_x", "threshold_source": "fixed", "threshold": THR,
                                         "threshold_provenance": "t", "purpose": "development"})
    assert r.status_code == 400 and "post_freeze_evaluation" in r.get_json()["error"]["message"]


def test_existing_advanced_run_without_group_unchanged(tmp_path, scores_dir):
    rng = np.random.default_rng(2)
    fv, rv = synth(rng, 10, 10)
    fd, rd = synth(rng, 10, 10)
    write_rows(scores_dir / "f.csv", block("val", "dsA", fv, rv) + block("dev_x", "dsB", fd, rd))
    lab = make_lab(tmp_path)
    entry = lab.catalogue.scan()["score_files"][0]
    rec = lab.runner.submit({"score_file_id": entry["id"], "iteration": "I-0", "modality": "visual", "dataset": "dsB",
                             "split": "dev_x", "n_boot": 200})
    assert rec["state"] == "done" and rec["eval_group"] is None and rec["config"]["eval_group"] is None


# ---------------------------------------------------------------- sealed re-analysis groups (D-096)

def setup_sealed(tmp_path, ledger, *, flag=True, logged=True):
    """A logged final score file (val + test_x rows), a groups file on its test_x block and a sealed_reanalysis descriptor."""
    rng = np.random.default_rng(7)
    fv, rv = synth(rng, 30, 30)
    ft, rt = synth(rng, 40, 40)
    scores = tmp_path / "data" / "_final" / "scores"
    scores.mkdir(parents=True, exist_ok=True)
    rows = block("val", "dsV", fv, rv) + block("test_x", "dsS", ft, rt, sprefix="s", gprefix="s")
    p = write_rows(scores / "I-0.csv", rows)
    if logged:
        log_entry(ledger, iteration="I-0", dataset="dsS", split="test_x", sha=file_sha256(p))
    test = [r for r in rows if r["split"] == "test_x"]
    fakes = [r["sample_id"] for r in test if r["label"] == 1]
    reals = [r["sample_id"] for r in test if r["label"] == 0]
    groups = [{"id": "G1", "title": "Small sample", "kind": "general", "description": "small", "overlap": "nested",
               "group_unit": "sample", "n_groups": 20, "sample_ids": fakes[:10] + reals[:10]},
              {"id": "G4", "title": "Whole sealed block", "kind": "general", "description": "all", "overlap": "all",
               "group_unit": "sample", "n_groups": 80, "sample_ids": fakes + reals}]
    gpath = tmp_path / "data" / "_manifests" / "sealed_reanalysis" / "g.groups.json"
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text(json.dumps({"datasets": [{"dataset": "dsS", "split": "test_x", "groups": groups, "full": None}]}), encoding="utf-8")
    imports = tmp_path / "data" / "_evallab" / "imports" / "sealed"
    imports.mkdir(parents=True, exist_ok=True)
    desc = {"schema": "evallab-groups/1", "dataset": "dsS", "title": "sealed", "experiment": "E-014", "iteration": "I-0",
            "modality": "visual", "head": "T-X", "split": "test_x", "run_ids": ["R-s0"], "score_file": "data/_final/scores/I-0.csv",
            "groups_file": "data/_manifests/sealed_reanalysis/g.groups.json", "groups_sha256": file_sha256(gpath),
            "threshold": {"source": "val_fit", "value": None, "provenance": "per-seed val fit (test)"}}
    if flag:
        desc["sealed_reanalysis"] = True
    (imports / "E-014_TX.lab.json").write_text(json.dumps(desc), encoding="utf-8")
    lab = make_lab(tmp_path, dirs=["data/_final/scores"], ledger_path=ledger)
    app = Flask(__name__)
    register(app, lab)
    return lab, app.test_client()


def test_sealed_reanalysis_group_runs_through_the_ledger_guard(tmp_path, ledger):
    from dfd.platform.evallab.common import SEALED_REANALYSIS_STATUS
    lab, c = setup_sealed(tmp_path, ledger)
    body = c.get("/api/evallab/groups").get_json()
    d = next(d for d in body["datasets"] if d["dataset"] == "dsS")
    assert d["status"] == SEALED_REANALYSIS_STATUS and d["sealed_reanalysis"] is True
    rec = c.post("/api/evallab/group-runs", json={"dataset": "dsS", "group": "G1", "n_boot": 200}).get_json()
    assert rec["state"] == "done", rec.get("error")
    assert rec["config"]["purpose"] == "final_reanalysis" and rec["config"]["threshold_source"] == "val_fit"
    assert rec["final_guard"]["required"] is True and rec["final_guard"]["ledger_entries"]
    assert rec["data"]["n_samples"] == 20 and rec["threshold"]["fitted_on_these_rows"] is False
    whole = c.post("/api/evallab/group-runs", json={"dataset": "dsS", "group": "G4", "n_boot": 200}).get_json()
    assert whole["data"]["n_samples"] == 80


def test_sealed_reanalysis_without_ledger_entry_is_refused(tmp_path, ledger):
    _, c = setup_sealed(tmp_path, ledger, logged=False)
    rec = c.post("/api/evallab/group-runs", json={"dataset": "dsS", "group": "G1", "n_boot": 200}).get_json()
    assert rec["state"] == "refused" and rec["error"]["code"] == "FINAL_GUARD"


def test_sealed_descriptor_without_declaration_is_still_refused(tmp_path, ledger):
    _, c = setup_sealed(tmp_path, ledger, flag=False)
    body = c.get("/api/evallab/groups").get_json()
    assert body["datasets"] == [] and "never for a sealed split" in body["skipped"][0]["reason"]


def test_two_datasets_in_one_score_file_are_counted_separately(tmp_path):
    """Regression: the per-file count cache must be keyed by dataset too (three sealed blocks share I-4.csv)."""
    setup_post(tmp_path, dataset="dsP", name="E-900")
    imports = tmp_path / "data" / "_evallab" / "imports"
    rng = np.random.default_rng(9)
    f, r = synth(rng, 5, 7)
    extra = block("post_x", "dsQ", f, r, run="E-900-X", it="E-900", gprefix="q", sprefix="q")
    with open(imports / "E-900.csv", "a", encoding="utf-8", newline="") as fh:
        for x in extra:
            fh.write(",".join(str(x[c]) for c in ("run_id", "iteration", "seed", "modality", "dataset", "split", "sample_id",
                                                  "group_id", "label", "score")) + "\n")
    gdoc = {"datasets": [{"dataset": "dsQ", "split": "post_x", "full": None,
                          "groups": [{"id": "G1", "title": "all", "kind": "general", "description": "all", "overlap": "all",
                                      "group_unit": "sample", "n_groups": 12, "sample_ids": [x["sample_id"] for x in extra]}]}]}
    gq = tmp_path / "data" / "_manifests" / "posteval" / "q.groups.json"
    gq.write_text(json.dumps(gdoc), encoding="utf-8")
    desc = json.loads((imports / "E-900.lab.json").read_text(encoding="utf-8"))
    desc.update(dataset="dsQ", groups_file="data/_manifests/posteval/q.groups.json", groups_sha256=file_sha256(gq))
    (imports / "Q.lab.json").write_text(json.dumps(desc), encoding="utf-8")
    _, c = lab_and_client(tmp_path)
    body = c.get("/api/evallab/groups").get_json()
    by = {d["dataset"]: d for d in body["datasets"]}
    assert by["dsP"]["groups"][0]["n_samples"] == 20
    assert (by["dsQ"]["groups"][0]["n_samples"], by["dsQ"]["groups"][0]["n_fake"]) == (12, 5)
