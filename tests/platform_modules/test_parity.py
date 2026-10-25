"""Parity of the real wrapped modules with the stored scores of data/_results/freeze/parity.json (slow; RUN_PARITY=1).

Only cases whose sample id resolves in a train or val split are run: dfd.data.loaders.select_records refuses the test splits unless final=True,
and it is never called with final=True here. The cases of parity.json were built by scripts/freeze/parity_check.py from FF++ val, ASVspoof 2019 LA
development (val) and ProGAN val/train images, so none is from a sealed split.
"""
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.slow
KEY = {"T-V": "face_video", "T-I": "image_detector", "T-A": "speech"}
SPLIT_SOURCES = [  # manifest, split csv, splits that may be read
    ("ffpp.manifest.csv", "ffpp-037e2976fade.csv", ("train", "val")),
    ("asvspoof2019_la.manifest.csv", "asvspoof2019_la-875bf337c379.csv", ("train", "val")),
    ("forensynths_progan_val.manifest.csv", "progan_val-b42dc9619141.csv", ("train", "val")),
]


@pytest.mark.skipif(os.environ.get("RUN_PARITY") != "1", reason="set RUN_PARITY=1 to run the real detectors")
def test_real_modules_match_stored_scores():
    from dfd.data import loaders as L
    from dfd.infer.library import inspect
    from dfd.platform import schema
    from dfd.platform.coordinator import Coordinator
    from dfd.platform.module import probe_from_media
    from dfd.platform.modules import AdapterProvider, build_default_modules

    par = json.loads((ROOT / "data/_results/freeze/parity.json").read_text())
    non_test = {}
    for man, spl, splits in SPLIT_SOURCES:
        for sp in splits:
            try:
                for r in L.select_records(ROOT / "data/_manifests" / man, ROOT / "data/_splits" / spl, sp):
                    non_test[r.sample_id] = (r, sp)
            except L.LoaderError:
                pass
    coord = Coordinator(build_default_modules(AdapterProvider(), timeout_s=900))
    rows, skipped = [], []
    for c in par["cases"]:
        hit = non_test.get(c["sample_id"])
        path = ROOT / "data" / hit[0].path if hit else None
        if hit is None or not path.is_file():
            skipped.append({"sample_id": c["sample_id"], "reason": "not in a train or val split, or not on disk"})
            continue
        rep = coord.analyse(probe_from_media(inspect(path, path.name)), "parity")
        assert schema.validate_report(rep) == []
        m = {x["module_id"]: x for x in rep["modules"]}[KEY[c["task"]]]
        got = m["score"]["value"] if m["status"] == "completed" else None
        diff = None if got is None else abs(got - c["stored_score"])
        rows.append({"task": c["task"], "sample_id": c["sample_id"], "split": hit[1], "stored_score": c["stored_score"], "module_score": got,
                     "abs_diff": None if diff is None else round(diff, 6), "module_status": m["status"], "finding": m["finding"],
                     "passed": diff is not None and diff <= par["tolerance"]})
    out = ROOT / "data/_results/platform"
    out.mkdir(parents=True, exist_ok=True)
    (out / "module_parity.json").write_text(json.dumps({"checkpoint": par["checkpoint"], "tolerance": par["tolerance"], "cases": rows, "skipped": skipped,
                                                        "passed": bool(rows) and all(r["passed"] for r in rows)}, indent=1))
    assert rows, "no case could be run"
    assert all(r["passed"] for r in rows), rows
