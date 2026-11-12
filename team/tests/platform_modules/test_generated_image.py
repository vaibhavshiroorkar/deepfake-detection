"""The experimental whole-image check for still images with a face (UX-7, D8, D-083), with a fake adapter (CPU only, no model is loaded).

It must leave image_detector unchanged, score through the frozen adapter's own whole-image function, never make a finding, never be a
key figure, and not run (not applicable) when the image analysis already used the whole-image path. Every number here is synthetic.
"""
import math

import cv2
import numpy as np
import pytest

from dfd.infer.adapters import AdapterResult, ModalityScore
from dfd.platform import explain, schema
from dfd.platform.coordinator import Coordinator
from dfd.platform.module import Probe
from dfd.platform.modules import AdapterProvider, build_default_modules
from dfd.platform.modules.generated_image import ALREADY_EXAMINED, EXPERIMENTAL_WARNING, GeneratedImageModule

META = {"checkpoint_id": "synthetic-test-checkpoint", "files": {"head_T-I.npz": "2" * 64},
        "tasks": {"T-V": {"threshold": {"value": 0.37}}, "T-I": {"threshold": {"value": 0.61}}, "T-A": {"threshold": {"value": 0.52}}}}


class FakeAdapter:
    inference_source = "real"

    def __init__(self, per, whole=0.73, heads=("T-V", "T-I", "T-A")):
        self.per, self.whole, self.heads, self.meta = per, whole, {h: None for h in heads}, META
        self.analyses, self.whole_calls = 0, []

    def analyse(self, media, progress=None):
        self.analyses += 1
        return AdapterResult(self.per, [], {}, 0.5, {})

    def _score_whole_image(self, task, bgr):
        self.whole_calls.append((task, bgr.shape))
        return self.whole


@pytest.fixture
def png(tmp_path):
    p = tmp_path / "x.png"
    cv2.imwrite(str(p), np.full((48, 64, 3), 127, np.uint8))
    return p


def probe(path, mt="image"):
    return Probe(path=str(path), filename=path.name, media_type=mt, mime_type="image/png", size_bytes=10, sha256="b" * 64,
                 duration_s=None, has_video=True, has_audio=False, width=64, height=48)


def run(adapter, p, with_new=True):
    prov = AdapterProvider(adapter=adapter)
    mods = build_default_modules(prov) + ([GeneratedImageModule(prov)] if with_new else [])
    r = Coordinator(mods).analyse(p, "test")
    assert schema.validate_report(r) == [], schema.validate_report(r)
    assert explain.check_report_explanations(r) == [], explain.check_report_explanations(r)
    return r, {m["module_id"]: m for m in r["modules"]}


FACE = {"image": ModalityScore(True, 0.12, 1, None, "face_crop")}
WHOLE = {"image": ModalityScore(True, 0.9, 1, None, "whole_image")}


def test_face_image_gets_an_experimental_whole_image_score_and_no_finding(png):
    ad = FakeAdapter(FACE, whole=0.7312345678)
    r, m = run(ad, probe(png))
    g = m["generated_image"]
    assert g["status"] == "completed" and g["finding"] == "inconclusive"
    assert g["score"]["value"] == round(0.7312345678, 6) and g["score"]["threshold"] is None and g["score"]["calibrated"] is False
    assert "probability" not in g["score"]["meaning"].replace("not a calibrated probability", "")
    assert g["warnings"][0]["code"] == EXPERIMENTAL_WARNING[0]
    assert ad.whole_calls == [("T-I", (48, 64, 3))] and ad.analyses == 1      # frozen function, whole decoded image, one analysis
    e = g["explanation"]
    assert e["group"] == "experimental" and e["key_figure"] is None and "experimental" in e["headline"].lower()
    assert "makes no finding" in " ".join(s["text"] for s in e["sentences"])
    assert any(o["code"] == "frozen_threshold_not_applied" and o["value"] == 0.61 for o in g["observations"])


def test_image_detector_result_is_unchanged_by_the_new_module(png):
    _, with_new = run(FakeAdapter(FACE), probe(png))
    _, without = run(FakeAdapter(FACE), probe(png), with_new=False)
    a, b = with_new["image_detector"], without["image_detector"]
    for k in ("status", "finding", "score", "tested_scope", "limitations", "warnings", "model", "units_analysed"):
        assert a[k] == b[k], k
    ta = " ".join(s["text"] for s in a["explanation"]["sentences"])
    tb = " ".join(s["text"] for s in b["explanation"]["sentences"])
    assert "separate experimental check on the whole image" in ta and "generated-image check did not run" in tb


def test_whole_image_path_is_not_scored_twice(png):
    ad = FakeAdapter(WHOLE)
    _, m = run(ad, probe(png))
    g = m["generated_image"]
    assert g["status"] == "not_applicable" and g["reason"] == ALREADY_EXAMINED and g["score"] is None
    assert ad.whole_calls == []


def test_not_an_image_and_no_t_i_head(png):
    assert GeneratedImageModule(AdapterProvider(adapter=FakeAdapter(FACE))).applicability(probe(png, "video")).kind == "not_applicable"
    _, m = run(FakeAdapter(FACE, heads=("T-V", "T-A")), probe(png))
    assert m["generated_image"]["status"] == "unsupported"


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_score_fails_closed(png, bad):
    _, m = run(FakeAdapter(FACE, whole=bad), probe(png))
    assert m["generated_image"]["status"] == "failed" and m["generated_image"]["score"] is None
    assert m["image_detector"]["status"] == "completed"


def test_limitations_state_the_measured_portrait_result_and_no_reliability_claim(png):
    """The limitation text quotes the E-028 measurement (most generated faces missed); nothing says the check is reliable or authentic."""
    _, m = run(FakeAdapter(FACE), probe(png))
    g = m["generated_image"]
    text = " ".join(g["limitations"] + [g["model"]["evaluation"], g["warnings"][0]["message"]]).lower()
    assert "most generated faces were missed" in text and "trained on progan images" in text
    for forbidden in ("reliable", "authentic", "proves", "not evaluated by this project"):
        assert forbidden not in text, forbidden
