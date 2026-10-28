"""Generated-video module: feature geometry, head round trip, applicability, result contract, real DINOv2 path on a synthetic clip.

The heads used here are TEST heads built in the test (clearly named); they say nothing about detector quality."""
import hashlib
import json
import shutil
import subprocess

import numpy as np
import pytest

from dfd.platform.coordinator import Coordinator
from dfd.platform.genvideo import canonical
from dfd.platform.genvideo.features import FEATURE_NAMES, PICK, DinoRunner, trajectory_features
from dfd.platform.genvideo.model import load_head, save_head
from dfd.platform.module import Context, Probe
from dfd.platform.modules.generated_video import GeneratedVideoModule
from dfd.platform.schema import validate_module_result, validate_report


def make_head(path, threshold=0.5, w_first=1.0, **meta):
    mean, std = np.zeros(21), np.ones(21)
    W0 = np.zeros((21, 4)); W0[0, 0] = w_first
    W1 = np.zeros((4, 3)); W1[0, 0] = 1.0
    W2 = np.zeros((3, 1)); W2[0, 0] = 1.0
    save_head(path, mean, std, [W0, W1, W2], [np.zeros(4), np.zeros(3), np.zeros(1)], threshold,
              {"head_id": "TEST-HEAD", "evaluation": "not evaluated", "tested_scope": "test head", "limitations": ["test head"],
               "protocol": canonical.PROTOCOL_ID, "validated": True, **meta})
    return path


def make_cls_head(path, **meta):
    """TEST logistic head on the 384-D mean CLS feature (the shape of the installed E-020 C2 head)."""
    w = np.zeros((384, 1)); w[0, 0] = 1.0
    save_head(path, np.zeros(384), np.ones(384), [w], [np.zeros(1)], 0.5,
              {"head_id": "TEST-CLS", "feature": "cls_mean", "protocol": canonical.PROTOCOL_ID, "evaluation": "test", "tested_scope": "test",
               "limitations": ["test"], **meta})
    return path


def probe(**kw):
    d = dict(path="x.mp4", filename="x.mp4", media_type="video", mime_type="video/mp4", size_bytes=1, sha256="0" * 64, duration_s=5.0,
             has_video=True, has_audio=False)
    d.update(kw)
    return Probe(**d)


class FakeRunner:
    revision = "fake-rev"

    def __init__(self, d0=10.0):
        self.d0 = d0

    def features(self, frames):
        f = np.zeros(21, dtype=np.float32)
        f[0] = self.d0
        return {"restrav21": f}


def approve(path):
    """Test-only approval table for a TEST head. The production table (APPROVED_HEADS) is empty."""
    return {hashlib.sha256(open(path, "rb").read()).hexdigest(): "TEST-VALIDATION"}


def module(tmp_path, d0=10.0, threshold=0.6):
    head = make_head(tmp_path / "head.npz", threshold)
    return GeneratedVideoModule(head, FakeRunner(d0), approved_heads=approve(head), canonicalise=lambda s, d: None, read_frames=lambda p: np.zeros((14, 224, 224, 3), np.uint8))


def test_trajectory_geometry():
    straight = np.outer(np.arange(8), np.ones(5))
    f = trajectory_features(straight)
    assert len(f) == 21 == len(FEATURE_NAMES)
    assert np.allclose(f[:7], np.sqrt(5)) and np.allclose(f[7:13], 0.0, atol=1e-3)
    turn = np.zeros((8, 2))
    for i in range(1, 8):
        turn[i] = turn[i - 1] + ([1, 0] if i % 2 else [0, 1])
    assert np.allclose(trajectory_features(turn)[7:13], 90.0, atol=1e-3)
    assert list(PICK) == [0, 2, 4, 6, 7, 9, 11, 13]


def test_head_round_trip(tmp_path):
    h = load_head(make_head(tmp_path / "h.npz"))
    assert h.meta["head_id"] == "TEST-HEAD"
    x = np.zeros((2, 21)); x[1, 0] = 5
    s = h.score(x)
    assert 0 < s[0] < s[1] < 1


def test_applicability(tmp_path):
    m = module(tmp_path)
    assert m.applicability(probe()).kind == "applicable"
    assert m.applicability(probe(media_type="image", has_video=True)).kind == "not_applicable"
    assert m.applicability(probe(media_type="audio", has_video=False)).kind == "not_applicable"
    short = m.applicability(probe(duration_s=1.0))
    assert short.kind == "not_applicable" and "shorter" in short.reason
    missing = GeneratedVideoModule(tmp_path / "nope.npz")
    a = missing.applicability(probe())
    assert a.kind == "unsupported" and "not found" in a.reason


@pytest.mark.parametrize("d0,finding", [(10.0, "indicates_manipulation"), (0.0, "no_indication_within_tested_scope")])
def test_result_contract_and_threshold(tmp_path, d0, finding):
    m = module(tmp_path, d0)
    r = m.run(Context(probe()))
    assert r.finding == finding and r.source == "real" and r.score["calibrated"] is False and r.units_analysed == 14
    assert validate_module_result(r.to_dict()) == []
    text = repr(r.to_dict()).lower()
    assert "genuine" not in text and "likely real" not in text
    assert any("Face-independent" in x for x in r.limitations) and any("first 2.0 s" in x for x in r.limitations)
    assert r.model["evaluation"] == "not evaluated" and "head=" in r.model["weights_sha256"]


def test_failures_become_failed_entries(tmp_path):
    head = make_head(tmp_path / "head.npz")

    def boom(s, d):
        raise ValueError("canonical re-encode failed: broken")
    m = GeneratedVideoModule(head, FakeRunner(), canonicalise=boom)
    r = Coordinator([m]).analyse(probe(), "t")
    assert validate_report(r) == []
    assert r["modules"][0]["status"] == "failed" and "canonical" in r["modules"][0]["errors"][0]["message"]


def test_missing_head_reported_unsupported_through_coordinator(tmp_path):
    r = Coordinator([GeneratedVideoModule(tmp_path / "none.npz")]).analyse(probe(), "t")
    assert validate_report(r) == []
    assert r["modules"][0]["status"] == "unsupported"


@pytest.fixture(scope="module")
def synthetic_clip(tmp_path_factory):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    p = tmp_path_factory.mktemp("gv") / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=3",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)], check=True)
    return p


def test_canonical_protocol_gives_14_frames_without_metadata(synthetic_clip, tmp_path):
    dst = tmp_path / "c.mp4"
    canonical.canonicalise(synthetic_clip, dst)
    fr = canonical.read_frames(dst)
    assert fr.shape == (14, 224, 224, 3)
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags:stream=codec_name,width,height,r_frame_rate,nb_frames",
                          "-of", "json", str(dst)], capture_output=True, text=True).stdout
    assert '"width": 224' in out and '"codec_name": "h264"' in out and "7/1" in out
    tags = json.loads(out).get("format", {}).get("tags", {})
    assert set(tags) <= {"major_brand", "minor_version", "compatible_brands", "encoder"}      # only the muxer's own constant tags remain
    assert "creation_time" not in tags and not tags.get("title") and not tags.get("comment")


def test_malformed_input_raises_clear_error(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="canonical re-encode failed"):
        canonical.canonicalise(bad, tmp_path / "o.mp4")


def test_real_dinov2_path_on_synthetic_clip(synthetic_clip, tmp_path):
    """Real frozen DINOv2-small, real ffmpeg: the pipeline produces finite 21-D features and a valid result (a plumbing test, not an accuracy test)."""
    pytest.importorskip("transformers")
    try:
        runner = DinoRunner(device="cpu")
        feats = runner.features(canonical.read_frames(_canon(synthetic_clip, tmp_path)))
    except (RuntimeError, StopIteration) as e:
        pytest.skip(f"DINOv2-small weights not available: {e}")
    assert feats["restrav21"].shape == (21,) and np.isfinite(feats["restrav21"]).all()
    assert feats["restrav21"][0] > 0 and feats["cls_all"].shape == (8, 384)
    m = GeneratedVideoModule(make_head(tmp_path / "h.npz"), runner)
    r = m.run(Context(probe(path=str(synthetic_clip))))
    assert validate_module_result(r.to_dict()) == [] and r.model["weights_sha256"].endswith(runner.revision)


def _canon(src, tmp_path):
    dst = tmp_path / "canon.mp4"
    canonical.canonicalise(src, dst)
    return dst


class ClsRunner(FakeRunner):
    def features(self, frames):
        c = np.zeros(384, dtype=np.float32); c[0] = self.d0
        return {"restrav21": np.zeros(21, np.float32), "cls_mean": c}


def _cls_module(tmp_path, d0, approved=True, **meta):
    head = make_cls_head(tmp_path / "c.npz", **meta)
    return GeneratedVideoModule(head, ClsRunner(d0), approved_heads=approve(head) if approved else {}, canonicalise=lambda s, d: None,
                                read_frames=lambda p: np.zeros((14, 224, 224, 3), np.uint8))


def test_head_identity_comes_from_head_meta(tmp_path):
    r = _cls_module(tmp_path, 5.0, validated=True, model_name="TEST appearance head", licence="test licence", score_meaning="test meaning").run(Context(probe()))
    assert r.model["name"] == "TEST appearance head" and r.model["licence"] == "test licence" and r.score["meaning"] == "test meaning"
    assert "ReStraV" not in repr(r.to_dict())
    assert r.finding == "indicates_manipulation" and r.warnings == []


def test_unvalidated_head_is_experimental_never_a_finding(tmp_path):
    for d0 in (5.0, -5.0):                                   # above and below the threshold
        r = _cls_module(tmp_path, d0, validated=False).run(Context(probe()))
        assert r.finding == "inconclusive" and r.score["calibrated"] is False
        assert r.warnings and r.warnings[0]["code"] == "EXPERIMENTAL_UNVALIDATED"
        assert validate_module_result(r.to_dict()) == []


def test_missing_validated_flag_counts_as_experimental(tmp_path):
    r = _cls_module(tmp_path, 5.0).run(Context(probe()))
    assert r.finding == "inconclusive"


def test_incompatible_and_wrong_protocol_heads_are_unsupported(tmp_path):
    wrong_dim = GeneratedVideoModule(make_cls_head(tmp_path / "a.npz", feature="restrav21"), ClsRunner())
    a = wrong_dim.applicability(probe())
    assert a.kind == "unsupported" and "incompatible" in a.reason
    old = GeneratedVideoModule(make_cls_head(tmp_path / "b.npz", protocol="canonical-v1"), ClsRunner())
    b = old.applicability(probe())
    assert b.kind == "unsupported" and "canonical-v1" in b.reason
    unknown = GeneratedVideoModule(make_cls_head(tmp_path / "c.npz", feature="mystery"), ClsRunner())
    assert unknown.applicability(probe()).kind == "unsupported"


def test_corrupt_head_file_is_unsupported_not_a_crash(tmp_path):
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"this is not an npz archive")
    r = Coordinator([GeneratedVideoModule(bad)]).analyse(probe(), "t")
    assert validate_report(r) == [] and r["modules"][0]["status"] == "unsupported"


def test_runner_failure_becomes_failed_entry(tmp_path):
    class Boom(FakeRunner):
        def features(self, frames):
            raise RuntimeError("CUDA out of memory")
    m = GeneratedVideoModule(make_head(tmp_path / "h.npz"), Boom(), canonicalise=lambda s, d: None,
                             read_frames=lambda p: np.zeros((14, 224, 224, 3), np.uint8))
    r = Coordinator([m]).analyse(probe(), "t")
    assert validate_report(r) == [] and r["modules"][0]["status"] == "failed" and "CUDA" in r["modules"][0]["errors"][0]["message"]


def test_installed_head_if_present_is_the_documented_c2():
    """The repository's installed head (git-ignored checkpoint) carries the correct identity and the measured negative DF26 result."""
    from dfd.platform.modules.generated_video import DEFAULT_HEAD
    if not DEFAULT_HEAD.is_file():
        pytest.skip("no installed head in this checkout")
    h = load_head(DEFAULT_HEAD)
    assert h.meta["feature"] == "cls_mean" and h.meta["candidate"] == "C2" and h.meta["validated"] is False
    assert h.meta["protocol"] == canonical.PROTOCOL_ID and "NOT the ReStraV" in h.meta["model_name"]
    assert "DF26" in h.meta["evaluation"] and "chance" in h.meta["tested_scope"] and len(h.W) == 1 and h.mean.shape == (384,)


# ---- validation gate (post-audit): the flag in the file authorises nothing by itself ----------------------------------------------------------

def _is_experimental(r):
    return (r.finding == "inconclusive" and r.score["calibrated"] is False and r.score["threshold"] is None
            and any(w["code"] == "EXPERIMENTAL_UNVALIDATED" or w["code"].startswith("VALIDATION_") for w in r.warnings)
            and validate_module_result(r.to_dict()) == [])


def _run_cls(head, d0, table=None):
    return GeneratedVideoModule(head, ClsRunner(d0), approved_heads=table, canonicalise=lambda s, d: None,
                                read_frames=lambda p: np.zeros((14, 224, 224, 3), np.uint8)).run(Context(probe()))


@pytest.mark.parametrize("d0", [1e6, -1e6, 5.0, -5.0])
def test_unvalidated_head_is_inconclusive_at_any_score(tmp_path, d0):
    assert _is_experimental(_cls_module(tmp_path, d0, validated=False, evaluation="E-020, experimental, NOT validated").run(Context(probe())))


@pytest.mark.parametrize("d0", [1e6, -1e6])
def test_validated_true_without_approved_hash_is_inconclusive(tmp_path, d0):
    r = _cls_module(tmp_path, d0, approved=False, validated=True, evaluation="independent validation E-999").run(Context(probe()))
    assert _is_experimental(r) and any(w["code"] == "VALIDATION_NOT_APPROVED" for w in r.warnings)


def test_default_approval_table_is_empty_so_default_construction_never_gives_a_finding(tmp_path):
    from dfd.platform.modules.generated_video import APPROVED_HEADS
    assert dict(APPROVED_HEADS) == {}
    head = make_cls_head(tmp_path / "d.npz", validated=True, evaluation="independent validation E-999")
    assert _run_cls(head, 1e6).finding == "inconclusive"


def test_editing_metadata_of_an_approved_head_removes_the_approval(tmp_path):
    head = make_cls_head(tmp_path / "e.npz", validated=True, evaluation="independent validation E-999")
    table = approve(head)
    z = np.load(head, allow_pickle=False)
    meta = json.loads(bytes(z["meta_json"]).decode())
    meta["head_id"] = "EDITED"
    arrs = {k: z[k] for k in z.files if k != "meta_json"}
    arrs["meta_json"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8)
    np.savez(head, **arrs)
    r = _run_cls(head, 1e6, table)
    assert _is_experimental(r) and any(w["code"] == "VALIDATION_NOT_APPROVED" for w in r.warnings)


@pytest.mark.parametrize("evaluation", ["E-020, experimental, NOT validated: DF26 AUC 0.4957", "did not generalise to DF26", "", "chance level on DF26"])
def test_approved_hash_but_contradicting_evaluation_text_fails_closed(tmp_path, evaluation):
    r = _cls_module(tmp_path, 1e6, validated=True, evaluation=evaluation).run(Context(probe()))
    assert _is_experimental(r) and any(w["code"] == "VALIDATION_CONTRADICTED" for w in r.warnings)


def test_contradicting_limitations_fail_closed(tmp_path):
    r = _cls_module(tmp_path, 1e6, validated=True, evaluation="independent validation E-999", limitations=["EXPERIMENTAL: do not use"]).run(Context(probe()))
    assert _is_experimental(r)


@pytest.mark.parametrize("flag", ["true", 1, "yes", None, [True]])
def test_non_boolean_validated_values_are_experimental(tmp_path, flag):
    r = _cls_module(tmp_path, 1e6, validated=flag, evaluation="independent validation E-999").run(Context(probe()))
    assert _is_experimental(r)


def test_gate_is_closed_by_default_not_broken(tmp_path):
    """A TEST head with an explicit approval entry and consistent text can give a finding; production has no such entry."""
    r = _cls_module(tmp_path, 1e6, validated=True, evaluation="independent validation E-999").run(Context(probe()))
    assert r.finding == "indicates_manipulation" and r.score["calibrated"] is False and r.warnings == []


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("validated", [True, False])
def test_non_finite_score_never_gives_a_finding(tmp_path, bad, validated):
    r = Coordinator([_cls_module(tmp_path, bad, validated=validated, evaluation="independent validation E-999")]).analyse(probe(), "t")
    m = r["modules"][0]
    assert validate_report(r) == [] and m["status"] != "completed" and m["finding"] is None


def test_non_finite_threshold_with_approval_is_inconclusive(tmp_path):
    head = make_cls_head(tmp_path / "t.npz", validated=True, evaluation="independent validation E-999")
    z = np.load(head, allow_pickle=False)
    arrs = {k: z[k] for k in z.files}
    arrs["threshold"] = np.array([np.nan])
    np.savez(head, **arrs)
    assert _run_cls(head, 5.0, approve(head)).finding == "inconclusive"


def test_missing_or_malformed_meta_is_unsupported(tmp_path):
    p = tmp_path / "m.npz"
    save_head(p, np.zeros(384), np.ones(384), [np.zeros((384, 1))], [np.zeros(1)], 0.5, {})        # empty meta: no protocol
    assert GeneratedVideoModule(p, ClsRunner()).applicability(probe()).kind == "unsupported"
    p2 = tmp_path / "m2.npz"
    np.savez(p2, mean=np.zeros(384), std=np.ones(384), threshold=np.array([0.5]), n_layers=np.array([1]), W0=np.zeros((384, 1)), b0=np.zeros(1),
             meta_json=np.frombuffer(b"{not json", dtype=np.uint8))
    assert GeneratedVideoModule(p2, ClsRunner()).applicability(probe()).kind == "unsupported"


def test_installed_head_stays_inconclusive_through_the_full_module_path():
    """Real installed C2 head (git-ignored; skipped if absent), fake feature runner: any score gives inconclusive, never a finding."""
    from dfd.platform.modules.generated_video import DEFAULT_HEAD
    if not DEFAULT_HEAD.is_file():
        pytest.skip("no installed head in this checkout")
    for d0 in (1e6, -1e6, 0.0):
        m = GeneratedVideoModule(DEFAULT_HEAD, ClsRunner(d0), canonicalise=lambda s, d: None, read_frames=lambda p: np.zeros((14, 224, 224, 3), np.uint8))
        r = Coordinator([m]).analyse(probe(), "t")
        mod = r["modules"][0]
        assert validate_report(r) == [] and mod["status"] == "completed" and mod["finding"] == "inconclusive"
        assert mod["score"]["calibrated"] is False and mod["score"]["threshold"] is None
        assert any(w["code"] == "EXPERIMENTAL_UNVALIDATED" for w in mod["warnings"])
        assert "DF26" in mod["tested_scope"] and "NOT the ReStraV" in mod["model"]["name"]


def test_hash_checked_by_the_gate_is_the_hash_of_the_loaded_bytes(tmp_path):
    """The file is read once: swapping it on disk after loading changes neither the approval nor the reported identity."""
    head = make_cls_head(tmp_path / "s.npz", validated=True, evaluation="independent validation E-999")
    table = approve(head)
    good_sha = next(iter(table))
    m = GeneratedVideoModule(head, ClsRunner(1e6), approved_heads=table, canonicalise=lambda s, d: None,
                             read_frames=lambda p: np.zeros((14, 224, 224, 3), np.uint8))
    assert m.applicability(probe()).kind == "applicable"                       # loads (and hashes) the file
    make_cls_head(head, validated=True, evaluation="independent validation E-999", head_id="SWAPPED")       # different bytes, same path
    r = m.run(Context(probe()))
    assert r.finding == "indicates_manipulation" and f"head={good_sha}" in r.model["weights_sha256"] and r.version == "TEST-CLS"
    # a module that loads the swapped file after the swap is not approved
    fresh = GeneratedVideoModule(head, ClsRunner(1e6), approved_heads=table, canonicalise=lambda s, d: None,
                                 read_frames=lambda p: np.zeros((14, 224, 224, 3), np.uint8))
    assert fresh.run(Context(probe())).finding == "inconclusive"
