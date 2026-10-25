"""Generated-video module: feature geometry, head round trip, applicability, result contract, real DINOv2 path on a synthetic clip.

The heads used here are TEST heads built in the test (clearly named); they say nothing about detector quality."""
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


def module(tmp_path, d0=10.0, threshold=0.6):
    head = make_head(tmp_path / "head.npz", threshold)
    return GeneratedVideoModule(head, FakeRunner(d0), canonicalise=lambda s, d: None, read_frames=lambda p: np.zeros((14, 224, 224, 3), np.uint8))


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


def _cls_module(tmp_path, d0, **meta):
    return GeneratedVideoModule(make_cls_head(tmp_path / "c.npz", **meta), ClsRunner(d0), canonicalise=lambda s, d: None,
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
