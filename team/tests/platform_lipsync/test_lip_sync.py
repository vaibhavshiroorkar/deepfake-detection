"""Unit tests of the lip_sync module with synthetic inputs (no real media, no GPU) and a weights load test (skipped if absent)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dfd.platform import schema
from dfd.platform.module import Context, Probe
from dfd.platform.modules import lip_sync as ls
from dfd.platform.syncnet.scoring import SyncResult

WEIGHTS = ls.DEFAULT_WEIGHTS


def probe(media_type="video", has_audio=True, has_video=True, dur=5.0):
    return Probe(path="x.mp4", filename="x.mp4", media_type=media_type, mime_type="video/mp4", size_bytes=1, sha256="0" * 64,
                 duration_s=dur, has_video=has_video, has_audio=has_audio)


def face(x=100, y=80, w=60, h=60):
    return SimpleNamespace(x=x, y=y, w=w, h=h, score=0.9)


def speechlike(seconds=4.0, sr=16000, seed=0):
    """Synthetic stand-in with the features the heuristic looks for: harmonic voice-like source, syllable-rate loudness modulation, noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    x = sum(np.sin(2 * np.pi * 120 * h * t) / h for h in range(1, 12))
    env = np.clip(np.sin(2 * np.pi * 3.5 * t), 0, None) ** 2 + 0.05
    return (0.25 * x * env / np.abs(x * env).max() + 0.004 * rng.standard_normal(len(t))).astype(np.float32)


class Fake(ls.LipSyncModule):
    """Module with every heavy step replaced; weights hash is a constant so no file is needed."""

    def __init__(self, wav=None, faces=None, result=None, **kw):
        super().__init__(weights_path="nonexistent.model", device="cpu", **kw)
        self.wav = speechlike() if wav is None else wav
        self.faces = [[face()] for _ in range(100)] if faces is None else faces
        self.result = result or SyncResult(0, 4.0, 5.0, 80, 15, np.zeros(31))
        self.scored = 0

    def weights_sha256(self):
        return "a" * 64

    def _weights_available(self):
        return True

    def _load_audio(self, path):
        return self.wav

    def _detect_faces(self, path):
        return self.faces

    def _make_crops(self, path, boxes):
        return np.zeros((len(boxes), 224, 224, 3), np.uint8)

    def _score(self, crops, wav):
        self.scored += 1
        return self.result


def run(mod, **kw):
    res = mod.run(Context(probe=probe(**kw)))
    assert schema.validate_module_result(res.to_dict()) == [], res
    return res


def test_applicability_branches():
    m = Fake()
    assert m.applicability(probe()).kind == "applicable"
    assert m.applicability(probe(media_type="image", has_audio=False)).kind == "not_applicable"
    assert m.applicability(probe(media_type="audio", has_video=False)).kind == "not_applicable"
    a = m.applicability(probe(has_audio=False))
    assert a.kind == "not_applicable" and "audio" in a.reason


def test_identity_fields():
    m = Fake()
    assert (m.module_id, m.section, m.source) == ("lip_sync", "cross_modal", "real")
    r = run(m)
    assert r.model["name"] == "SyncNet v2 (Chung and Zisserman, Oxford VGG)"
    assert r.model["licence"] == "code MIT (joonson/syncnet_python); weights licence not stated by the source"
    assert r.model["evaluation"] == "not evaluated"
    assert r.score["calibrated"] is False and r.score["threshold"] is None


def test_consistent_and_mismatch_mapping():
    ok = run(Fake(result=SyncResult(1, 4.0, 6.0, 80, 15, np.zeros(31))))
    assert ok.finding == "consistent" and ok.score["value"] == pytest.approx(0.6)
    bad = run(Fake(result=SyncResult(-7, 5.0, 6.0, 80, 15, np.zeros(31))))
    assert bad.finding == "potential_mismatch"
    assert any(w["code"] == "OFFSET_BEYOND_TOLERANCE" for w in bad.warnings)
    assert {o["code"] for o in bad.observations} >= {"av_offset_frames", "sync_confidence", "sync_min_dist"}


def test_low_confidence_is_inconclusive_with_score_kept():
    r = run(Fake(result=SyncResult(9, 6.0, 0.4, 80, 15, np.zeros(31))))
    assert r.finding == "inconclusive" and r.warnings[0]["code"] == "LOW_SYNC_CONFIDENCE" and r.score is not None


def test_score_clipped_to_unit_interval():
    r = run(Fake(result=SyncResult(0, 1.0, 55.0, 80, 15, np.zeros(31))))
    assert r.score["value"] == 1.0


def test_no_audio_samples_is_not_applicable():
    r = run(Fake(wav=np.zeros(0, np.float32)))
    assert r.status == "not_applicable" and "audio" in r.reason


def test_too_short_is_inconclusive_and_not_scored():
    m = Fake(wav=np.full(16000, 0.2, np.float32))
    r = run(m)
    assert r.finding == "inconclusive" and r.warnings[0]["code"] == "CLIP_TOO_SHORT" and m.scored == 0


def test_silence_is_inconclusive():
    m = Fake(wav=np.zeros(16000 * 4, np.float32))
    r = run(m)
    assert r.finding == "inconclusive" and r.warnings[0]["code"] == "NO_AUDIBLE_AUDIO" and m.scored == 0


def test_no_face_is_not_applicable():
    r = run(Fake(faces=[[] for _ in range(100)]))
    assert r.status == "not_applicable" and "face" in r.reason


def test_several_faces_is_inconclusive_never_picks_one():
    m = Fake(faces=[[face(), face(x=300)] for _ in range(100)])
    r = run(m)
    assert r.finding == "inconclusive" and r.warnings[0]["code"] == "SPEAKER_ATTRIBUTION_NOT_ESTABLISHED" and m.scored == 0
    assert "does not pick one" in r.warnings[0]["message"]


def test_track_interpolates_gaps_and_follows_one_face():
    per = [[face(x=100)] if i % 5 else [] for i in range(50)]
    boxes, info = ls.build_track(per)
    assert boxes.shape == (50, 4) and not np.isnan(boxes).any() and info["reason"] is None
    # rare second face does not hijack the track
    per = [[face(x=100)] for _ in range(50)]
    per[10] = [face(x=400, w=90, h=90), face(x=102)]
    boxes, _ = ls.build_track(per)
    assert abs(boxes[10, 0] - 102) < 1e-6


def test_vocabulary_and_wording():
    results = [run(Fake(result=SyncResult(o, 4.0, c, 80, 15, np.zeros(31)))) for o, c in [(0, 6.0), (8, 6.0), (0, 0.1)]]
    results.append(run(Fake(faces=[[] for _ in range(100)])))
    for r in results:
        assert r.finding in (None, "consistent", "potential_mismatch", "inconclusive")
        blob = str(r.to_dict()).lower()
        assert "indicates_manipulation" not in blob and "deepfake detected" not in blob and chr(0x2014) not in blob
    assert any("not proof of manipulation" in l for l in results[0].limitations)
    assert any("synchronised fake passes" in l for l in results[0].limitations)
    assert any("Tested scope" in l for l in results[0].limitations)


def test_inference_error_propagates():
    class Boom(Fake):
        def _score(self, crops, wav):
            raise RuntimeError("cuda out of memory")
    with pytest.raises(RuntimeError):
        Boom().run(Context(probe=probe()))


def test_mfcc_shapes_and_window_arithmetic():
    from dfd.platform.syncnet.preprocess import audio_mfcc_tensor
    from dfd.platform.syncnet.scoring import scorable_windows
    w = np.random.default_rng(0).normal(0, 0.1, 16000 * 2).astype(np.float32)
    m = audio_mfcc_tensor(w)
    assert m.shape[0] == 13 and m.shape[1] == 1 + int(np.ceil((32000 - 400) / 160))
    assert scorable_windows(50, 32000) == 45 and scorable_windows(60, 32000) == 45


@pytest.mark.skipif(not WEIGHTS.is_file(), reason="syncnet_v2.model not present")
def test_weights_load_strict_and_hash_matches_source_record():
    import json

    import torch

    from dfd.platform.syncnet.model import S, load_weights
    m = load_weights(S(), WEIGHTS)           # strict key matching, raises on any missing or unexpected key
    assert sum(p.numel() for p in m.parameters()) == 13_631_616
    with torch.inference_mode():
        a = m.forward_aud(torch.zeros(2, 1, 13, 20))
        v = m.forward_lip(torch.zeros(2, 3, 5, 224, 224))
    assert a.shape == (2, 1024) and v.shape == (2, 1024)
    rec = json.loads(WEIGHTS.with_name("SOURCE.json").read_text(encoding="utf-8"))
    assert ls.LipSyncModule(WEIGHTS).weights_sha256() == rec["sha256"]


def test_speech_heuristic_accepts_speechlike_and_rejects_steady_and_noise():
    sr = 16000
    t = np.arange(4 * sr) / sr
    rng = np.random.default_rng(1)
    assert ls.looks_speech_like(ls.speech_like_metrics(speechlike()))
    assert not ls.looks_speech_like(ls.speech_like_metrics((0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)))      # steady tone
    assert not ls.looks_speech_like(ls.speech_like_metrics((0.2 * rng.standard_normal(4 * sr)).astype(np.float32)))        # white noise
    assert not ls.looks_speech_like(ls.speech_like_metrics(np.zeros(4 * sr, np.float32)))


def test_non_speech_audio_is_inconclusive_not_scored():
    t = np.arange(16000 * 4) / 16000
    m = Fake(wav=(0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32))
    r = run(m)
    assert r.finding == "inconclusive" and r.warnings[0]["code"] == "AUDIO_NOT_SPEECH_LIKE" and m.scored == 0
    assert "not a speech" in r.warnings[0]["message"]


def test_offset_at_search_boundary_is_inconclusive():
    for off in (15, -15):
        r = run(Fake(result=SyncResult(off, 4.0, 6.0, 80, 15, np.zeros(31))))
        assert r.finding == "inconclusive" and r.warnings[0]["code"] == "OFFSET_AT_SEARCH_BOUNDARY"
    assert run(Fake(result=SyncResult(14, 4.0, 6.0, 80, 15, np.zeros(31)))).finding == "potential_mismatch"


def test_confidence_between_old_and_new_minimum_is_inconclusive():
    r = run(Fake(result=SyncResult(0, 4.0, 2.0, 80, 15, np.zeros(31))))
    assert r.finding == "inconclusive" and r.warnings[0]["code"] == "LOW_SYNC_CONFIDENCE"


def test_offset_band_is_asymmetric_and_old_tolerance_still_available():
    r = lambda off: SyncResult(off, 4.0, 6.0, 80, 15, np.zeros(31))
    assert ls.OFFSET_BAND_FRAMES == (-6, 2) and ls.MODULE_VERSION == "syncnet-v2-port-3"
    for off, want in ((-6, "consistent"), (-3, "consistent"), (2, "consistent"), (3, "potential_mismatch"), (-7, "potential_mismatch")):
        assert run(Fake(result=r(off))).finding == want
    sym = Fake(result=r(-3), offset_band=None)                  # port-2 behaviour: symmetric +-2
    assert sym.offset_band == (-2, 2) and run(sym).finding == "potential_mismatch"
    assert run(Fake(result=r(-3), offset_band=(-2, 2))).finding == "potential_mismatch"


def test_missing_weights_file_is_unsupported_not_a_failure(tmp_path):
    m = ls.LipSyncModule(weights_path=tmp_path / "absent.model", device="cpu")
    a = m.applicability(probe())
    assert a.kind == "unsupported" and "weights" in a.reason
    assert m.applicability(probe(media_type="image", has_audio=False)).kind == "not_applicable"   # media checks still come first


def test_missing_weights_file_through_coordinator_is_unsupported(tmp_path):
    from dfd.platform.coordinator import Coordinator
    r = Coordinator([ls.LipSyncModule(weights_path=tmp_path / "absent.model", device="cpu")]).analyse(probe(), "t")
    assert schema.validate_report(r) == [] and r["modules"][0]["status"] == "unsupported" and r["modules"][0]["finding"] is None
