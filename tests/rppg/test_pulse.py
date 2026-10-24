"""Unit tests of the S-1 pulse pipeline (dfd.rppg.pulse): synthetic traces and a synthetic video, CPU only."""
import numpy as np
import pytest

import cv2

from dfd.rppg import pulse as P


def synth_trace(fps=30.0, seconds=10.0, f_hz=1.2, amp=(0.3, 1.0, 0.6), noise=0.15, seed=1):
    """Skin-like RGB trace: constant colour plus a sinusoid in all channels (strongest in G) plus white noise."""
    rng = np.random.default_rng(seed)
    n = int(round(fps * seconds))
    t = np.arange(n) / fps
    base = np.array([180.0, 130.0, 110.0])
    pulse = np.sin(2 * np.pi * f_hz * t)[:, None] * np.array(amp)[None, :]
    return base + pulse + rng.normal(0.0, noise, (n, 3)), fps


def test_pos_shape_and_projection_matrix():
    rgb, fps = synth_trace()
    H = P.pos(rgb, fps)
    assert H.shape == (len(rgb),) and np.isfinite(H).all()
    # the two projection rows of equation (33) are orthogonal to each other and to [1, 1, 1]
    assert np.allclose(P._P @ P._P.T, np.diag(np.diag(P._P @ P._P.T)))
    assert np.allclose(P._P @ np.ones(3), 0.0)


def test_pos_window_too_short():
    with pytest.raises(ValueError):
        P.pos(np.ones((10, 3)), 30.0)


def test_peak_at_known_frequency_and_snr_above_shuffle():
    rgb, fps = synth_trace(f_hz=1.2)
    r = P.pulse_snr(rgb, fps)
    assert abs(r["peak_hz"] - 1.2) < 0.05
    assert abs(r["bpm"] - 72.0) < 3.0
    rng = np.random.default_rng(0)
    sh = P.shuffled_control(rgb, fps, rng, n_shuffles=20)
    assert sh.shape == (20,) and np.isfinite(sh).all()
    assert r["snr_db"] > np.percentile(sh, 95)
    assert r["snr_db"] > sh.max()


def test_flat_trace_is_low_snr():
    rng = np.random.default_rng(3)
    fps = 30.0
    flat = np.array([180.0, 130.0, 110.0]) + rng.normal(0.0, 0.15, (300, 3))
    sig, _ = synth_trace()
    s_flat = P.pulse_snr(flat, fps)["snr_db"]
    s_sig = P.pulse_snr(sig, fps)["snr_db"]
    assert s_flat < s_sig - 6.0   # at least 6 dB lower than a trace with a pulse
    sh = P.shuffled_control(flat, fps, np.random.default_rng(0), n_shuffles=20)
    # for a trace without a pulse the real SNR lies inside the range of its shuffles
    assert sh.min() - 3.0 < s_flat < sh.max() + 3.0


def test_shuffle_is_reproducible_and_independent():
    rgb, fps = synth_trace()
    a = P.shuffled_control(rgb, fps, np.random.default_rng(5), 10)
    b = P.shuffled_control(rgb, fps, np.random.default_rng(5), 10)
    c = P.shuffled_control(rgb, fps, np.random.default_rng(6), 10)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert len(set(np.round(a, 6))) > 1


def test_second_harmonic_counts_as_signal():
    fps, n = 30.0, 300
    t = np.arange(n) / fps
    x = np.sin(2 * np.pi * 1.0 * t) + 0.6 * np.sin(2 * np.pi * 2.0 * t)
    r1 = P.spectrum_snr(x, fps)
    x1 = np.sin(2 * np.pi * 1.0 * t)
    r2 = P.spectrum_snr(x1 + 0.6 * np.sin(2 * np.pi * 2.6 * t), fps)   # not a harmonic
    assert abs(r1["peak_hz"] - 1.0) < 0.02
    assert r1["snr_db"] > r2["snr_db"]


def test_fill_missing_and_gap():
    rgb = np.tile(np.arange(10.0)[:, None], (1, 3))
    rgb[3:5] = np.nan
    rgb[0] = np.nan
    out, share, gap, nv = P.fill_missing(rgb)
    assert nv == 7 and abs(share - 0.7) < 1e-12 and gap == 2
    assert np.allclose(out[3], 3.0) and np.allclose(out[4], 4.0) and np.allclose(out[0], 1.0)
    allnan = P.fill_missing(np.full((5, 3), np.nan))
    assert allnan[1] == 0.0 and np.isnan(allnan[0]).all()


def test_segment_plan():
    assert P.segment_plan(900, 30.0) == (15, 300)
    assert P.segment_plan(305, 30.0) == (5, 300)      # offset moved earlier so the 10 s fit
    assert P.segment_plan(299, 30.0) is None
    assert P.segment_plan(500, 25.0) == (12, 250) or P.segment_plan(500, 25.0) == (13, 250)
    assert P.segment_plan(0, 30.0) is None


def test_wilson_known_values():
    lo, hi = P.wilson(0, 10)
    assert lo == 0.0 and abs(hi - 0.2775) < 1e-3
    lo, hi = P.wilson(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3
    assert np.isnan(P.wilson(0, 0)[0])


class FakeFace:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h


class FakeDetector:
    def __init__(self, box, drop=()):
        self.box, self.drop, self.i = box, set(drop), 0

    def detect(self, frame):
        i, self.i = self.i, self.i + 1
        return [] if i in self.drop else [FakeFace(*self.box)]


@pytest.fixture(scope="module")
def synth_video(tmp_path_factory):
    """MJPG avi, 320x240, 30 fps, 12 s: skin-coloured square with a 1.2 Hz modulation of the green channel."""
    path = tmp_path_factory.mktemp("rppg") / "synth.avi"
    fps, n = 30, 360
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 240))
    if not vw.isOpened():
        pytest.skip("MJPG writer not available")
    for i in range(n):
        frame = np.full((240, 320, 3), 40, np.uint8)
        g = 150 + 4.0 * np.sin(2 * np.pi * 1.2 * i / fps)
        frame[60:200, 100:240] = (130, int(round(g)), 200)     # BGR: skin-like (R=200, G~150, B=130)
        vw.write(frame)
    vw.release()
    return str(path)


def test_extract_trace_recovers_modulation(synth_video):
    det = FakeDetector((90, 50, 160, 160), drop={10, 11, 12})
    tr = P.extract_trace(synth_video, det, 0, 300)
    assert tr["n_read"] == 300 and abs(tr["fps"] - 30.0) < 1e-6
    assert np.isnan(tr["rgb"][10]).all() and np.isfinite(tr["rgb"][20]).all()
    filled, share, gap, _ = P.fill_missing(tr["rgb"])
    assert abs(share - 297 / 300) < 1e-9 and gap == 3
    assert np.ptp(filled[:, 1]) > 2.5 * np.ptp(filled[:, 0])      # green carries the modulation
    r = P.pulse_snr(filled, 30.0)
    assert abs(r["peak_hz"] - 1.2) < 0.05


def test_analyse_clip_rules(synth_video):
    det = FakeDetector((90, 50, 160, 160))
    tr = P.extract_trace(synth_video, det, 15, 300)
    res = P.analyse_clip(tr, np.random.default_rng(0))
    assert res["usable"] and res["reason"] == "ok" and abs(res["peak_hz"] - 1.2) < 0.05
    assert res["shuffled"].shape == (P.PARAMS.n_shuffles,) and res["snr_db"] > np.nanmax(res["shuffled"])
    # no face at all: unusable with reason face_share, no measurement
    det2 = FakeDetector((90, 50, 160, 160), drop=range(10000))
    res2 = P.analyse_clip(P.extract_trace(synth_video, det2, 15, 300), np.random.default_rng(0))
    assert not res2["usable"] and res2["reason"] == "face_share" and np.isnan(res2["snr_db"])
    # too few frames read
    res3 = P.analyse_clip(dict(rgb=np.ones((100, 3)), fps=30.0, n_read=100, n_total=100, width=1, height=1),
                          np.random.default_rng(0))
    assert not res3["usable"] and res3["reason"] == "short_read"
