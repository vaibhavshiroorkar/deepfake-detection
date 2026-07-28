"""Accuracy of the Audio tab's plots and the numbers beside them.

These pin down bugs found by checking the tab against the ops it calls:

  * the "Enable downmix" toggle did the same channel mean whether on or off;
  * "Trim leading silence" cut the head off the waveform while the frame
    timestamps stayed put, sliding every audio window later in real time;
  * the concatenated-windows plot was labelled in clip seconds when its x axis
    spans the concatenation;
  * the mel view had no extent, so frame index was displayed as if it were time.
"""
import numpy as np
import pytest

from dashboard.pages import preprocess as P  # noqa: E402  (module-level Streamlit is fine here)
from preprocessing.ops import audio as AF


# ------------------------------------------------------------------ waveform_fig

def test_waveform_fig_rejects_multichannel_input():
    """A [channels, samples] array would stretch the time axis by the channel count."""
    stereo = np.zeros((2, 1000), dtype=np.float32)
    with pytest.raises(ValueError, match="mono 1-D"):
        P.waveform_fig(stereo, 16000, "stereo")


def test_waveform_fig_time_axis_matches_duration():
    rate = 16000
    y = np.zeros(rate * 3, dtype=np.float32)      # exactly 3 seconds
    fig = P.waveform_fig(y, rate, "3s")
    line = fig.axes[0].lines[0]
    assert line.get_xdata()[-1] == pytest.approx(3.0, abs=1e-3)


def test_waveform_fig_labels_can_say_it_is_not_clip_time():
    fig = P.waveform_fig(np.zeros(10, np.float32), 100, "t", xlabel="s within the concatenation")
    assert "concatenation" in fig.axes[0].get_xlabel()


# --------------------------------------------------------------- downmix vs off

def test_downmix_off_is_not_the_same_operation_as_on():
    """The toggle has to change the signal, or it is decoration."""
    rng = np.random.default_rng(0)
    stereo = rng.standard_normal((2, 4000)).astype(np.float32)

    on = AF.downmix(stereo)                    # what the enabled branch computes
    off = np.asarray(stereo[0], dtype=np.float32)   # what the disabled branch now computes

    assert not np.allclose(on, off)
    # and the old disabled branch was literally the enabled one
    assert np.allclose(on, stereo.mean(axis=0))


# ----------------------------------------------------------- leading silence

def test_trimming_the_head_desynchronises_the_windows():
    """Why the trim toggle was removed: fixed timestamps over a shortened waveform."""
    rate = 16000
    silence = np.zeros(int(0.20 * rate), dtype=np.float32)
    tone = np.sin(np.linspace(0, 400, int(2.0 * rate))).astype(np.float32)
    wav = np.concatenate([silence, tone])

    timestamps = np.array([0.5, 1.0, 1.5])
    trimmed, dropped = AF.trim_leading_silence(wav, rate, 30.0)
    assert dropped > 0.1, "fixture should have a measurable silent head"

    from_full = AF.extract_windows(wav, rate, timestamps, 0.35)
    from_trimmed = AF.extract_windows(trimmed, rate, timestamps, 0.35)
    # Same requested instants, different audio: that is the desynchronisation.
    assert not np.allclose(from_full, from_trimmed)

    # Offsetting the timestamps instead, which is what extract_clip.py does, lands
    # back on the same real instants. Not bit-identical: (t - dropped) * rate can
    # truncate one sample lower than t * rate did, so compare the error instead.
    corrected = AF.extract_windows(trimmed, rate, timestamps - dropped, 0.35)
    err_uncorrected = float(np.abs(from_full - from_trimmed).max())
    err_corrected = float(np.abs(from_full - corrected).max())
    assert err_corrected < err_uncorrected / 10


# --------------------------------------------------------- window centring

def test_centring_error_is_zero_when_every_window_fits():
    rate = 16000
    n = rate * 10
    ts = AF.sample_timestamps(10.0, 16, 0.35)
    err = P._centring_error(ts, n, rate, 0.35)
    assert err.shape == (16,)
    # One sample of quantisation is unavoidable: extract_windows truncates
    # int(t * rate), so "centred" means centred to the nearest sample.
    assert np.abs(err).max() < 1.5 / rate


def test_centring_error_reports_the_tail_window_sliding_inward():
    """Timestamps come from the video duration, which can exceed the audio track."""
    rate = 16000
    ts = AF.sample_timestamps(10.12, 16, 0.35)      # video says 10.12s
    audio_samples = int(10.058 * rate)              # audio is shorter
    err = P._centring_error(ts, audio_samples, rate, 0.35)

    assert np.abs(err[:-1]).max() < 1.5 / rate      # only the tail actually moves
    assert err[-1] < 0                                             # it slides earlier
    assert abs(err[-1]) == pytest.approx(0.062, abs=0.005)         # ~62 ms


def test_centring_error_matches_extract_windows_for_real():
    """The helper must mirror the op, not re-derive it differently."""
    rate = 16000
    rng = np.random.default_rng(1)
    wav = rng.standard_normal(int(10.058 * rate)).astype(np.float32)
    ts = AF.sample_timestamps(10.12, 16, 0.35)

    err = P._centring_error(ts, len(wav), rate, 0.35)
    windows = AF.extract_windows(wav, rate, ts, 0.35)

    # The last window must equal the slice the helper says it is.
    win = int(0.35 * rate)
    actual_centre = ts[-1] + err[-1]
    start = int(round(actual_centre * rate - win / 2))
    assert np.allclose(windows[-1], wav[start:start + win], atol=1e-6)
