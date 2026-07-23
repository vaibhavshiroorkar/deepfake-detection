import numpy as np
from dashboard.lib.media import sample_timestamps


def test_sample_timestamps_count_and_inset():
    ts = sample_timestamps(duration_sec=8.0, num_frames=16, window_sec=0.35)
    assert ts.shape == (16,)
    assert ts[0] >= 0.35 / 2 - 1e-9          # inset by half a window
    assert ts[-1] <= 8.0 - 0.35 / 2 + 1e-9
    assert np.all(np.diff(ts) > 0)           # strictly increasing


def test_sample_timestamps_degenerate_short_clip():
    ts = sample_timestamps(duration_sec=0.1, num_frames=4, window_sec=0.35)
    assert ts.shape == (4,)
    assert np.all(ts >= 0)
