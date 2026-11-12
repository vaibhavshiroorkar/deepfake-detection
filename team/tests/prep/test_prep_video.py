import numpy as np
import pytest

from dfd.prep import video as V
from prep_test_helpers import frame_number, make_video


def test_sample_indices_count_range_spacing():
    for total in (17, 50, 637, 1000):
        for n in (1, 4, 16):
            idx = V.sample_indices(total, n)
            assert len(idx) == n
            assert idx == sorted(set(idx))
            assert 0 <= idx[0] and idx[-1] < total
            if n > 1:
                gaps = np.diff(idx)
                assert gaps.max() - gaps.min() <= 1  # uniform up to rounding
                assert abs(gaps.mean() - total / n) < 1.0


def test_sample_indices_centres_of_segments():
    assert V.sample_indices(100, 4) == [12, 37, 62, 87]


def test_sample_indices_short_video_returns_all_frames():
    assert V.sample_indices(5, 16) == [0, 1, 2, 3, 4]
    assert V.sample_indices(16, 16) == list(range(16))
    assert V.sample_indices(0, 16) == []
    with pytest.raises(ValueError):
        V.sample_indices(10, 0)


def test_probe(video50):
    info = V.probe(video50)
    assert (info.width, info.height, info.n_frames) == (64, 48, 50)
    assert info.fps == pytest.approx(25.0)


@pytest.mark.parametrize("backend", V.BACKENDS)
def test_backends_return_the_requested_frames(video50, backend):
    want = V.sample_indices(50, 8)
    got = V.read_frames(video50, n=8, backend=backend)
    assert [i for i, _ in got] == want
    for i, fr in got:
        assert fr.shape == (48, 64, 3) and fr.dtype == np.uint8
        assert frame_number(fr) == i  # pixel content belongs to that frame number


def test_backends_agree_pixel_for_pixel(video50):
    ref = V.read_frames(video50, n=16, backend="cv_seq")
    for b in ("cv_seek", "ffmpeg"):
        other = V.read_frames(video50, n=16, backend=b)
        assert [i for i, _ in other] == [i for i, _ in ref]
        assert all(np.array_equal(x, y) for (_, x), (_, y) in zip(ref, other))


def test_explicit_indices_and_generator_is_lazy(video50):
    it = V.iter_frames(video50, indices=[3, 40], backend="cv_seq")
    assert iter(it) is it  # a generator: frames are not collected into a list
    assert [i for i, _ in it] == [3, 40]


def test_short_video_yields_fewer_frames(tmp_path, ffmpeg_ok):
    p = make_video(tmp_path / "short.mp4", n_frames=6)
    assert [i for i, _ in V.iter_frames(p, n=16)] == list(range(6))


def test_bad_inputs(tmp_path):
    with pytest.raises(V.VideoOpenError):
        V.probe(tmp_path / "missing.mp4")
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    with pytest.raises(V.VideoOpenError):
        V.probe(junk)
    with pytest.raises(ValueError):
        V.iter_frames(junk, backend="nope", info=V.VideoInfo(1, 1, 1, 1.0))


def test_is_stable(tmp_path):
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    assert not V.is_stable(f, 20)       # just written
    assert V.is_stable(f, 0)
    g = tmp_path / "b.mp4.part"
    g.write_bytes(b"x")
    assert not V.is_stable(g, 0)        # .part never counts
    assert not V.is_stable(tmp_path / "none.mp4", 0)
