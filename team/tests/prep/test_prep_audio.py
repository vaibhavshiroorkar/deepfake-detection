import numpy as np

from dfd.prep import audio as A
from prep_test_helpers import make_video


def test_extract_audio_16k_mono(tmp_path, ffmpeg_ok):
    p = make_video(tmp_path / "av.mp4", n_frames=50, with_audio=True)  # 2 s
    x = A.extract_audio(p)
    assert x.dtype == np.float32 and x.ndim == 1
    assert abs(len(x) / A.DEFAULT_SR - 2.0) < 0.15
    assert 0.05 < np.abs(x).max() <= 1.0    # the ffmpeg sine has amplitude about 0.125, not silence
    assert A.has_audio_stream(p)


def test_video_without_audio_gives_empty(video50):
    assert A.extract_audio(video50).size == 0
    assert not A.has_audio_stream(video50)
    assert A.extract_audio(video50, check_stream=True).size == 0


def test_extract_to_wav_roundtrip(tmp_path, ffmpeg_ok):
    import soundfile as sf
    p = make_video(tmp_path / "av.mp4", n_frames=25, with_audio=True)
    out = tmp_path / "o" / "a.wav"
    assert A.extract_audio_to_wav(p, out)
    info = sf.info(str(out))
    assert (info.samplerate, info.channels) == (16000, 1)


def test_load_flac_and_resample(tmp_path, ffmpeg_ok):
    import soundfile as sf
    t = np.arange(16000) / 16000
    x = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    f = tmp_path / "a.flac"
    sf.write(str(f), x, 16000)
    y = A.load_audio(f)
    assert y.dtype == np.float32 and len(y) == 16000 and np.abs(y - x).max() < 1e-3
    w = tmp_path / "b.wav"
    sf.write(str(w), np.stack([x, x], 1), 16000)          # stereo to mono
    assert len(A.load_audio(w)) == 16000
    z = tmp_path / "c.wav"
    sf.write(str(z), np.tile(x, 3)[: 44100], 44100)        # other rate: resampled to 16 kHz
    assert abs(len(A.load_audio(z)) - 16000) <= 2
