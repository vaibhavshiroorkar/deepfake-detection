import numpy as np
import soundfile as sf
import torch

from dfd.data.manifest import Record
from dfd.sweep import encoders, extract_audio, extract_images, extract_video, train_reference
from dfd.sweep.tasks import TASKS


def test_extraction_job_names_match_the_task_definitions():
    for tid, jobs in (("T-V", extract_video.JOBS), ("T-I", extract_images.JOBS), ("T-A", extract_audio.JOBS)):
        names = [f"{j[0]}_{j[-1]}" if tid == "T-V" else j[0] for j in jobs]
        t = TASKS[tid]
        assert names == [t.train, t.val, t.devx], (tid, names)


def test_window_is_repeat_padded_and_exactly_64600_samples():
    w = extract_audio.window_of(np.arange(1, 11, dtype=np.float32))
    assert w.shape == (64600,) and w[0] == 1 and w[10] == 1 and w[9] == 10          # tiled
    long = np.arange(100000, dtype=np.float32)
    assert (extract_audio.window_of(long) == long[:64600]).all()                    # the first 4 seconds
    assert extract_audio.window_of(np.zeros(0, np.float32)).shape == (64600,)
    assert encoders.WINDOW == 64600 and encoders.WHISPER_FRAMES == 202             # 4 s / 20 ms, rounded up


def _rec(i, path, label=0):
    return Record(sample_id=f"t:{i}", dataset="t", family="t", path=path, modality="audio", label=label, method="m", group_id=f"t:g{i}",
                  group_basis="class", identity_known=False)


def test_audio_cache_returns_the_same_samples_as_a_fresh_decode(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(0)
    sf.write(data / "a.wav", rng.uniform(-0.5, 0.5, 16000).astype(np.float32), 16000)   # 1 s: repeat-padded to 4 s
    recs = [_rec(0, "a.wav")]
    cache = tmp_path / "cache"
    first = extract_audio.AudioSet(recs, root=data, cache=cache)[0][0].numpy()
    again = extract_audio.AudioSet(recs, root=data, cache=cache)[0][0].numpy()          # now read from the cache
    assert first.shape == (64600,) and np.array_equal(first, again) and len(list(cache.glob("*.npy"))) == 1


def test_mel_filterbank_and_reference_models_run_on_cpu():
    fb = train_reference.mel_filterbank()
    assert fb.shape == (64, 257) and (fb >= 0).all() and (fb.sum(1) > 0).all()
    m = train_reference.build_model("T-A")
    wav = torch.randn(3, 64600)
    assert m(wav).shape == (3,) and train_reference.LogMel()(wav).shape[:3] == (3, 1, 64)
    img = train_reference.build_model("T-V")
    assert img(torch.randn(2, 3, 224, 224)).shape == (2,)
    assert 3e5 < sum(p.numel() for p in img.parameters()) < 5e5                       # a small network, as intended


def test_reference_budget_is_the_fixed_one_of_the_protocol():
    assert (train_reference.EPOCHS, train_reference.LR, train_reference.SEEDS) == (10, 1e-3, (0, 1, 2))
    assert len(train_reference.strided(list(range(1000)), 10)) == 10
