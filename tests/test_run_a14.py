"""run_a14.py: the audio perturbations of P-001 amendment A14."""

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_a14.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_a14", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clip(valid=48000, length=64000):
    wave = np.zeros(length, dtype=np.float32)
    wave[:valid] = np.linspace(-0.5, 0.5, valid, dtype=np.float32)
    return wave, valid


def test_loudness_levels_the_valid_samples_and_keeps_the_padding_silent():
    module = _module()
    wave, valid = _clip()
    out, n = module.perturb(wave, valid, "loudness")
    assert n == valid
    rms = np.sqrt(np.mean(out[:valid].astype(np.float64) ** 2))
    assert abs(20 * np.log10(rms) - (-23.0)) < 1e-3
    assert not out[valid:].any()


def test_edges_cut_250_ms_at_each_end_and_shrink_the_valid_count():
    module = _module()
    wave, valid = _clip()
    out, n = module.perturb(wave, valid, "edges")
    # The valid count must shrink, or the model would read the new zeros as audio: digital silence,
    # the very edge cue A14 removes.
    assert n == valid - 2 * 4000
    assert np.array_equal(out[:n], wave[4000:valid - 4000])
    assert not out[n:].any()
    assert out.shape == wave.shape


def test_both_cuts_then_levels():
    module = _module()
    wave, valid = _clip()
    out, n = module.perturb(wave, valid, "both")
    assert n == valid - 8000
    rms = np.sqrt(np.mean(out[:n].astype(np.float64) ** 2))
    assert abs(20 * np.log10(rms) - (-23.0)) < 1e-3


def test_too_short_to_cut_is_left_as_is_and_flagged():
    module = _module()
    wave, valid = _clip(valid=20000)
    out, n = module.perturb(wave, valid, "edges")
    assert n == valid and np.array_equal(out, wave)
    assert module.too_short(valid)


def test_input_is_not_modified():
    module = _module()
    wave, valid = _clip()
    before = wave.copy()
    module.perturb(wave, valid, "both")
    assert np.array_equal(wave, before)
