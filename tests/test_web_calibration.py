import math

import pytest

from deepfake_detection.webapp import calibration as cal


def test_check_score_is_the_mean_seed_logit_and_survives_saturated_probabilities() -> None:
    assert cal.check_score([0.5, 0.5, 0.5]) == pytest.approx(0.0)
    assert cal.check_score([0.9]) == pytest.approx(math.log(9))
    assert math.isfinite(cal.check_score([0.0, 1.0, 1.0]))


def test_percentile_is_the_weighted_share_below_with_half_the_ties() -> None:
    ref = cal.Reference.from_scores([1.0, 2.0, 3.0, 4.0], [1, 1, 1, 1])
    assert ref.percentile(2.5) == pytest.approx(0.5)
    assert ref.percentile(0.0) == 0.0
    assert ref.percentile(9.0) == 1.0
    assert ref.percentile(2.0) == pytest.approx(0.375)


def test_each_corpus_carries_equal_weight() -> None:
    weights = cal.corpus_weights(["a", "b", "b", "b"])
    assert weights == pytest.approx([0.5, 1 / 6, 1 / 6, 1 / 6])
    ref = cal.Reference.from_scores([10.0, 0.0, 0.0, 0.0], weights)
    assert ref.percentile(5.0) == pytest.approx(0.5)


def _calibration() -> cal.Calibration:
    ref = cal.Reference.from_scores([float(i) for i in range(100)], [1] * 100)
    return cal.Calibration(references={"face": ref, "voice": ref, "sync": ref}, threshold=0.97, lower=0.8)


def test_no_available_check_means_cant_check() -> None:
    assert _calibration().decide({"face": None, "voice": None, "sync": None}).kind == "cant_check"


def test_outcomes_follow_the_combined_percentile() -> None:
    c = _calibration()
    signs = c.decide({"face": 99.5, "voice": 10.0, "sync": None})
    assert (signs.kind, signs.raised) == ("signs", ["face"])
    assert c.decide({"face": 90.0, "voice": 10.0, "sync": 10.0}).kind == "cant_tell"
    assert c.decide({"face": 50.0, "voice": 10.0, "sync": 10.0}).kind == "no_signs"


def test_built_threshold_flags_at_most_five_in_a_hundred_calibration_reals() -> None:
    rows = [{"corpus": "a" if i % 3 else "b", "label": 0,
             "scores": {"face": float(i % 97), "voice": float((i * 7) % 89),
                        "sync": float((i * 13) % 83) if i % 2 else None}} for i in range(600)]
    rows += [{"corpus": "a", "label": 1, "scores": {"face": 500.0, "voice": 0.0, "sync": None}}]
    built, summary = cal.build(rows)
    reals = [r for r in rows if r["label"] == 0]
    flagged = [built.decide(r["scores"]).kind == "signs" for r in reals]
    weights = cal.corpus_weights([r["corpus"] for r in reals])
    assert sum(w for w, f in zip(weights, flagged, strict=True) if f) <= 0.05 + 1e-9
    assert summary["a"]["fake"]["signs"] == 1


def test_committed_calibration_matches_its_source_files() -> None:
    """A result file changed without `scripts/build_web_calibration.py` being rerun fails here."""
    import hashlib

    from deepfake_detection.paths import PROJECT_ROOT

    path = PROJECT_ROOT / "results" / "web-calibration.json"
    loaded = cal.Calibration.from_json(path.read_text(encoding="utf-8"))
    assert 0 < loaded.lower < loaded.threshold < 1
    for source, digest in loaded.meta["sources"].items():
        assert hashlib.sha256((PROJECT_ROOT / source).read_bytes()).hexdigest() == digest, source


def test_json_round_trip_keeps_every_decision() -> None:
    c = _calibration()
    again = cal.Calibration.from_json(c.to_json())
    for scores in ({"face": 99.5, "voice": None, "sync": 3.0}, {"face": 1.0, "voice": 2.0, "sync": 85.0}):
        assert again.decide(scores) == c.decide(scores)
