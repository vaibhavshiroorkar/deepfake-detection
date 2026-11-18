import math

import pytest

from deepfake_detection.webapp import calibration as cal


def test_check_score_is_the_mean_seed_logit_and_survives_saturated_probabilities() -> (
    None
):
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
    return cal.Calibration(
        references={"face": ref, "voice": ref, "sync": ref}, threshold=0.97, lower=0.8
    )


def test_no_available_check_means_cant_check() -> None:
    assert (
        _calibration().decide({"face": None, "voice": None, "sync": None}).kind
        == "cant_check"
    )


def test_outcomes_follow_the_combined_percentile() -> None:
    c = _calibration()
    signs = c.decide({"face": 99.5, "voice": 10.0, "sync": None})
    assert (signs.kind, signs.raised) == ("signs", ["face"])
    assert c.decide({"face": 90.0, "voice": 10.0, "sync": 10.0}).kind == "cant_tell"
    assert c.decide({"face": 50.0, "voice": 10.0, "sync": 10.0}).kind == "no_signs"


def test_built_threshold_flags_at_most_five_in_a_hundred_calibration_reals() -> None:
    rows = [
        {
            "corpus": "a" if i % 3 else "b",
            "label": 0,
            "scores": {
                "face": float(i % 97),
                "voice": float((i * 7) % 89),
                "sync": float((i * 13) % 83) if i % 2 else None,
            },
        }
        for i in range(600)
    ]
    rows += [
        {
            "corpus": "a",
            "label": 1,
            "scores": {"face": 500.0, "voice": 0.0, "sync": None},
        }
    ]
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
        assert (
            hashlib.sha256((PROJECT_ROOT / source).read_bytes()).hexdigest() == digest
        ), source


def test_json_round_trip_keeps_every_decision() -> None:
    c = _calibration()
    again = cal.Calibration.from_json(c.to_json())
    for scores in (
        {"face": 99.5, "voice": None, "sync": 3.0},
        {"face": 1.0, "voice": 2.0, "sync": 85.0},
    ):
        assert again.decide(scores) == c.decide(scores)


def _with_combined() -> cal.Calibration:
    ref = cal.Reference.from_scores([float(i) for i in range(100)], [1] * 100)
    combined = cal.Reference.from_scores([i / 1000 for i in range(1000)], [1] * 1000)
    return cal.Calibration(
        {"face": ref, "voice": ref, "sync": ref},
        threshold=0.95,
        lower=0.8,
        combined=combined,
    )


def test_one_part_keeps_the_single_clip_thresholds() -> None:
    c = _with_combined()
    assert c.limits(1) == (0.95, 0.8)


def test_more_parts_raise_the_bar_so_real_videos_are_not_flagged_more_often() -> None:
    c = _with_combined()
    threshold, lower = c.limits(10)
    assert threshold == pytest.approx(0.995, abs=0.0015)  # 1 - 0.05 / 10
    assert lower == pytest.approx(0.98, abs=0.0015)  # 1 - 0.20 / 10


def test_the_most_unusual_part_decides_and_is_named() -> None:
    c = _with_combined()
    quiet = {"face": 10.0, "voice": 10.0, "sync": 10.0}  # 10th percentile
    loud = {"face": 99.9, "voice": 10.0, "sync": None}  # above every reference clip
    decision = c.decide_parts([quiet, loud, quiet])
    assert (decision.kind, decision.part, decision.raised) == ("signs", 1, ["face"])
    assert decision.parts == pytest.approx([0.105, 1.0, 0.105])
    assert decision.percentiles["sync"] is None


def test_a_part_flagged_alone_is_only_unsure_among_many() -> None:
    c = _with_combined()
    middling = {
        "face": 96.5,
        "voice": 10.0,
        "sync": 10.0,
    }  # 96.5th percentile: signs alone
    assert c.decide_parts([middling]).kind == "signs"
    assert (
        c.decide_parts(
            [middling] + [{"face": 1.0, "voice": 1.0, "sync": 1.0}] * 19
        ).kind
        == "no_signs"
    )


def test_parts_with_no_check_are_skipped_and_all_empty_is_cant_check() -> None:
    c = _with_combined()
    empty = {"face": None, "voice": None, "sync": None}
    assert c.decide_parts([empty, empty]).kind == "cant_check"
    decision = c.decide_parts([empty, {"face": 99.9, "voice": None, "sync": None}])
    assert (decision.kind, decision.part, decision.parts) == ("signs", 1, [None, 1.0])


def test_many_parts_need_the_combined_reference() -> None:
    with pytest.raises(ValueError, match="combined"):
        _calibration().limits(2)


def test_built_calibration_keeps_the_real_combined_scores_and_round_trips() -> None:
    rows = [
        {
            "corpus": "a" if i % 3 else "b",
            "label": 0,
            "scores": {
                "face": float(i % 97),
                "voice": float((i * 7) % 89),
                "sync": float(i % 5),
            },
        }
        for i in range(600)
    ]
    built, _ = cal.build(rows)
    assert built.limits(1) == (built.threshold, built.lower)
    assert built.combined.quantile(0.95) == pytest.approx(built.threshold)
    again = cal.Calibration.from_json(built.to_json())
    assert again.limits(7) == built.limits(7)


def test_a_calibration_can_be_built_for_the_face_check_alone() -> None:
    rows = [
        {
            "corpus": "a" if i % 2 else "b",
            "label": 0,
            "scores": {"face": float(i), "voice": None, "sync": None},
        }
        for i in range(200)
    ]
    built, _ = cal.build(rows, checks=("face",))
    assert built.decide({"face": 199.5, "voice": None, "sync": None}).kind == "signs"
    assert built.decide({"face": 10.0, "voice": None, "sync": None}).kind == "no_signs"
    again = cal.Calibration.from_json(built.to_json())
    assert again.decide({"face": 150.0, "voice": None, "sync": None}) == built.decide(
        {"face": 150.0, "voice": None, "sync": None}
    )


def test_measured_lines_for_videos_in_parts_replace_the_bonferroni_bound() -> None:
    c = _with_combined()
    measured = cal.Calibration(
        c.references,
        c.threshold,
        c.lower,
        combined=c.combined,
        parts_lines={2: (0.985, 0.94), 4: (0.988, 0.95)},
    )
    assert measured.limits(1) == (0.95, 0.8)  # one part: the single-clip lines
    assert measured.limits(2) == (0.985, 0.94)
    assert measured.limits(3) == (
        0.988,
        0.95,
    )  # between measured counts: the next measured one up
    assert measured.limits(9) == (
        0.988,
        0.95,
    )  # beyond the data: the largest count measured
    assert measured.limits(9) < c.limits(9)  # easier to reach than the Bonferroni bound
    again = cal.Calibration.from_json(measured.to_json())
    assert again.limits(3) == measured.limits(3)


def test_display_scale_puts_real_at_0_neutral_at_half_and_fake_at_1():
    from deepfake_detection.webapp.calibration import display_score

    flag, lower = 0.98, 0.92
    assert display_score(0.0, flag, lower) == 0.0
    assert display_score(lower, flag, lower) == 0.4
    assert display_score((flag + lower) / 2, flag, lower) == 0.5
    assert display_score(flag, flag, lower) == 0.6
    assert display_score(1.0, flag, lower) == 1.0
    assert display_score(None, flag, lower) is None
    steps = [display_score(x / 100, flag, lower) for x in range(101)]
    assert steps == sorted(steps)


def test_a_model_named_as_raising_the_flag_is_on_the_fake_side_of_the_page_scale():
    # A second model at its own 95th percentile sat under the flag line in force and showed 0.40 on the
    # page while being named as raising the flag (seen 11-18 on a cloned-voice clip).
    c = _calibration()
    outcome = c.decide({"face": 99.5, "voice": 95.5, "sync": None})
    flag, lower = c.limits(1)
    assert outcome.kind == "signs"
    assert outcome.raised == ["face"]
    for name in outcome.raised:
        assert (
            cal.display_score(outcome.percentiles[name], flag, lower) >= cal.SHOWN_FLAG
        )
