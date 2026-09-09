"""The pilot tier thins a frozen split; it must not distort or leak it."""

import pytest

from deepfake_detection.data.manifest import ClipRecord
from deepfake_detection.data.protocols import stratified_subsample


def record(clip_id: str, manipulation_type: str, method: str, source: str) -> ClipRecord:
    return ClipRecord.from_mapping(
        {
            "clip_id": clip_id,
            "dataset": "fixture",
            "video_path": f"{clip_id}.mp4",
            "manipulation_type": manipulation_type,
            "method": method,
            "source": source,
        }
    )


def population() -> tuple[ClipRecord, ...]:
    real = [
        record(f"real{index}", "RealVideo-RealAudio", "real", f"s{index}")
        for index in range(10)
    ]
    swaps = [
        record(f"swap{index}", "FakeVideo-RealAudio", "faceswap", f"s{index % 10}")
        for index in range(200)
    ]
    lips = [
        record(f"lip{index}", "FakeVideo-FakeAudio", "wav2lip", f"s{index % 10}")
        for index in range(40)
    ]
    return tuple(real + swaps + lips)


def test_subsample_keeps_every_real_clip() -> None:
    sampled = stratified_subsample(population(), seed=17, fake_ratio=3)

    real = [item for item in sampled if not item.video_fake]
    assert len(real) == 10


def test_subsample_honours_the_fake_ratio() -> None:
    sampled = stratified_subsample(population(), seed=17, fake_ratio=3)

    fake = [item for item in sampled if item.video_fake]
    assert len(fake) == 30
    assert len(sampled) == 40


def test_subsample_keeps_every_method_represented() -> None:
    """A rare method must not round away to nothing, or the holdout ablations
    lose the very families they are meant to test."""
    sampled = stratified_subsample(population(), seed=17, fake_ratio=3)

    assert {item.method for item in sampled} == {"real", "faceswap", "wav2lip"}


def test_subsample_is_deterministic_for_a_seed() -> None:
    first = stratified_subsample(population(), seed=17, fake_ratio=3)
    second = stratified_subsample(population(), seed=17, fake_ratio=3)
    third = stratified_subsample(population(), seed=29, fake_ratio=3)

    assert [item.clip_id for item in first] == [item.clip_id for item in second]
    assert [item.clip_id for item in first] != [item.clip_id for item in third]


def test_subsample_introduces_no_new_sources() -> None:
    original = population()
    sampled = stratified_subsample(original, seed=17, fake_ratio=3)

    assert {item.source for item in sampled} <= {item.source for item in original}


def test_subsample_takes_every_fake_when_the_ratio_exceeds_supply() -> None:
    sampled = stratified_subsample(population(), seed=17, fake_ratio=1000)

    assert len(sampled) == len(population())


def test_subsample_rejects_a_partition_with_no_real_clips() -> None:
    fakes = (record("swap0", "FakeVideo-RealAudio", "faceswap", "s0"),)
    with pytest.raises(ValueError, match="no real clips"):
        stratified_subsample(fakes, seed=17, fake_ratio=3)


def test_subsample_rejects_a_nonpositive_ratio() -> None:
    with pytest.raises(ValueError, match="Fake ratio"):
        stratified_subsample(population(), seed=17, fake_ratio=0)
