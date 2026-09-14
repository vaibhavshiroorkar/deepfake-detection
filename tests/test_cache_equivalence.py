"""The equivalence table is a safety valve, so its edges are worth pinning.

It exists because one corpus was cached under a different code version and was
therefore uncomparable with every other. It must stay narrow: equal hashes and
verified pairs, nothing else, and never a prefix or a tolerance.
"""

from deepfake_detection.views.equivalence import (
    PROGRAM_V1,
    STREAMS_V1,
    evidence_for,
    same_preprocessing,
)


def test_a_hash_is_the_same_as_itself() -> None:
    assert same_preprocessing(PROGRAM_V1, PROGRAM_V1)


def test_the_verified_pair_is_accepted_in_both_directions() -> None:
    assert same_preprocessing(PROGRAM_V1, STREAMS_V1)
    assert same_preprocessing(STREAMS_V1, PROGRAM_V1)


def test_an_unrelated_hash_is_refused() -> None:
    assert not same_preprocessing(PROGRAM_V1, "f" * 64)


def test_a_shared_prefix_is_not_enough() -> None:
    """The whole point is that near-identical settings are not identical data."""
    almost = PROGRAM_V1[:-1] + ("0" if PROGRAM_V1[-1] != "0" else "1")

    assert not same_preprocessing(PROGRAM_V1, almost)


def test_every_accepted_pair_names_its_evidence() -> None:
    """An entry without evidence is an assertion, which is what this avoids."""
    reason = evidence_for(PROGRAM_V1, STREAMS_V1)

    assert "runs/cache-equivalence.json" in reason
    assert "identical" in reason


def test_an_unverified_pair_has_no_evidence() -> None:
    assert evidence_for(PROGRAM_V1, "f" * 64) == ""
