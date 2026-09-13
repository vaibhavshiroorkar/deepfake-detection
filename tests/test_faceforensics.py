"""The FF++ adapter decides the split, so its filename parsing is load-bearing.

A fake is named `<target>_<source>.mp4`, where the target supplies the video and
the source supplies the face. The split protocol groups on the identity being
replaced, which is the target's. Reading those two backwards puts a clip's two
halves in different partitions and leaks identity across the split, which no
metric would report.
"""

import pytest

pytest.importorskip("pandas")

from deepfake_detection.data.faceforensics import (
    METHODS,
    identity,
    manifest_from_ffpp,
)


def _drop(root):
    real = root / "real"
    real.mkdir(parents=True)
    for name in ("000.mp4", "001.mp4"):
        (real / name).write_bytes(b"video")
    for method in ("Deepfakes", "FaceSwap"):
        folder = root / "fake" / method
        folder.mkdir(parents=True)
        (folder / "000_001.mp4").write_bytes(b"video")
        (folder / "001_000.mp4").write_bytes(b"video")
    actors = root / "fake" / "DeepFakeDetection"
    actors.mkdir(parents=True)
    (actors / "01_02__outside_talking__YVGY8LOK.mp4").write_bytes(b"video")
    return root


def test_the_replaced_identity_is_the_source_not_the_face_donor() -> None:
    assert identity("000_003") == ("ffpp-000", "ffpp-003")
    assert identity("000") == ("ffpp-000", "")


def test_the_actor_set_uses_the_same_rule() -> None:
    source, target = identity("01_02__outside_talking__YVGY8LOK")

    assert (source, target) == ("ffpp-01", "ffpp-02")


def test_a_manipulation_and_its_original_share_a_source_identity(tmp_path) -> None:
    """Otherwise the pair straddles the split and the fake leaks its original."""
    frame = manifest_from_ffpp(_drop(tmp_path / "ffpp"), tmp_path)

    original = frame[frame["clip_id"] == "ffpp__real__000"].iloc[0]
    manipulated = frame[frame["clip_id"] == "ffpp__Deepfakes__000_001"].iloc[0]
    assert original["source"] == manipulated["source"] == "ffpp-000"


def test_every_fake_keeps_its_original_audio(tmp_path) -> None:
    """FF++ manipulates video only. An audio branch trained on these labels
    would be learning from a genuine soundtrack marked fake."""
    frame = manifest_from_ffpp(_drop(tmp_path / "ffpp"), tmp_path)

    fakes = frame[frame["method"] != "real"]
    assert set(fakes["manipulation_type"]) == {"FakeVideo-RealAudio"}


def test_methods_can_be_held_out_for_the_unseen_family_protocol(tmp_path) -> None:
    frame = manifest_from_ffpp(
        _drop(tmp_path / "ffpp"), tmp_path, methods=("Deepfakes",)
    )

    assert set(frame["method"]) == {"real", "ffpp-deepfakes"}


def test_an_empty_drop_is_refused_rather_than_returning_no_rows(tmp_path) -> None:
    empty = tmp_path / "nothing"
    empty.mkdir()

    with pytest.raises(ValueError, match="No FaceForensics"):
        manifest_from_ffpp(empty, tmp_path)


def test_every_named_family_is_one_the_corpus_ships() -> None:
    assert "DeepFakeDetection" in METHODS
    assert len(set(METHODS)) == len(METHODS) == 6
