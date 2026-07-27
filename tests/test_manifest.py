"""
Tests for the clip-level labelling convention.

FakeAVCeleb's `type` column is the ground truth for what was manipulated. This
module pins down how that becomes a binary label, because getting it wrong is
silent -- the pipeline still runs, the metrics still print, and every number
downstream is meaningless.
"""
import io
from pathlib import Path

import pandas as pd
import pytest

from preprocessing.manifest import clip_label, manifest_from_meta


def test_only_real_video_real_audio_is_labelled_real():
    """A clip is real only if BOTH tracks are real."""
    assert clip_label("RealVideo-RealAudio") == 0
    assert clip_label("RealVideo-FakeAudio") == 1
    assert clip_label("FakeVideo-RealAudio") == 1
    assert clip_label("FakeVideo-FakeAudio") == 1


def test_unrecognised_type_raises_instead_of_guessing():
    """
    A typo must not quietly become "fake". Silently mislabelling is the
    failure mode that produces confident, meaningless metrics.
    """
    with pytest.raises(ValueError, match="RealVideo-Realaudio"):
        clip_label("RealVideo-Realaudio")


# --- meta_data.csv -> manifest -------------------------------------------- #
# The same function backs audit_dataset.py's CSV and the dashboard's in-memory
# manifest for an un-audited drop, so both see identical rows.

_META = (
    "source,target1,target2,method,category,type,race,gender,path,\n"
    "id1,-,-,real,A,RealVideo-RealAudio,African,men,00109.mp4,"
    "FakeAVCeleb/RealVideo-RealAudio/African/men/id1\n"
    "id2,-,-,wav2lip,D,FakeVideo-FakeAudio,Asian,women,00001.mp4,"
    "FakeAVCeleb/FakeVideo-FakeAudio/Asian/women/id2\n"
    # same physical file as row 2, relabelled -- FakeAVCeleb really ships these
    "id2,-,-,faceswap-wav2lip,D,FakeVideo-FakeAudio,Asian,women,00001.mp4,"
    "FakeAVCeleb/FakeVideo-FakeAudio/Asian/women/id2\n"
)


def _meta():
    return pd.read_csv(io.StringIO(_META))


def test_manifest_rows_drop_the_leading_dataset_component_and_stay_relative():
    df = manifest_from_meta(_meta(), Path("data/drop"), Path("data"),
                            require_exists=False)
    assert df["video_path"].iloc[0] == str(
        Path("drop/RealVideo-RealAudio/African/men/id1/00109.mp4"))
    assert df["clip_id"].iloc[0] == "A__id1__00109"
    assert list(df["label"]) == [0, 1]


def test_duplicate_physical_files_are_deduped_to_one_row():
    df = manifest_from_meta(_meta(), Path("data/drop"), Path("data"),
                            require_exists=False)
    assert len(df) == 2
    assert df["method"].iloc[1] == "wav2lip"    # first spelling wins


def test_missing_files_are_dropped_when_existence_is_required(tmp_path):
    df = manifest_from_meta(_meta(), tmp_path / "drop", tmp_path)
    assert len(df) == 0                          # nothing was written to disk


def test_wrong_column_count_raises_instead_of_mislabelling():
    # A positional rename onto the wrong shape would silently scramble every
    # column, which is exactly the silent-mislabelling failure we avoid.
    with pytest.raises(ValueError, match="columns"):
        manifest_from_meta(pd.DataFrame({"a": [1], "b": [2]}), Path("r"), Path("d"))
