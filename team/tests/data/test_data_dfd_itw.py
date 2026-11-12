"""DFD and In-the-Wild builders: synthetic fixtures for the logic, real files for the counts."""

import pytest

from data_test_helpers import touch
from dfd.data.builders import dfd as dfd_builder
from dfd.data.builders import in_the_wild
from dfd.data.manifest import default_data_root, validate_records

REAL_ROOT = default_data_root()


def test_dfd_synthetic(tmp_path):
    o = tmp_path / "dfd/original_sequences/actors/c23/videos"
    m = tmp_path / "dfd/manipulated_sequences/DeepFakeDetection/c23/videos"
    for n in ("01__exit_phone_room.mp4", "02__kitchen_pan.mp4"):
        touch(o / n)
    for n in ("01_02__exit_phone_room__YVGY8LOK.mp4", "02_01__kitchen_pan__ABCD1234.mp4", "03_01__x__ZZ.mp4"):
        touch(m / n)
    touch(m / "partial.tmp")
    recs, rep = dfd_builder.build(tmp_path)
    assert validate_records(recs) == []
    by = {r.sample_id: r for r in recs}
    f = by["dfd:manipulated/01_02__exit_phone_room__YVGY8LOK.mp4"]
    assert (f.label, f.group_id, f.method, f.notes, f.family) == (
        1, "dfd:01", "DeepFakeDetection", "source=02;experiment=YVGY8LOK", "dfd")
    r = by["dfd:original/01__exit_phone_room.mp4"]
    assert r.group_id == "dfd:01" and r.label == 0 and r.identity_known
    assert rep["actors_total"] == 3 and rep["connected_components_over_actor_graph"] == 1
    assert rep["manipulated_found"] == 3 and rep["complete"] is False
    assert any("partial.tmp" in x for x in rep["unparsed_or_ignored_files"])


def test_in_the_wild_synthetic(tmp_path):
    b = tmp_path / "in_the_wild/release_in_the_wild"
    for n in ("0.wav", "1.wav", "2.wav"):
        touch(b / n)
    touch(b / "attribution.txt")
    meta = b / "meta.csv"
    meta.write_text("file,speaker,label\n0.wav,Ann B,spoof\n1.wav,Ann B,bona-fide\n2.wav,Cy D,bona-fide\n",
                    encoding="utf-8")
    recs, rep = in_the_wild.build(tmp_path)
    assert validate_records(recs) == []
    by = {r.sample_id: r for r in recs}
    assert (by["in_the_wild:0.wav"].label, by["in_the_wild:0.wav"].method) == (1, "spoof")
    assert by["in_the_wild:1.wav"].method == "real" and by["in_the_wild:1.wav"].group_id == "in_the_wild:Ann B"
    assert rep["speakers"] == 2 and rep["speakers_with_both_labels"] == 1 and rep["label_counts"] == {0: 2, 1: 1}
    assert "attribution.txt" in rep["non_wav_files_ignored"]
    meta.write_text("file,speaker,label\n0.wav,Ann B,maybe\n", encoding="utf-8")
    with pytest.raises(ValueError):
        in_the_wild.build(tmp_path)
    meta.write_text("file,speaker,label\n9.wav,Ann B,spoof\n", encoding="utf-8")
    with pytest.raises(ValueError):
        in_the_wild.build(tmp_path)


def test_dfd_real_files_counts():
    if not (REAL_ROOT / "dfd" / "manipulated_sequences").is_dir():
        pytest.skip("DFD SKIPPED: data/dfd not present")
    recs, rep = dfd_builder.build(REAL_ROOT)
    if not rep["complete"]:
        pytest.skip(f"DFD SKIPPED: incomplete, {rep['originals_found']} originals, {rep['manipulated_found']} manipulated")
    assert validate_records(recs) == []
    assert (rep["originals_found"], rep["manipulated_found"], len(recs)) == (363, 3068, 3431)
    assert rep["actors_total"] == 28 and rep["actor_groups_in_manifest"] == 28


def test_in_the_wild_real_files_counts():
    if not (REAL_ROOT / "in_the_wild" / "release_in_the_wild" / "meta.csv").is_file():
        pytest.skip("In-the-Wild SKIPPED: meta.csv not present")
    recs, rep = in_the_wild.build(REAL_ROOT)
    assert validate_records(recs) == []
    assert rep["label_counts"] == {0: 19963, 1: 11816} and len(recs) == 31779
    assert rep["speakers"] == 54 and rep["wav_files_not_in_meta"] == 0
