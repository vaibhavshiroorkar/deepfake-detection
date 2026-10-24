"""Builder tests: synthetic fixtures for the logic, real files for acceptance check A7.

A7 real-data tests: ASVspoof 2019 LA and the ProGAN val and test sets run when the folders are complete;
FF++ runs only when all 1,000 originals and 1,000 files per method are present, otherwise it is skipped
with a message that says how many files were found.
"""

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

from data_test_helpers import touch, write_asvspoof_fixture, write_ffpp_fixture
from dfd.data.build_manifests import BUILDERS, build_one, main as build_main
from dfd.data.builders import LayoutNotVerified, asvspoof2019_la, celebdf, ffpp, forensynths, stubs
from dfd.data.manifest import default_data_root, read_manifest, validate_records

REAL_ROOT = default_data_root()


# ---------------------------------------------------------------- FF++ synthetic

def test_ffpp_components_and_official_splits(tmp_path, monkeypatch):
    # Official files already present: no network access may happen.
    def no_net(*a, **k):
        raise AssertionError("network used")
    monkeypatch.setattr(urllib.request, "urlopen", no_net)
    pairs = [("001", "002"), ("003", "004"), ("005", "006")]
    write_ffpp_fixture(tmp_path, pairs, official={"train": [["001", "002"]], "val": [["003", "004"]],
                                                  "test": [["005", "006"]]})
    # temporary download file without .mp4 must be ignored
    touch(tmp_path / "ffpp/manipulated_sequences/Deepfakes/c23/videos/tmpabc123")
    recs, rep = ffpp.build(tmp_path)
    assert validate_records(recs) == []
    assert len(recs) == 6 + 2 * 6  # 6 originals + 2 methods x 6 manipulated
    assert rep["originals_found"] == 6 and rep["complete"] is False
    assert rep["manipulated_found"] == {"Deepfakes": 6, "Face2Face": 6, "FaceSwap": 0, "NeuralTextures": 0}
    assert any("tmpabc123" in f for f in rep["ignored_files"])
    by = {r.sample_id: r for r in recs}
    # a video, its source and every fake of either share the group and the official split
    g = by["ffpp:original:001"].group_id
    assert by["ffpp:original:002"].group_id == g
    assert by["ffpp:Deepfakes:001_002"].group_id == g
    assert by["ffpp:Face2Face:002_001"].group_id == g
    assert by["ffpp:original:003"].group_id != g
    assert by["ffpp:Deepfakes:003_004"].official_split == "val"
    assert by["ffpp:original:005"].official_split == "test"
    assert all(not r.identity_known and r.group_basis == "source_video" for r in recs)
    assert rep["components_spanning_count"] == 0
    assert rep["components_that_are_not_exactly_one_official_pair"] == 0
    assert rep["groups"] == 3


def test_ffpp_chain_merges_and_spanning_split_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    # edges 001-002 and 002-003 make one component of three videos
    write_ffpp_fixture(tmp_path, [("001", "002"), ("002", "003")],
                       official={"train": [["001", "002"]], "val": [["002", "003"]], "test": []})
    recs, rep = ffpp.build(tmp_path)
    assert len({r.group_id for r in recs}) == 1
    assert rep["components_spanning_count"] == 1
    assert rep["components_that_are_not_exactly_one_official_pair"] == 1


def test_ffpp_missing_folders_gives_empty_but_valid_report(tmp_path):
    for n in ("train", "val", "test"):
        touch(tmp_path / "_manifests" / "ffpp_official_splits" / f"{n}.json", b"[]")
    recs, rep = ffpp.build(tmp_path)
    assert recs == [] and rep["originals_found"] == 0 and rep["complete"] is False


# ---------------------------------------------------------------- ASVspoof synthetic

def test_asvspoof_synthetic(tmp_path):
    write_asvspoof_fixture(tmp_path)
    recs, rep = asvspoof2019_la.build(tmp_path)
    assert validate_records(recs) == []
    assert len(recs) == 4 + 2 + 3
    assert rep["partitions"]["dev"]["flac_files_not_in_cm_protocol"] == 1  # the enrolment utterance
    assert rep["partitions"]["eval"]["flac_files_not_in_cm_protocol"] == 1
    assert all("A10456" not in r.sample_id for r in recs)  # extras never enter the manifest
    assert rep["speakers_in_more_than_one_partition_count"] == 0
    assert rep["complete"] is False  # counts differ from the documented ones
    r = next(x for x in recs if x.sample_id.endswith("LA_T_1002"))
    assert (r.label, r.method, r.group_id, r.official_split) == (1, "A01", "asvspoof2019_la:LA_0001", "train")
    assert r.path.startswith("asvspoof2019_la/LA/ASVspoof2019_LA_train/flac/")
    assert r.identity_known and r.group_basis == "identity" and r.modality == "audio"
    assert {x.method for x in recs if x.label == 0} == {"real"}


def test_asvspoof_speaker_overlap_is_reported(tmp_path):
    write_asvspoof_fixture(tmp_path)
    p = tmp_path / "asvspoof2019_la/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
    p.write_text(p.read_text().replace("LA_0003", "LA_0001"), encoding="utf-8")
    _, rep = asvspoof2019_la.build(tmp_path)
    assert rep["speakers_in_more_than_one_partition_count"] == 1


def test_asvspoof_protocol_line_parsing():
    assert asvspoof2019_la.parse_protocol_line("LA_0079 LA_T_1138215 - - bonafide") == (
        "LA_0079", "LA_T_1138215", "-", "bonafide")
    assert asvspoof2019_la.parse_protocol_line("  \n") is None
    with pytest.raises(ValueError):
        asvspoof2019_la.parse_protocol_line("LA_0079 LA_T_1 - bonafide")
    with pytest.raises(ValueError):
        asvspoof2019_la.parse_protocol_line("LA_0079 LA_T_1 - - maybe")
    with pytest.raises(ValueError):
        asvspoof2019_la.parse_protocol_line("LA_0079 LA_T_1 - - spoof")


# ---------------------------------------------------------------- ForenSynths and diffusion synthetic

def _images(root: Path, rel: str, n: int, ext: str = ".png"):
    for i in range(n):
        touch(root / rel / f"{i:03d}{ext}")


def test_progan_synthetic(tmp_path):
    for cls in ("airplane", "bird"):
        _images(tmp_path, f"forensynths/progan_test/progan/{cls}/0_real", 3)
        _images(tmp_path, f"forensynths/progan_test/progan/{cls}/1_fake", 3)
    recs, rep = forensynths.build_progan_test(tmp_path)
    assert validate_records(recs) == []
    assert len(recs) == 12 and rep["classes"] == 2 and rep["complete"] is False
    r = next(x for x in recs if "airplane/1_fake/001" in x.sample_id)
    assert (r.label, r.method, r.group_id, r.group_basis) == (
        1, "progan", "forensynths_progan_test:airplane", "class")
    assert r.path == "forensynths/progan_test/progan/airplane/1_fake/001.png"
    assert not r.identity_known


def test_cnn_and_diffusion_layouts_synthetic(tmp_path):
    base = "forensynths/cnn_synth_subset"
    _images(tmp_path, f"{base}/biggan/0_real", 2)
    _images(tmp_path, f"{base}/biggan/1_fake", 2)
    _images(tmp_path, f"{base}/stylegan/car/0_real", 1)
    _images(tmp_path, f"{base}/stylegan/car/1_fake", 1)
    _images(tmp_path, f"{base}/whichfaceisreal/1_fake", 1, ".jpeg")
    recs, rep = forensynths.build_cnn_subset(tmp_path)
    groups = {r.group_id: r.group_basis for r in recs}
    assert groups == {
        "forensynths_cnn_subset:biggan": "generator",
        "forensynths_cnn_subset:stylegan:car": "class",
        "forensynths_cnn_subset:whichfaceisreal": "generator",
    }
    assert {r.method for r in recs if r.label == 1} == {"biggan", "stylegan", "whichfaceisreal"}
    # diffusion: fake-only and real-only folders under the extra diffusion_datasets level
    _images(tmp_path, "diffusion_test/diffusion_datasets/ldm_100/1_fake", 2)
    _images(tmp_path, "diffusion_test/diffusion_datasets/laion/0_real", 2, ".jpg")
    drecs, drep = forensynths.build_diffusion_test(tmp_path)
    assert {r.group_id for r in drecs} == {"diffusion_test:ldm_100", "diffusion_test:laion"}
    assert {(r.group_id.split(":")[1], r.label) for r in drecs} == {("ldm_100", 1), ("laion", 0)}
    assert validate_records(recs + drecs) == []


# ---------------------------------------------------------------- Celeb-DF v2 synthetic and stubs

def test_celebdf_v2_synthetic(tmp_path):
    b = tmp_path / "celebdf_v2"
    touch(b / "Celeb-real" / "id0_0000.mp4")
    touch(b / "Celeb-real" / "id1_0003.mp4")
    touch(b / "Celeb-synthesis" / "id0_id16_0000.mp4")
    touch(b / "YouTube-real" / "00001.mp4")
    touch(b / "Celeb-real" / "partial.tmp")
    (b / "List_of_testing_videos.txt").write_text(
        "1 YouTube-real/00001.mp4\n0 Celeb-synthesis/id0_id16_0000.mp4\n", encoding="utf-8")
    recs, rep = celebdf.build(tmp_path)
    assert validate_records(recs) == []
    by = {r.sample_id: r for r in recs}
    fake = by["celebdf_v2:Celeb-synthesis/id0_id16_0000.mp4"]
    assert (fake.label, fake.group_id, fake.notes, fake.official_split) == (1, "celebdf_v2:id0", "ids=id0|id16", "test")
    real = by["celebdf_v2:Celeb-real/id0_0000.mp4"]
    assert real.group_id == fake.group_id and real.identity_known and real.official_split == ""
    yt = by["celebdf_v2:YouTube-real/00001.mp4"]
    assert not yt.identity_known and yt.group_basis == "source_video"
    assert {r.family for r in recs} == {"celebdf"}
    assert rep["test_list_lines"] == 2 and rep["complete"] is False
    assert any("partial.tmp" in x for x in rep["unparsed_or_ignored_files"])


def test_celebdf_name_parsing_errors():
    assert celebdf.parse_celeb_name("Celeb-real", "id12_0001.mp4")["target"] == "id12"
    with pytest.raises(ValueError):
        celebdf.parse_celeb_name("Celeb-real", "id0_id16_0000.mp4")
    with pytest.raises(ValueError):
        celebdf.parse_celeb_name("Celeb-synthesis", "id0_0000.mp4")


def test_stubs_raise_and_parsers_work(tmp_path):
    for fn in (celebdf.build_pp, stubs.build_lavdf, stubs.build_fakeavceleb, stubs.build_wilddeepfake):
        with pytest.raises(LayoutNotVerified, match="layout not yet verified"):
            fn(tmp_path)
    assert stubs.extract_voxceleb_ids("RealVideo/id00012/xyz_id00012_id00345.mp4") == ["id00012", "id00345"]
    assert stubs.parse_dfd_name("01__exit_phone_room.mp4") == {
        "label": 0, "target": "01", "source": None, "scene_or_sequence": "exit_phone_room", "experiment": None}
    d = stubs.parse_dfd_name("01_02__walk_down_hall_angry__YVGY8LOK.mp4")
    assert (d["label"], d["target"], d["source"], d["experiment"]) == (1, "01", "02", "YVGY8LOK")
    with pytest.raises(ValueError):
        stubs.parse_dfd_name("nonsense.mp4")


def test_cli_writes_manifest_and_stub_exit_code(tmp_path, capsys):
    write_asvspoof_fixture(tmp_path)
    assert build_main(["--dataset", "asvspoof2019_la", "--data-root", str(tmp_path)]) == 0
    assert (tmp_path / "_manifests" / "asvspoof2019_la.manifest.csv").exists()
    assert (tmp_path / "_manifests" / "asvspoof2019_la.report.json").exists()
    assert len(read_manifest(tmp_path / "_manifests" / "asvspoof2019_la.manifest.csv")) == 9
    assert build_main(["--dataset", "lavdf", "--data-root", str(tmp_path)]) == 2
    assert "layout not yet verified" in capsys.readouterr().out
    assert build_main(["--dataset", "nope", "--data-root", str(tmp_path)]) == 2


def test_cli_module_runs_as_script(tmp_path):
    write_asvspoof_fixture(tmp_path)
    src = Path(__file__).resolve().parents[2] / "src"
    out = subprocess.run([sys.executable, "-m", "dfd.data.build_manifests", "--dataset", "asvspoof2019_la",
                          "--data-root", str(tmp_path)], capture_output=True, text=True,
                         env={"PYTHONPATH": str(src), "SYSTEMROOT": "C:\\Windows"})
    assert out.returncode == 0, out.stderr


def test_registry_covers_all_documented_datasets():
    assert set(BUILDERS) == {"ffpp", "asvspoof2019_la", "forensynths_progan_val", "forensynths_progan_test",
                             "forensynths_cnn_subset", "diffusion_test", "celebdf_v2", "celebdf_pp", "dfd",
                             "in_the_wild", "lavdf", "fakeavceleb", "wilddeepfake"}


# ---------------------------------------------------------------- A7 on the real files

def test_A7_ffpp_c23_documented_counts():
    if not (REAL_ROOT / "ffpp").is_dir():
        pytest.skip("A7 FF++ SKIPPED: data/ffpp does not exist")
    recs, rep = ffpp.build(REAL_ROOT)
    if not rep["complete"]:
        pytest.skip(
            f"A7 FF++ SKIPPED: data incomplete, found {rep['originals_found']} originals "
            f"(documented {ffpp.EXPECTED_ORIGINALS}) and {rep['manipulated_found']} manipulated "
            f"(documented {ffpp.EXPECTED_PER_METHOD} each)"
        )
    assert validate_records(recs) == []
    assert sum(r.label == 0 for r in recs) == 1000
    assert sum(r.label == 1 for r in recs) == 4000
    for m in ffpp.METHODS:
        assert sum(r.method == m for r in recs) == 1000
    assert rep["components_spanning_count"] == 0
    assert rep["ids_without_official_split"] == []


def test_A7_asvspoof2019_la_documented_counts():
    if not (REAL_ROOT / "asvspoof2019_la" / "LA" / "ASVspoof2019_LA_cm_protocols").is_dir():
        pytest.skip("A7 ASVspoof SKIPPED: data/asvspoof2019_la/LA is not extracted yet")
    recs, rep = asvspoof2019_la.build(REAL_ROOT)
    assert validate_records(recs) == []
    assert len(recs) == 121461
    assert {k: v["rows"] for k, v in rep["partitions"].items()} == {"train": 25380, "dev": 24844, "eval": 71237}
    assert all(v["listed_files_missing_on_disk"] == 0 for v in rep["partitions"].values())
    # documented as speaker-disjoint partitions; this checks it on the real protocol files
    assert rep["speakers_in_more_than_one_partition_count"] == 0
    assert rep["complete"] is True


@pytest.mark.parametrize("key", ["forensynths_progan_val", "forensynths_progan_test"])
def test_A7_progan_documented_counts(key):
    folder = "progan_val" if key.endswith("val") else "progan_test"
    if not (REAL_ROOT / "forensynths" / folder).is_dir():
        pytest.skip(f"A7 ProGAN SKIPPED: data/forensynths/{folder} missing")
    recs, rep = BUILDERS[key](REAL_ROOT)
    if len(recs) < 8000:
        pytest.skip(f"A7 ProGAN SKIPPED: {key} incomplete, {len(recs)} of 8000 images")
    assert validate_records(recs) == []
    assert sum(r.label == 0 for r in recs) == 4000 and sum(r.label == 1 for r in recs) == 4000
    assert rep["classes"] == 20 and rep["per_class_counts_unique"] == [(200, 200)]


def test_real_cnn_and_diffusion_structure():
    base = REAL_ROOT / "forensynths" / "cnn_synth_subset"
    if not base.is_dir():
        pytest.skip("cnn_synth_subset missing")
    recs, rep = forensynths.build_cnn_subset(REAL_ROOT)
    if len(recs) < 56003:
        pytest.skip(f"cnn subset incomplete: {len(recs)} of 56003 members listed in DATA_LOG.md")
    assert set(rep["generators"]) == {"biggan", "gaugan", "stylegan", "stylegan2", "whichfaceisreal", "deepfake",
                                      "stargan", "cyclegan"}
    assert len(recs) == 56003
    if not (REAL_ROOT / "diffusion_test").is_dir():
        pytest.skip("diffusion_test missing")
    drecs, drep = forensynths.build_diffusion_test(REAL_ROOT)
    assert drep["complete"] is True and len(drecs) == 10000


def test_celebdf_v2_real_files_counts():
    if not (REAL_ROOT / "celebdf_v2" / "Celeb-synthesis").is_dir():
        pytest.skip("Celeb-DF v2 SKIPPED: data/celebdf_v2 not present")
    recs, rep = celebdf.build(REAL_ROOT)
    if not rep["complete"]:
        pytest.skip(f"Celeb-DF v2 SKIPPED: incomplete, found {rep['found']}")
    assert validate_records(recs) == []
    assert rep["found"] == {"Celeb-real": 590, "YouTube-real": 300, "Celeb-synthesis": 5639}
    assert sum(r.official_split == "test" for r in recs) == 518
    assert rep["unparsed_or_ignored_files"] == []


def test_ffpp_complete_synthetic_tree_with_the_real_official_split_files(tmp_path):
    """Empty files named like a complete FF++ c23 download, laid out from the real official pair lists.

    Shows what the A7 FF++ test will see when the real download is complete (it does not replace it)."""
    src = REAL_ROOT / "_manifests" / "ffpp_official_splits"
    if not all((src / f"{n}.json").exists() for n in ("train", "val", "test")):
        pytest.skip("official FF++ split files not stored under data/_manifests/ffpp_official_splits")
    official = {n: json.loads((src / f"{n}.json").read_text()) for n in ("train", "val", "test")}
    pairs = [tuple(p) for v in official.values() for p in v]
    write_ffpp_fixture(tmp_path, pairs, methods=ffpp.METHODS, official=official)
    recs, rep = ffpp.build(tmp_path)
    assert rep["complete"] is True
    assert sum(r.label == 0 for r in recs) == 1000 and sum(r.label == 1 for r in recs) == 4000
    assert rep["groups"] == 500 and rep["components_spanning_count"] == 0
    assert rep["ids_without_official_split"] == []
