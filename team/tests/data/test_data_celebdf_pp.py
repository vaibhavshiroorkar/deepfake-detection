from dfd.data import identity_exclusion as IE
from dfd.data.builders import celebdf
from dfd.data.manifest import Record, write_manifest


def make_tree(root):
    base = root / "celebdf_pp"
    files = ["Celeb-real/id3_0001.mp4", "Celeb-real/id12_0004.mp4", "YouTube-real/00170.mp4",
             "Celeb-synthesis/FaceSwap/BlendFace/id3_id28_0001.mp4", "Celeb-synthesis/FaceSwap/BlendFace/id5_id12_0002.mp4",
             "Celeb-synthesis/FaceReenact/DaGAN/id7_id9_0003.mp4",
             "Celeb-synthesis/TalkingFace/SadTalker/id12_0002_test_id08701_8NPsTNx-erA.mp4",
             "Celeb-synthesis/TalkingFace/SadTalker/id4_0001_test_id03524_Lt7XKpDEmh0.mp4"]
    for f in files:
        p = base / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    listing = "\n".join([("1 " if f.split("/")[0].endswith("real") else "0 ") + f for f in files] + ["0 Celeb-synthesis/FaceSwap/Celeb-DF-v2/id1_id0_0007.mp4"])
    (base / "List_of_testing_videos.txt").write_text(listing, encoding="utf-8")
    return files


def test_name_parsers():
    assert celebdf.parse_pp_fake_name("FaceSwap", "id3_id28_0001.mp4") == {"target": "id3", "source": "id28"}
    assert celebdf.parse_pp_fake_name("TalkingFace", "id29_0002_test_id08701_8NPsTNx-erA.mp4") == {"target": "id29", "source": "id08701"}
    import pytest
    with pytest.raises(ValueError):
        celebdf.parse_pp_fake_name("TalkingFace", "id3_id28_0001.mp4")


def test_builder_groups_methods_and_official_split(tmp_path):
    make_tree(tmp_path)
    recs, rep = celebdf.build_pp(tmp_path)
    by = {r.sample_id.split(":", 1)[1]: r for r in recs}
    assert len(recs) == 8 and rep["unparsed_or_ignored_files"] == [] and rep["rows"] == 8
    assert all(r.dataset == "celebdf_pp" and r.family == "celebdf" and r.official_split == "test" for r in recs)
    f = by["Celeb-synthesis/FaceSwap/BlendFace/id3_id28_0001.mp4"]
    assert (f.label, f.method, f.group_id, f.notes) == (1, "BlendFace", "celebdf_pp:id3", "ids=id3|id28")
    t = by["Celeb-synthesis/TalkingFace/SadTalker/id12_0002_test_id08701_8NPsTNx-erA.mp4"]
    assert t.group_id == "celebdf_pp:id12" and t.notes == "ids=id12|id08701" and t.method == "SadTalker"
    y = by["YouTube-real/00170.mp4"]
    assert y.label == 0 and y.group_basis == "source_video" and y.identity_known is False
    assert by["Celeb-real/id3_0001.mp4"].group_id == "celebdf_pp:id3" and by["Celeb-real/id3_0001.mp4"].label == 0
    assert rep["complete"] is False                                   # counts differ from the real archive, as expected here
    assert rep["found"]["FaceSwap/BlendFace"] == 2 and rep["found"]["TalkingFace/SadTalker"] == 2


def _rec(sid, group, notes, label=1, method="m"):
    return Record(sample_id=sid, dataset="celebdf_pp", family="celebdf", path=f"celebdf_pp/{sid}.mp4", modality="video", label=label, method=method,
                  group_id=group, group_basis="identity", identity_known=True, official_split="test", notes=notes)


def test_identity_exclusion_rule_target_and_celebrity_source_but_not_voxceleb_ids():
    dev = {"id3", "id12"}
    recs = [_rec("a", "celebdf_pp:id3", "ids=id3|id28"),            # target in dev: excluded
            _rec("b", "celebdf_pp:id5", "ids=id5|id12"),            # celebrity source in dev: excluded
            _rec("c", "celebdf_pp:id7", "ids=id7|id9"),             # clean
            _rec("d", "celebdf_pp:id4", "ids=id4|id08701"),         # VoxCeleb speaker id is not a celebrity of this family: kept
            _rec("e", "celebdf_pp:id12", "ids=id12", label=0, method="real")]
    asg, rows = IE.exclude(recs, dev)
    assert set(asg) == {"c", "d"} and all(v == "test_x" for v in asg.values())
    by = {r["sample_id"]: r for r in rows}
    assert by["a"]["matched_target"] == "id3" and by["b"]["matched_source"] == "id12" and by["b"]["matched_target"] == ""
    assert by["e"]["label"] == 0
    assert IE.identity_of_group("celebdf_v2:youtube:00001") is None and IE.identity_of_group("celebdf_v2:id41") == "id41"


def test_dev_identities_come_from_the_reference_dev_x_only():
    ref = [_rec("r1", "celebdf_v2:id0", "ids=id0"), _rec("r2", "celebdf_v2:id1", "ids=id1"), _rec("r3", "celebdf_v2:youtube:5", "")]
    assert IE.dev_identities(ref, {"r1": "dev_x", "r2": "test_x", "r3": "dev_x"}) == {"id0"}


def test_cli_writes_split_and_exclusions(tmp_path, capsys):
    from dfd.data.splits import read_split
    recs = [_rec("celebdf_pp:a", "celebdf_pp:id3", "ids=id3|id28"), _rec("celebdf_pp:c", "celebdf_pp:id7", "ids=id7|id9")]
    recs = [Record(**{**r.__dict__, "path": f"celebdf_pp/{r.sample_id.split(':')[1]}.mp4"}) for r in recs]
    ref = [Record(sample_id="celebdf_v2:r", dataset="celebdf_v2", family="celebdf", path="celebdf_v2/r.mp4", modality="video", label=1, method="m",
                  group_id="celebdf_v2:id3", group_basis="identity", identity_known=True, official_split="", notes="ids=id3|id1")]
    for r, body in zip(recs + ref, (b"1", b"2", b"3")):
        f = tmp_path / "data" / r.path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(body)
    write_manifest(tmp_path / "pp.csv", recs)
    write_manifest(tmp_path / "v2.csv", ref)
    (tmp_path / "v2split.csv").write_text("sample_id,split\ncelebdf_v2:r,dev_x\n")
    assert IE.main(["--manifest", str(tmp_path / "pp.csv"), "--reference-manifest", str(tmp_path / "v2.csv"), "--reference-split",
                    str(tmp_path / "v2split.csv"), "--name", "pp", "--out-dir", str(tmp_path / "out"), "--data-root", str(tmp_path / "data")]) == 0
    out = tmp_path / "out"
    split_csv = next(out.glob("pp-*.csv"))
    assert read_split(split_csv) == {"celebdf_pp:c": "test_x"}
    assert "celebdf_pp:a" in (out / "pp.exclusions.csv").read_text()


def test_duplicate_content_of_the_reference_is_dropped_once(tmp_path):
    from dfd.data.hashing import HashCache
    root = tmp_path / "data"
    (root / "celebdf_pp").mkdir(parents=True)
    (root / "celebdf_v2").mkdir()
    (root / "celebdf_v2" / "r.mp4").write_bytes(b"same content")
    (root / "celebdf_pp" / "a.mp4").write_bytes(b"same content")      # byte-identical real video
    (root / "celebdf_pp" / "b.mp4").write_bytes(b"different")
    pp = [_rec("celebdf_pp:a", "celebdf_pp:id9", "ids=id9", label=0, method="real"), _rec("celebdf_pp:b", "celebdf_pp:id8", "ids=id8|id7")]
    pp[0] = Record(**{**pp[0].__dict__, "path": "celebdf_pp/a.mp4"})
    pp[1] = Record(**{**pp[1].__dict__, "path": "celebdf_pp/b.mp4"})
    ref = [Record(sample_id="celebdf_v2:r", dataset="celebdf_v2", family="celebdf", path="celebdf_v2/r.mp4", modality="video", label=0, method="real",
                  group_id="celebdf_v2:id9", group_basis="identity", identity_known=True, official_split="", notes="ids=id9")]
    rows = IE.drop_duplicates(pp, {"celebdf_pp:a": "test_x", "celebdf_pp:b": "test_x"}, ref, {"celebdf_v2:r": "test_x"}, root, HashCache(None))
    assert [r["sample_id"] for r in rows] == ["celebdf_pp:a"] and rows[0]["matches"] == "celebdf_v2:r"
    assert IE.drop_duplicates(pp, {"celebdf_pp:a": "test_x"}, ref, {"celebdf_v2:r": "train"}, root, HashCache(None)) == []   # only dev_x and test_x of the reference count
