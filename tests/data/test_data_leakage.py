"""Acceptance check A6 and the other leakage checks."""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from data_test_helpers import make_record, synthetic_records, touch
from dfd.data import splits
from dfd.data.check_leakage import main as check_main
from dfd.data.leakage import check, identity_overlap, identity_set
from dfd.data.manifest import write_manifest

SRC = Path(__file__).resolve().parents[2] / "src"


def _write(tmp_path, recs, assignment, name="x"):
    m = tmp_path / f"{name}.manifest.csv"
    write_manifest(m, recs)
    sid, sp = splits.write_split(recs, assignment, tmp_path / "splits", name, {})
    return m, sp


def _clean(tmp_path):
    """A clean pair of datasets: 'train_ds' split train/val/test_in and 'x_ds' fully test_x."""
    a = synthetic_records(30, 6, dataset="trn", family="famA")
    b = synthetic_records(5, 6, dataset="xds", family="famB")
    asg = splits.assign_groups(a, {"train": 0.6, "val": 0.2, "test_in": 0.2}, seed=1)
    asg.update({r.sample_id: "test_x" for r in b})
    return a + b, asg


def test_A6_clean_splits_exit_zero(tmp_path, capsys):
    recs, asg = _clean(tmp_path)
    m, s = _write(tmp_path, recs, asg)
    assert check_main(["--manifest", str(m), "--split", str(s)]) == 0
    assert "PASS" in capsys.readouterr().out
    assert check(recs, asg).ok


def test_A6_group_overlap_exits_nonzero(tmp_path, capsys):
    recs, asg = _clean(tmp_path)
    victim = next(r for r in recs if asg[r.sample_id] == "train")
    other = next(r for r in recs if r.group_id == victim.group_id and r.sample_id != victim.sample_id)
    asg[other.sample_id] = "val"  # one sample of a train group manufactured into val
    m, s = _write(tmp_path, recs, asg)
    assert check_main(["--manifest", str(m), "--split", str(s)]) == 1
    out = capsys.readouterr().out
    assert "GROUP_OVERLAP" in out and victim.group_id in out and "FAIL" in out


def test_A6_cross_family_leak_exits_nonzero(tmp_path, capsys):
    recs, asg = _clean(tmp_path)
    # same family in train and in test_x, different groups and datasets
    recs = recs + [make_record(900 + i, f"x{i}", i % 2, family="famA", dataset="other") for i in range(4)]
    asg.update({r.sample_id: "test_x" for r in recs if r.dataset == "other"})
    m, s = _write(tmp_path, recs, asg)
    assert check_main(["--manifest", str(m), "--split", str(s)]) == 1
    assert "FAMILY_LEAK" in capsys.readouterr().out
    # test_in plus test_x for one family is not a leak
    recs2 = [r for r in recs if r.dataset != "other"]
    assert check(recs2, {k: v for k, v in asg.items() if k in {r.sample_id for r in recs2}}).ok


def test_merge_family_flag(tmp_path, capsys):
    recs = synthetic_records(10, 4, dataset="d1", family="lavdf") + synthetic_records(3, 4, dataset="d2", family="fakeavceleb")
    asg = splits.assign_groups(recs[:40], {"train": 0.7, "val": 0.3}, seed=2)
    asg.update({r.sample_id: "test_x" for r in recs[40:]})
    m, s = _write(tmp_path, recs, asg)
    assert check_main(["--manifest", str(m), "--split", str(s)]) == 0  # warns: no identities readable
    assert "overlap NOT checked" in capsys.readouterr().out
    assert check_main(["--manifest", str(m), "--split", str(s), "--merge-family", "lavdf,fakeavceleb"]) == 1
    assert "FAMILY_LEAK" in capsys.readouterr().out


def test_duplicate_sample_ids_across_manifests(tmp_path):
    recs, asg = _clean(tmp_path)
    m1, s = _write(tmp_path, recs, asg, "a")
    m2 = tmp_path / "b.manifest.csv"
    write_manifest(m2, [recs[0]])
    assert check_main(["--manifest", str(m1), str(m2), "--split", str(s)]) == 1


def test_duplicate_sample_id_in_split_files_and_unknown_samples(tmp_path, capsys):
    recs, asg = _clean(tmp_path)
    m, s1 = _write(tmp_path, recs, asg, "a")
    s2 = tmp_path / "second.csv"
    s2.write_text("sample_id,split\n" + f"{recs[0].sample_id},val\nghost:1,train\n", encoding="utf-8")
    assert check_main(["--manifest", str(m), "--split", str(s1), str(s2)]) == 1
    out = capsys.readouterr().out
    assert "DUPLICATE_SAMPLE_ID" in out and "UNKNOWN_SAMPLE" in out


def test_duplicate_file_content(tmp_path):
    root = tmp_path / "data"
    recs = [make_record(i, f"g{i}", i % 2, path=f"ds/f{i}.png") for i in range(4)]
    for r in recs:
        touch(root / r.path, f"unique{r.sample_id}".encode())
    asg = {recs[0].sample_id: "train", recs[1].sample_id: "train", recs[2].sample_id: "val",
           recs[3].sample_id: "test_in"}
    assert check(recs, asg, data_root=root, hash_files=True).ok
    # same bytes in train and val: fails
    touch(root / recs[2].path, b"uniqueds:s0")
    rep = check(recs, asg, data_root=root, hash_files=True)
    assert not rep.ok and any("DUPLICATE_CONTENT" in e for e in rep.errors)
    # same bytes inside one split: warning, or failure with strict
    touch(root / recs[2].path, b"unique-restored")
    touch(root / recs[1].path, b"uniqueds:s0")
    rep = check(recs, asg, data_root=root, hash_files=True)
    assert rep.ok and rep.warnings
    assert not check(recs, asg, data_root=root, hash_files=True, strict_duplicates=True).ok
    # hashing off: nothing read, nothing flagged
    assert check(recs, asg, data_root=tmp_path / "nowhere").ok
    # a missing file cannot be verified: fails
    assert not check(recs, asg, data_root=tmp_path / "nowhere", hash_files=True).ok


def test_identity_helpers_and_voxceleb_overlap_check():
    lav = [make_record(1, "id00001", 0, dataset="lavdf", family="lavdf", notes="ids=id00001"),
           make_record(2, "id00002", 1, dataset="lavdf", family="lavdf", path="lavdf/id00002/v.mp4")]
    fab = [make_record(3, "id00002", 1, dataset="fab", family="fakeavceleb", path="fab/id00002/x.mp4"),
           make_record(4, "id00009", 0, dataset="fab", family="fakeavceleb")]
    assert identity_set(lav) == {"id00001", "id00002"}
    res = identity_overlap(lav, fab)
    assert res["shared"] == ["id00002"] and res["shared_count"] == 1
    assert identity_overlap(lav[:1], fab[1:])["shared_count"] == 0
    # inside check(): LAV-DF in train and FakeAVCeleb in test_x with a shared identity fails
    recs = lav + fab
    asg = {lav[0].sample_id: "train", lav[1].sample_id: "train", fab[0].sample_id: "test_x",
           fab[1].sample_id: "test_x"}
    rep = check(recs, asg)
    assert not rep.ok and any("IDENTITY_OVERLAP" in e for e in rep.errors)
    # remove the shared identity: passes
    ok = check(lav[:1] + fab[1:], {lav[0].sample_id: "train", fab[1].sample_id: "test_x"})
    assert ok.ok and any("none" in i for i in ok.info)


def test_manifest_rows_without_split_are_a_warning(tmp_path):
    recs, asg = _clean(tmp_path)
    asg.pop(recs[0].sample_id)
    rep = check(recs, asg)
    assert rep.ok and any("no split" in w for w in rep.warnings)


def test_cli_as_subprocess_exit_codes(tmp_path):
    recs, asg = _clean(tmp_path)
    m, s = _write(tmp_path, recs, asg)
    env = {"PYTHONPATH": str(SRC), "SYSTEMROOT": "C:\\Windows"}
    cmd = [sys.executable, "-m", "dfd.data.check_leakage", "--manifest", str(m), "--split", str(s)]
    assert subprocess.run(cmd, env=env, capture_output=True).returncode == 0
    victim = next(r for r in recs if asg[r.sample_id] == "train")
    mate = next(r for r in recs if r.group_id == victim.group_id and r.sample_id != victim.sample_id)
    asg[mate.sample_id] = "test_in"
    m2, s2 = _write(tmp_path, recs, asg, "bad")
    cmd2 = [sys.executable, "-m", "dfd.data.check_leakage", "--manifest", str(m2), "--split", str(s2)]
    r = subprocess.run(cmd2, env=env, capture_output=True, text=True)
    assert r.returncode == 1 and "GROUP_OVERLAP" in r.stdout
    assert subprocess.run(cmd + ["--manifest"], env=env, capture_output=True).returncode == 2  # argparse error


def test_input_error_exit_code(tmp_path):
    assert check_main(["--manifest", str(tmp_path / "missing.csv"), "--split", str(tmp_path / "s.csv")]) == 2
