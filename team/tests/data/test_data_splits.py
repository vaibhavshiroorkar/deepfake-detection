import hashlib
import json
from collections import defaultdict
from dataclasses import replace

import pytest

from data_test_helpers import make_record, synthetic_records
from dfd.data import splits
from dfd.data.manifest import read_manifest, write_manifest

FR = {"train": 0.7, "val": 0.15, "test_in": 0.15}


def _by_group(recs, asg):
    out = defaultdict(set)
    for r in recs:
        out[r.group_id].add(asg[r.sample_id])
    return out


def test_group_split_never_splits_a_group_and_is_deterministic():
    recs = synthetic_records(40, 6)
    a = splits.assign_groups(recs, FR, seed=7)
    b = splits.assign_groups(list(reversed(recs)), FR, seed=7)
    assert a == b  # independent of row order
    assert all(len(v) == 1 for v in _by_group(recs, a).values())
    c = splits.assign_groups(recs, FR, seed=8)
    assert a != c  # the seed matters
    counts = defaultdict(int)
    for v in a.values():
        counts[v] += 1
    assert set(counts) == {"train", "val", "test_in"}
    assert abs(counts["train"] / len(recs) - 0.7) < 0.08


def test_both_classes_in_every_part_even_with_class_pure_groups():
    recs = []
    i = 0
    for g in range(12):  # six real-only and six fake-only groups
        for _ in range(5):
            recs.append(make_record(i, f"g{g}", 0 if g < 6 else 1))
            i += 1
    asg = splits.assign_groups(recs, {"train": 0.5, "val": 0.25, "test_in": 0.25}, seed=3)
    for part in ("train", "val", "test_in"):
        labels = {r.label for r in recs if asg[r.sample_id] == part}
        assert labels == {0, 1}, part
    assert all(len(v) == 1 for v in _by_group(recs, asg).values())


def test_official_mapping_and_group_conflict():
    recs = [make_record(i, f"g{i}", i % 2, official=o)
            for i, o in enumerate(["train", "dev", "eval", "val", "test"])]
    asg = splits.assign_official(recs)
    assert [asg[r.sample_id] for r in recs] == ["train", "val", "test_in", "val", "test_in"]
    bad = [make_record(0, "g", 0, official="train"), make_record(1, "g", 1, official="eval")]
    with pytest.raises(splits.SplitError, match="spans official splits"):
        splits.assign_official(bad)
    with pytest.raises(splits.SplitError):
        splits.assign_official([make_record(0, "g", 0, official="")])


def test_auto_mode_and_fixed_mode():
    official = [make_record(i, f"g{i}", i % 2, official="train") for i in range(4)]
    assert set(splits.assign(official, "auto").values()) == {"train"}
    plain = synthetic_records(10, 4)
    assert set(splits.assign(plain, "auto", seed=1, fractions={"train": 0.8, "val": 0.2}).values()) == {"train", "val"}
    mixed = [replace(plain[0], official_split="train")] + plain[1:]
    with pytest.raises(splits.SplitError, match="mixed"):
        splits.assign(mixed, "auto", fractions=FR)
    assert set(splits.assign(plain, "fixed", fixed="test_x").values()) == {"test_x"}
    with pytest.raises(splits.SplitError):
        splits.assign(plain, "fixed")
    with pytest.raises(splits.SplitError):
        splits.assign(plain, "group")


def test_write_split_hash_id_and_sidecar(tmp_path):
    recs = synthetic_records(20, 4)
    m = tmp_path / "m.csv"
    write_manifest(m, recs)
    asg = splits.assign_groups(recs, FR, seed=11)
    sid, p = splits.write_split(recs, asg, tmp_path / "out", "demo", {"seed": 11}, [m])
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    assert sid == f"demo-{digest[:12]}" and p.name == f"{sid}.csv"
    side = json.loads((tmp_path / "out" / f"{sid}.json").read_text())
    assert side["csv_sha256"] == digest and side["parameters"] == {"seed": 11}
    assert side["rows"] == 80 and sum(c["samples"] for c in side["counts"].values()) == 80
    assert side["manifests"][0]["sha256"] == hashlib.sha256(m.read_bytes()).hexdigest()
    assert splits.read_split(p) == asg
    # the same content gives the same id, different content a different one
    sid2, _ = splits.write_split(recs, asg, tmp_path / "out2", "demo", {"seed": 11}, [m])
    assert sid2 == sid
    other = splits.assign_groups(recs, FR, seed=12)
    sid3, _ = splits.write_split(recs, other, tmp_path / "out3", "demo", {"seed": 12}, [m])
    assert sid3 != sid


def test_read_split_rejects_bad_files(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("sample_id,split\na,train\na,val\n")
    with pytest.raises(splits.SplitError, match="duplicate"):
        splits.read_split(p)
    p.write_text("sample_id,split\na,holdout\n")
    with pytest.raises(splits.SplitError):
        splits.read_split(p)


def test_cli(tmp_path):
    recs = synthetic_records(10, 4)
    m = tmp_path / "m.csv"
    write_manifest(m, recs)
    rc = splits.main(["--manifest", str(m), "--name", "t", "--mode", "group", "--fractions", "train=0.8,val=0.2",
                      "--out-dir", str(tmp_path / "s")])
    assert rc == 0
    assert len(list((tmp_path / "s").glob("t-*.csv"))) == 1
    assert splits.main(["--manifest", str(m), "--name", "t", "--mode", "official", "--out-dir", str(tmp_path / "s")]) == 2
    assert read_manifest(m)  # manifest untouched
