"""D-035: split dev_x, the seeded slice rule and acceptance check A9 (data side)."""

import json
from dataclasses import replace

import pytest

from data_test_helpers import make_record, synthetic_records, touch
from dfd.data import dedup, splits
from dfd.data.check_leakage import main as check_main
from dfd.data.leakage import check
from dfd.data.manifest import default_data_root, read_manifest, write_manifest

REAL_ROOT = default_data_root()


def _by_group(recs, asg):
    out = {}
    for r in recs:
        out.setdefault(r.group_id, set()).add(asg[r.sample_id])
    return out


def _write(tmp_path, recs, asg, name="x"):
    m = tmp_path / f"{name}.manifest.csv"
    write_manifest(m, recs)
    _, sp = splits.write_split(recs, asg, tmp_path / "splits", name, {})
    return m, sp


def test_dev_x_is_a_split_name_and_readable(tmp_path):
    assert "dev_x" in splits.SPLIT_NAMES
    p = tmp_path / "s.csv"
    p.write_text("sample_id,split\na,dev_x\nb,test_x\n")
    assert splits.read_split(p) == {"a": "dev_x", "b": "test_x"}
    assert set(splits.assign(synthetic_records(4, 2), "fixed", fixed="dev_x").values()) == {"dev_x"}


def test_dev_slice_follows_the_D035_rule_exactly():
    import numpy as np

    recs = synthetic_records(20, 4)
    asg, drawn = splits.assign_dev_slice(recs, 5)
    names = sorted({r.group_id for r in recs})
    perm = np.random.default_rng(20261024).permutation(len(names))
    assert drawn == sorted(names[i] for i in perm[:5])
    assert {r.group_id for r in recs if asg[r.sample_id] == "dev_x"} == set(drawn)
    assert all(len(v) == 1 for v in _by_group(recs, asg).values())
    assert splits.assign_dev_slice(list(reversed(recs)), 5)[0] == asg  # independent of row order
    assert sum(v == "dev_x" for v in asg.values()) == 5 * 4  # every sample of a drawn group
    with pytest.raises(splits.SplitError):
        splits.assign_dev_slice(recs, 0)
    with pytest.raises(splits.SplitError):
        splits.assign_dev_slice(recs, 20)


def test_dev_slice_generator_unit_and_identity_only():
    recs, i = [], 0
    for gen, classes in (("a", ["x", "y"]), ("b", [None]), ("c", ["x"]), ("d", [None])):
        for cl in classes:
            local = gen if cl is None else f"{gen}:{cl}"
            for k in range(4):
                recs.append(make_record(i, local, k % 2, dataset="ds"))
                i += 1
    asg, drawn = splits.assign_dev_slice(recs, 2, unit="generator")
    gens = {r.group_id.split(":")[1] for r in recs if asg[r.sample_id] == "dev_x"}
    assert gens == set(drawn) and len(gens) == 2
    for r in recs:  # every class group of a drawn generator moved together
        assert (asg[r.sample_id] == "dev_x") == (r.group_id.split(":")[1] in gens)
    mixed = [replace(r, group_basis="identity", identity_known=True) if k < 16 else r for k, r in enumerate(recs)]
    asg2, drawn2 = splits.assign_dev_slice(mixed, 2, unit="group", identity_only=True)
    assert all(asg2[r.sample_id] == "test_x" for r in mixed if not r.identity_known)
    assert all(d in {r.group_id for r in mixed if r.identity_known} for d in drawn2)


def test_dev_slice_cli_records_parameters(tmp_path):
    m = tmp_path / "m.csv"
    write_manifest(m, synthetic_records(12, 4))
    out = tmp_path / "s"
    assert splits.main(["--manifest", str(m), "--name", "x", "--mode", "dev_slice", "--dev-k", "3",
                        "--out-dir", str(out)]) == 0
    prm = json.loads(next(out.glob("x-*.json")).read_text())
    assert prm["parameters"]["dev_k"] == 3 and prm["parameters"]["dev_seed"] == 20261024
    assert len(prm["parameters"]["dev_units"]) == 3 and prm["parameters"]["units_total"] == 12
    assert prm["counts"]["dev_x"]["groups"] == 3 and prm["counts"]["test_x"]["groups"] == 9
    assert splits.main(["--manifest", str(m), "--name", "x", "--mode", "dev_slice", "--out-dir", str(out)]) == 2


# ------------------------------------------------------------------ A9

def _devx_fixture():
    recs = synthetic_records(10, 4, dataset="cx", family="fam")
    asg, _ = splits.assign_dev_slice(recs, 3)
    return recs, asg


def test_A9_clean_dev_x_test_x_passes_and_same_family_is_allowed(tmp_path):
    recs, asg = _devx_fixture()
    m, s = _write(tmp_path, recs, asg)
    assert {"dev_x", "test_x"} <= set(asg.values())
    assert check_main(["--manifest", str(m), "--split", str(s)]) == 0
    assert check(recs, asg).ok  # one family in dev_x and test_x is not a leak


def test_A9_group_in_dev_x_and_test_x_fails(tmp_path, capsys):
    recs, asg = _devx_fixture()
    victim = next(r for r in recs if asg[r.sample_id] == "dev_x")
    mate = next(r for r in recs if r.group_id == victim.group_id and r.sample_id != victim.sample_id)
    asg[mate.sample_id] = "test_x"
    m, s = _write(tmp_path, recs, asg)
    assert check_main(["--manifest", str(m), "--split", str(s)]) == 1
    out = capsys.readouterr().out
    assert "GROUP_OVERLAP_DEV_X_TEST_X" in out and victim.group_id in out


def test_A9_content_in_dev_x_and_test_x_fails_and_dedup_drops_the_test_x_copy(tmp_path):
    root = tmp_path / "data"
    recs = [make_record(i, f"g{i}", i % 2, dataset="cx", path=f"cx/f{i}.png") for i in range(4)]
    for r in recs:
        touch(root / r.path, f"u{r.sample_id}".encode())
    touch(root / recs[3].path, (root / recs[0].path).read_bytes())  # test_x copy of a dev_x file
    asg = {recs[0].sample_id: "dev_x", recs[1].sample_id: "dev_x", recs[2].sample_id: "test_x",
           recs[3].sample_id: "test_x"}
    assert not check(recs, asg, data_root=root, hash_files=True).ok
    ex = dedup.find_exclusions(recs, asg, root)
    assert [e["sample_id"] for e in ex] == [recs[3].sample_id] and ex[0]["matches"] == recs[0].sample_id
    assert check(recs, dedup.apply_exclusions(asg, {recs[3].sample_id}), data_root=root, hash_files=True).ok


def test_A9_dev_x_never_triggers_the_family_rule():
    a = [make_record(i, f"g{i}", i % 2, dataset="a", family="famA") for i in range(4)]
    asg = {r.sample_id: ("train" if i < 2 else "dev_x") for i, r in enumerate(a)}
    assert check(a, asg).ok


# ------------------------------------------------------------------ real files

@pytest.mark.parametrize("name,k,total", [("celebdf_v2", 15, 59), ("in_the_wild", 12, 54), ("cnn_subset", 2, 8)])
def test_real_dev_x_slices_are_group_disjoint(name, k, total):
    key = {"cnn_subset": "forensynths_cnn_subset"}.get(name, name)
    man = REAL_ROOT / "_manifests" / f"{key}.manifest.csv"
    cands = sorted((REAL_ROOT / "_splits").glob(f"{name}-*.csv"))
    if not man.exists() or len(cands) != 1:
        pytest.skip(f"real {name} manifest or its single split file not present")
    prm = json.loads(cands[0].with_suffix(".json").read_text())["parameters"]
    if prm.get("mode") != "dev_slice" and "dedup_of" not in prm:
        pytest.skip(f"{name} split is not a dev slice")
    recs = read_manifest(man)
    asg = splits.read_split(cands[0])
    assert {"dev_x", "test_x"} <= set(asg.values())
    assert check([r for r in recs if r.sample_id in asg], asg).ok
    dev_units = {splits.unit_of(r, "generator" if name == "cnn_subset" else "group")
                 for r in recs if asg.get(r.sample_id) == "dev_x"}
    assert len(dev_units) == k


# ------------------------------------------------------------------ D-035 Amendment 1: merged draw units

def _gen_fixture(tmp_path):
    """Generators a, b share real content (b's reals are copies of a's); c and d are independent."""
    root = tmp_path / "data"
    recs, i = [], 0
    for gen in ("a", "b", "c", "d"):
        for label in (0, 1):
            for k in range(3):
                r = make_record(i, gen, label, dataset="ds", path=f"ds/{gen}/{label}/{k}.png")
                recs.append(r)
                touch(root / r.path, f"{gen}-{label}-{k}".encode())
                i += 1
    for k in range(3):  # b's reals are byte copies of a's reals
        touch(root / f"ds/b/0/{k}.png", f"a-0-{k}".encode())
    return root, recs


def test_units_sharing_real_content_are_merged_generically(tmp_path):
    root, recs = _gen_fixture(tmp_path)
    m = splits.merge_units_by_real_content(recs, "generator", root)
    assert m == {"a": "a+b", "b": "a+b", "c": "c", "d": "d"}
    # shared FAKE content alone does not merge units
    touch(root / "ds/d/1/0.png", (root / "ds/c/1/0.png").read_bytes())
    assert splits.merge_units_by_real_content(recs, "generator", root)["d"] == "d"


def test_merged_unit_moves_together_in_the_draw(tmp_path):
    root, recs = _gen_fixture(tmp_path)
    m = splits.merge_units_by_real_content(recs, "generator", root)
    for k in (1, 2):
        asg, drawn = splits.assign_dev_slice(recs, k, unit="generator", unit_map=m)
        gens_dev = {r.group_id.split(":")[1] for r in recs if asg[r.sample_id] == "dev_x"}
        assert ("a" in gens_dev) == ("b" in gens_dev)  # never one without the other
        assert len(drawn) == k
    # the unit names (sorted) feed the permutation: 3 units here
    import numpy as np

    names = sorted(set(m.values()))
    perm = np.random.default_rng(20261024).permutation(len(names))
    _, drawn = splits.assign_dev_slice(recs, 1, unit="generator", unit_map=m)
    assert drawn == [names[perm[0]]]


def test_merge_flag_is_recorded_by_the_cli(tmp_path):
    root, recs = _gen_fixture(tmp_path)
    man = tmp_path / "m.csv"
    write_manifest(man, recs)
    out = tmp_path / "s"
    rc = splits.main(["--manifest", str(man), "--name", "g", "--mode", "dev_slice", "--dev-k", "1",
                      "--dev-unit", "generator", "--dev-merge-real-content", "--data-root", str(root),
                      "--out-dir", str(out)])
    assert rc == 0
    prm = json.loads(next(out.glob("g-*.json")).read_text())["parameters"]
    assert prm["dev_merge_real_content"] is True and prm["units"] == ["a+b", "c", "d"]
    assert prm["units_total"] == 3
