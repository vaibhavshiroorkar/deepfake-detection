"""D-026: test_x samples whose content equals a train, val or test_in sample are dropped."""

import json
from pathlib import Path

import pytest

from data_test_helpers import make_record, touch
from dfd.data import dedup, splits
from dfd.data.check_leakage import main as check_main
from dfd.data.hashing import HashCache
from dfd.data.leakage import check
from dfd.data.manifest import default_data_root, read_manifest, write_manifest

REAL_ROOT = default_data_root()


def _fixture(tmp_path):
    """dev dataset 'dev' (train/val/test_in) and test_x dataset 'tx'; tx/s1 and tx/s2 copy dev files."""
    root = tmp_path / "data"
    dev = [make_record(i, f"g{i}", i % 2, dataset="dev", family="fa", path=f"dev/f{i}.png") for i in range(4)]
    tx = [make_record(i, f"h{i}", i % 2, dataset="tx", family="fb", path=f"tx/f{i}.png") for i in range(4)]
    for r in dev + tx:
        touch(root / r.path, f"content-{r.dataset}-{r.sample_id}".encode())
    touch(root / tx[1].path, (root / dev[0].path).read_bytes())  # equals a train sample
    touch(root / tx[2].path, (root / dev[3].path).read_bytes())  # equals a test_in sample
    touch(root / tx[3].path, (root / tx[0].path).read_bytes())   # inside test_x only: kept
    asg_dev = {dev[0].sample_id: "train", dev[1].sample_id: "train", dev[2].sample_id: "val",
               dev[3].sample_id: "test_in"}
    asg_tx = {r.sample_id: "test_x" for r in tx}
    return root, dev, tx, asg_dev, asg_tx


def test_find_exclusions(tmp_path):
    root, dev, tx, asg_dev, asg_tx = _fixture(tmp_path)
    ex = dedup.find_exclusions(dev + tx, {**asg_dev, **asg_tx}, root)
    assert sorted(e["sample_id"] for e in ex) == sorted([tx[1].sample_id, tx[2].sample_id])
    by = {e["sample_id"]: e for e in ex}
    assert by[tx[1].sample_id]["matches"] == dev[0].sample_id
    assert by[tx[2].sample_id]["matches"] == dev[3].sample_id
    assert "D-026" in ex[0]["reason"]


def test_leakage_fails_before_and_passes_after_dedup(tmp_path):
    root, dev, tx, asg_dev, asg_tx = _fixture(tmp_path)
    recs = dev + tx
    allasg = {**asg_dev, **asg_tx}
    assert not check(recs, allasg, data_root=root, hash_files=True).ok
    ex = dedup.find_exclusions(recs, allasg, root)
    kept = dedup.apply_exclusions(allasg, {e["sample_id"] for e in ex})
    rep = check(recs, kept, data_root=root, hash_files=True)
    assert rep.ok
    assert len(kept) == len(allasg) - 2
    # a dev sample is never removed
    assert all(r.sample_id in kept for r in dev)


def test_dedup_cli_writes_new_split_and_exclusions(tmp_path, capsys):
    root, dev, tx, asg_dev, asg_tx = _fixture(tmp_path)
    md, mt = tmp_path / "dev.manifest.csv", tmp_path / "tx.manifest.csv"
    write_manifest(md, dev)
    write_manifest(mt, tx)
    out = tmp_path / "splits"
    _, sd = splits.write_split(dev, asg_dev, out, "dev", {})
    _, st = splits.write_split(tx, asg_tx, out, "tx", {})
    args = ["--manifest", str(md), str(mt), "--split", str(sd), str(st), "--name", "demo",
            "--data-root", str(root), "--out-dir", str(out)]
    assert dedup.main(args) == 0
    summary = json.loads((out / "demo-exclusions.json").read_text())
    assert summary["excluded_total"] == 2 and summary["excluded_by_dataset"] == {"tx": 2}
    assert list(summary["new_split_ids"]) == [st.stem]  # only the file that lost rows is rewritten
    new_csv = out / f"{summary['new_split_ids'][st.stem]}.csv"
    assert len(splits.read_split(new_csv)) == 2
    side = json.loads(new_csv.with_suffix(".json").read_text())
    assert side["parameters"]["dedup_of"] == st.stem and side["parameters"]["dropped"] == 2
    lines = (out / "demo-exclusions.csv").read_text().splitlines()
    assert len(lines) == 3 and lines[0].startswith("sample_id,dataset,group_id,sha256")
    # the leakage check with hashing passes on the new split files and fails on the old ones
    assert check_main(["--manifest", str(md), str(mt), "--split", str(sd), str(new_csv), "--hash-files",
                       "--data-root", str(root)]) == 0
    assert check_main(["--manifest", str(md), str(mt), "--split", str(sd), str(st), "--hash-files",
                       "--data-root", str(root)]) == 1
    assert (root / "_manifests" / "content_hash_cache.json").exists()


def test_missing_file_is_an_input_error(tmp_path):
    root, dev, tx, asg_dev, asg_tx = _fixture(tmp_path)
    (root / tx[0].path).unlink()
    with pytest.raises(FileNotFoundError):
        dedup.find_exclusions(dev + tx, {**asg_dev, **asg_tx}, root)


def test_hash_cache_reuses_unchanged_files(tmp_path):
    f = touch(tmp_path / "a.bin", b"abc")
    c = HashCache(tmp_path / "cache.json")
    h1 = c.sha256(f, "a.bin")
    assert (c.hits, c.misses) == (0, 1)
    c.save()
    c2 = HashCache(tmp_path / "cache.json")
    assert c2.sha256(f, "a.bin") == h1 and (c2.hits, c2.misses) == (1, 0)
    f.write_bytes(b"abcd")  # size changed: recomputed
    assert c2.sha256(f, "a.bin") != h1 and c2.misses == 1


def test_real_image_sets_pass_hash_check_after_dedup():
    """Runs when the image sets and the dedup output exist; skipped with a message otherwise."""
    names = ["forensynths_progan_val", "forensynths_progan_test", "forensynths_cnn_subset", "diffusion_test"]
    mans = [REAL_ROOT / "_manifests" / f"{n}.manifest.csv" for n in names]
    ex = REAL_ROOT / "_splits" / "images-exclusions.json"
    if not all(m.exists() for m in mans) or not ex.exists():
        pytest.skip("real dedup output (data/_splits/images-exclusions.json) not produced yet")
    summary = json.loads(ex.read_text())
    chosen = []
    for key in ("progan_val", "progan_test", "cnn_subset", "diffusion_test"):
        cands = sorted((REAL_ROOT / "_splits").glob(f"{key}-*.csv"))
        if len(cands) != 1:
            pytest.skip(f"expected exactly one split file for {key}, found {len(cands)}")
        chosen.append(cands[0])
    recs = [r for m in mans for r in read_manifest(m)]
    asg = {}
    for c in chosen:
        asg.update(splits.read_split(c))
    rep = check(recs, asg, data_root=REAL_ROOT, hash_files=True, cache=HashCache(REAL_ROOT / "_manifests" / "content_hash_cache.json"))
    assert rep.ok, rep.text()
    assert summary["excluded_total"] > 0
