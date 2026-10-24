from dataclasses import replace

import pytest

from data_test_helpers import make_record, synthetic_records
from dfd.data.manifest import COLUMNS, ManifestError, read_manifest, validate_records, validate_record, write_manifest


def test_roundtrip(tmp_path):
    recs = synthetic_records(3, 4)
    recs[0] = replace(recs[0], notes='has, comma and "quote"')
    p = tmp_path / "m.csv"
    assert write_manifest(p, recs) == 12
    back = read_manifest(p)
    assert back == recs
    header = p.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == list(COLUMNS)


def test_validation_rules():
    ok = make_record(1, "g", 0)
    assert validate_record(ok) == []
    assert validate_record(replace(ok, label=2))
    assert validate_record(replace(ok, modality="text"))
    assert validate_record(replace(ok, group_basis="vibes"))
    assert validate_record(replace(ok, method="gen"))  # label 0 needs method real
    assert validate_record(make_record(2, "g", 1, method="real"))  # label 1 cannot be real
    assert validate_record(replace(ok, path="C:/abs/x.png"))
    assert validate_record(replace(ok, path="a\\b.png"))
    assert validate_record(replace(ok, path="../x.png"))
    assert validate_record(replace(ok, sample_id="other:s1"))  # must start with dataset
    assert validate_record(replace(ok, group_id="other:g"))
    assert validate_record(replace(ok, identity_known=True))  # needs group_basis identity
    assert validate_record(replace(ok, identity_known=True, group_basis="identity")) == []


def test_duplicate_sample_id_detected_and_not_written(tmp_path):
    a = make_record(1, "g", 0)
    assert any("duplicate" in p for p in validate_records([a, a]))
    with pytest.raises(ManifestError):
        write_manifest(tmp_path / "m.csv", [a, a])
    assert not (tmp_path / "m.csv").exists()


def test_reader_rejects_bad_header_and_values(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ManifestError):
        read_manifest(p)
    good = tmp_path / "g.csv"
    write_manifest(good, [make_record(1, "g", 0)])
    text = good.read_text(encoding="utf-8").replace(",0,real,", ",zero,real,")
    bad = tmp_path / "bad2.csv"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(ManifestError):
        read_manifest(bad)
