from pathlib import Path

import pytest

from deepfake_detection.data.speech import (
    ASVSPOOF_PARTITIONS,
    build_asvspoof2019_la,
    build_in_the_wild,
    load_audio_manifest,
    parse_asvspoof_line,
    write_audio_manifest,
)


def _asvspoof_tree(root: Path) -> None:
    protocols = root / "LA" / "ASVspoof2019_LA_cm_protocols"
    protocols.mkdir(parents=True)
    for partition, (protocol, folder) in ASVSPOOF_PARTITIONS.items():
        flac = root / "LA" / folder / "flac"
        flac.mkdir(parents=True)
        lines = [
            f"LA_{partition}_s1 {partition}_0 - - bonafide",
            f"LA_{partition}_s1 {partition}_1 - A07 spoof",
        ]
        (protocols / protocol).write_text("\n".join(lines) + "\n", encoding="utf-8")
        for index in range(2):
            (flac / f"{partition}_{index}.flac").write_bytes(b"x")
    (root / "LA" / "ASVspoof2019_LA_dev" / "flac" / "enrol.flac").write_bytes(b"x")


def test_asvspoof_rows_come_from_protocol_only(tmp_path: Path) -> None:
    _asvspoof_tree(tmp_path)
    records, report = build_asvspoof2019_la(tmp_path)
    assert len(records) == 6
    assert {r.method for r in records} == {"real", "A07"}
    assert report["partitions"]["dev"]["unlisted_on_disk"] == 1
    assert report["speakers_in_two_partitions"] == 0
    assert report["complete"] is False  # fixture is far smaller than the real corpus


def test_asvspoof_rejects_spoof_without_attack() -> None:
    with pytest.raises(ValueError):
        parse_asvspoof_line("LA_0001 LA_T_1 - - spoof")
    assert parse_asvspoof_line("   ") is None


def test_in_the_wild_checks_meta_against_disk(tmp_path: Path) -> None:
    base = tmp_path / "release_in_the_wild"
    base.mkdir()
    (base / "meta.csv").write_text(
        "file,speaker,label\n0.wav,Alice,bona-fide\n1.wav,Alice,spoof\n",
        encoding="utf-8",
    )
    (base / "0.wav").write_bytes(b"x")
    with pytest.raises(ValueError, match="not on disk"):
        build_in_the_wild(tmp_path)
    (base / "1.wav").write_bytes(b"x")
    records, report = build_in_the_wild(tmp_path)
    assert [r.audio_fake for r in records] == [False, True]
    assert report["speakers_with_both_labels"] == 1


def test_audio_manifest_round_trip(tmp_path: Path) -> None:
    _asvspoof_tree(tmp_path)
    records, _ = build_asvspoof2019_la(tmp_path)
    path = tmp_path / "manifest.csv"
    write_audio_manifest(records, path)
    assert load_audio_manifest(path) == records
