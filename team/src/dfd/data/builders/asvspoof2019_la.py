"""ASVspoof 2019 LA manifest builder.

Layout (seen on the extracted archive):
  data/asvspoof2019_la/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{train.trn,dev.trl,eval.trl}.txt
  data/asvspoof2019_la/LA/ASVspoof2019_LA_{train,dev,eval}/flac/<FILE_ID>.flac

Protocol line: ``SPEAKER_ID FILE_ID - ATTACK_ID KEY`` with KEY ``bonafide`` or ``spoof``; ATTACK_ID is ``-``
for bona fide and A01..A19 for spoofs.

The CM protocols are the authority and the only source of rows. The dev and eval flac folders also hold
extra bona fide speaker-verification enrolment utterances (for example LA_D_A1045602) that appear in the
ASV protocols but not in the CM protocols; they are not part of the 24,844 and 71,237 CM entries. They are
counted in the report (``flac_files_not_in_cm_protocol``) and never put in the manifest.

Group = speaker id (identity_known true). The official partitions are documented as speaker-disjoint; the
report states what the real files show (``speakers_in_more_than_one_partition``).
Official split names are kept as ``train``, ``dev``, ``eval`` (dev is used as val, eval as test_in by splits.py).
"""

from __future__ import annotations

from pathlib import Path

from ..manifest import Record, rel_posix

KEY = "asvspoof2019_la"
FAMILY = "asvspoof2019_la"
PARTITIONS = {
    # official name: (protocol file, audio folder)
    "train": ("ASVspoof2019.LA.cm.train.trn.txt", "ASVspoof2019_LA_train"),
    "dev": ("ASVspoof2019.LA.cm.dev.trl.txt", "ASVspoof2019_LA_dev"),
    "eval": ("ASVspoof2019.LA.cm.eval.trl.txt", "ASVspoof2019_LA_eval"),
}
EXPECTED = {"train": 25380, "dev": 24844, "eval": 71237}


def parse_protocol_line(line: str) -> tuple[str, str, str, str] | None:
    """Return (speaker, file_id, attack, key) or None for a blank line. Raises ValueError if malformed."""
    parts = line.split()
    if not parts:
        return None
    if len(parts) != 5:
        raise ValueError(f"expected 5 fields, got {len(parts)}: {line!r}")
    speaker, file_id, _env, attack, key = parts
    if key not in ("bonafide", "spoof"):
        raise ValueError(f"KEY must be bonafide or spoof: {line!r}")
    if key == "spoof" and attack == "-":
        raise ValueError(f"spoof line without ATTACK_ID: {line!r}")
    return speaker, file_id, attack, key


def build(data_root: Path) -> tuple[list[Record], dict]:
    base = data_root / KEY
    la = base / "LA"
    if not la.is_dir():
        la = base  # tolerate an archive extracted one level higher
    proto_dir = la / "ASVspoof2019_LA_cm_protocols"
    if not proto_dir.is_dir():
        raise FileNotFoundError(f"protocol folder not found: {proto_dir}")

    records: list[Record] = []
    per_part: dict[str, dict] = {}
    speakers_by_part: dict[str, set[str]] = {}
    for part, (proto_name, audio_dir) in PARTITIONS.items():
        proto = proto_dir / proto_name
        flac_dir = la / audio_dir / "flac"
        on_disk = {p.name for p in flac_dir.iterdir()} if flac_dir.is_dir() else set()
        listed: set[str] = set()
        n_real = n_fake = missing = 0
        spk: set[str] = set()
        with open(proto, "r", encoding="utf-8") as fh:
            for line in fh:
                parsed = parse_protocol_line(line)
                if parsed is None:
                    continue
                speaker, file_id, attack, key = parsed
                fname = file_id + ".flac"
                listed.add(fname)
                if fname not in on_disk:
                    missing += 1
                label = 0 if key == "bonafide" else 1
                n_real += label == 0
                n_fake += label == 1
                spk.add(speaker)
                records.append(
                    Record(
                        sample_id=f"{KEY}:{file_id}",
                        dataset=KEY,
                        family=FAMILY,
                        path=rel_posix(flac_dir / fname, data_root),
                        modality="audio",
                        label=label,
                        method="real" if label == 0 else attack,
                        group_id=f"{KEY}:{speaker}",
                        group_basis="identity",
                        identity_known=True,
                        official_split=part,
                        notes="",
                    )
                )
        speakers_by_part[part] = spk
        per_part[part] = {
            "rows": n_real + n_fake,
            "expected": EXPECTED[part],
            "bonafide": n_real,
            "spoof": n_fake,
            "speakers": len(spk),
            "listed_files_missing_on_disk": missing,
            "flac_files_on_disk": len(on_disk),
            "flac_files_not_in_cm_protocol": len(on_disk - listed),
        }

    seen_in: dict[str, list[str]] = {}
    for part, spk in speakers_by_part.items():
        for s in spk:
            seen_in.setdefault(s, []).append(part)
    multi = {s: p for s, p in seen_in.items() if len(p) > 1}
    report = {
        "dataset": KEY,
        "rows": len(records),
        "expected_total": sum(EXPECTED.values()),
        "partitions": per_part,
        "speakers_in_more_than_one_partition": multi,
        "speakers_in_more_than_one_partition_count": len(multi),
        "complete": all(v["rows"] == v["expected"] and v["listed_files_missing_on_disk"] == 0
                        for v in per_part.values()),
    }
    return records, report
