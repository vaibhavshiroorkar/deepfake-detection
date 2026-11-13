"""Audio-only spoofing corpora for the audio path.

ASVspoof 2019 LA and In-the-Wild have no video, so they cannot enter the
audio-visual ``ClipRecord`` manifest. They train and stress-test the audio
branch on its own. Ported from the team project's builders, which checked the
protocol files against the files on disk; that check is kept.

Group is the speaker. ASVspoof partitions are speaker-disjoint by design, and
the report counts any speaker found in two partitions. In-the-Wild has no
official split, and every speaker has both labels, so any split must group on
speaker or a model can learn the voice instead of the spoof.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ASVSPOOF_PARTITIONS = {
    "train": ("ASVspoof2019.LA.cm.train.trn.txt", "ASVspoof2019_LA_train"),
    "dev": ("ASVspoof2019.LA.cm.dev.trl.txt", "ASVspoof2019_LA_dev"),
    "eval": ("ASVspoof2019.LA.cm.eval.trl.txt", "ASVspoof2019_LA_eval"),
}
ASVSPOOF_EXPECTED = {"train": 25380, "dev": 24844, "eval": 71237}
IN_THE_WILD_LABELS = {"bona-fide": False, "spoof": True}


@dataclass(frozen=True, slots=True)
class AudioRecord:
    clip_id: str
    dataset: str
    audio_path: Path
    audio_fake: bool
    method: str
    speaker: str
    partition: str = ""

    def __post_init__(self) -> None:
        for name in ("clip_id", "dataset", "method", "speaker"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} cannot be blank")
        if self.audio_fake == (self.method == "real"):
            raise ValueError("method 'real' must match a genuine label")


def parse_asvspoof_line(line: str) -> tuple[str, str, str, bool] | None:
    """Return (speaker, file_id, attack, is_spoof), or None for a blank line."""
    parts = line.split()
    if not parts:
        return None
    if len(parts) != 5:
        raise ValueError(f"expected 5 fields, got {len(parts)}: {line!r}")
    speaker, file_id, _environment, attack, key = parts
    if key not in {"bonafide", "spoof"}:
        raise ValueError(f"key must be bonafide or spoof: {line!r}")
    if key == "spoof" and attack == "-":
        raise ValueError(f"spoof line without an attack id: {line!r}")
    return speaker, file_id, attack, key == "spoof"


def build_asvspoof2019_la(root: Path) -> tuple[list[AudioRecord], dict]:
    """Rows come only from the CM protocols.

    The dev and eval folders also hold speaker-verification enrolment files
    that the CM protocols do not list. They are counted, never included.
    """
    la = root / "LA" if (root / "LA").is_dir() else root
    protocols = la / "ASVspoof2019_LA_cm_protocols"
    if not protocols.is_dir():
        raise FileNotFoundError(f"protocol folder not found: {protocols}")
    records: list[AudioRecord] = []
    partitions: dict[str, dict] = {}
    speakers_by_partition: dict[str, set[str]] = {}
    for partition, (protocol_name, folder) in ASVSPOOF_PARTITIONS.items():
        flac = la / folder / "flac"
        on_disk = {p.name for p in flac.iterdir()} if flac.is_dir() else set()
        listed: set[str] = set()
        counts = Counter()
        speakers: set[str] = set()
        with (protocols / protocol_name).open(encoding="utf-8") as handle:
            for line in handle:
                parsed = parse_asvspoof_line(line)
                if parsed is None:
                    continue
                speaker, file_id, attack, spoof = parsed
                name = f"{file_id}.flac"
                listed.add(name)
                counts["missing"] += name not in on_disk
                counts["spoof" if spoof else "bonafide"] += 1
                speakers.add(speaker)
                records.append(
                    AudioRecord(
                        clip_id=f"asvspoof2019_la:{file_id}",
                        dataset="ASVspoof2019_LA",
                        audio_path=flac / name,
                        audio_fake=spoof,
                        method=attack if spoof else "real",
                        speaker=speaker,
                        partition=partition,
                    )
                )
        speakers_by_partition[partition] = speakers
        partitions[partition] = {
            "rows": counts["bonafide"] + counts["spoof"],
            "expected": ASVSPOOF_EXPECTED[partition],
            "bonafide": counts["bonafide"],
            "spoof": counts["spoof"],
            "speakers": len(speakers),
            "missing_on_disk": counts["missing"],
            "unlisted_on_disk": len(on_disk - listed),
        }
    seen: dict[str, list[str]] = defaultdict(list)
    for partition, speakers in speakers_by_partition.items():
        for speaker in speakers:
            seen[speaker].append(partition)
    crossing = {speaker: parts for speaker, parts in seen.items() if len(parts) > 1}
    report = {
        "dataset": "ASVspoof2019_LA",
        "rows": len(records),
        "partitions": partitions,
        "speakers_in_two_partitions": len(crossing),
        "complete": all(
            entry["rows"] == entry["expected"] and entry["missing_on_disk"] == 0
            for entry in partitions.values()
        ),
    }
    return records, report


def build_in_the_wild(root: Path) -> tuple[list[AudioRecord], dict]:
    base = root / "release_in_the_wild" if (root / "release_in_the_wild").is_dir() else root
    meta = base / "meta.csv"
    if not meta.is_file():
        raise FileNotFoundError(f"meta.csv not found: {meta}")
    on_disk = {p.name for p in base.iterdir() if p.is_file()}
    records: list[AudioRecord] = []
    seen: set[str] = set()
    per_speaker: dict[str, Counter] = defaultdict(Counter)
    with meta.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["file", "speaker", "label"]:
            raise ValueError(f"unexpected meta.csv header: {reader.fieldnames}")
        for row in reader:
            name, speaker, label = row["file"], row["speaker"].strip(), row["label"]
            if label not in IN_THE_WILD_LABELS:
                raise ValueError(f"unknown label {label!r}")
            if name in seen:
                raise ValueError(f"duplicate file {name}")
            if name not in on_disk:
                raise ValueError(f"meta.csv lists {name} but it is not on disk")
            seen.add(name)
            fake = IN_THE_WILD_LABELS[label]
            per_speaker[speaker][label] += 1
            records.append(
                AudioRecord(
                    clip_id=f"in_the_wild:{name}",
                    dataset="In-the-Wild",
                    audio_path=base / name,
                    audio_fake=fake,
                    method="spoof" if fake else "real",
                    speaker=speaker,
                )
            )
    wavs = {n for n in on_disk if n.lower().endswith(".wav")}
    report = {
        "dataset": "In-the-Wild",
        "rows": len(records),
        "fake": sum(r.audio_fake for r in records),
        "speakers": len(per_speaker),
        "speakers_with_both_labels": sum(len(c) == 2 for c in per_speaker.values()),
        "wav_not_in_meta": len(wavs - seen),
        "complete": bool(records) and not (wavs - seen),
    }
    return records, report


AUDIO_FIELDS = ("clip_id", "dataset", "audio_path", "audio_fake", "method", "speaker", "partition")


def write_audio_manifest(records: Sequence[AudioRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIO_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "clip_id": record.clip_id,
                    "dataset": record.dataset,
                    "audio_path": str(record.audio_path),
                    "audio_fake": int(record.audio_fake),
                    "method": record.method,
                    "speaker": record.speaker,
                    "partition": record.partition,
                }
            )


def load_audio_manifest(path: Path) -> list[AudioRecord]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            AudioRecord(
                clip_id=row["clip_id"],
                dataset=row["dataset"],
                audio_path=Path(row["audio_path"]),
                audio_fake=row["audio_fake"] == "1",
                method=row["method"],
                speaker=row["speaker"],
                partition=row.get("partition", ""),
            )
            for row in csv.DictReader(handle)
        ]
