from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

MANIPULATION_TYPES = frozenset(
    {
        "RealVideo-RealAudio",
        "FakeVideo-RealAudio",
        "RealVideo-FakeAudio",
        "FakeVideo-FakeAudio",
    }
)


@dataclass(frozen=True, slots=True)
class ClipRecord:
    clip_id: str
    dataset: str
    video_path: Path
    manipulation_type: str
    method: str
    source: str
    targets: tuple[str, ...]
    clip_fake: bool
    video_fake: bool
    audio_fake: bool
    race: str = "unknown"
    gender: str = "unknown"
    leading_silence_sec: float = 0.0

    def __post_init__(self) -> None:
        for name in ("clip_id", "dataset", "method", "source"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} cannot be blank")
        if str(self.video_path) in {"", "."}:
            raise ValueError("video_path cannot be blank")
        if not math.isfinite(self.leading_silence_sec) or self.leading_silence_sec < 0:
            raise ValueError("Leading silence must be a finite nonnegative value")
        expected_video_fake = self.manipulation_type.startswith("FakeVideo-")
        expected_audio_fake = self.manipulation_type.endswith("-FakeAudio")
        if self.manipulation_type not in MANIPULATION_TYPES:
            raise ValueError(f"Unknown manipulation type: {self.manipulation_type}")
        if (
            self.video_fake != expected_video_fake
            or self.audio_fake != expected_audio_fake
            or self.clip_fake != (expected_video_fake or expected_audio_fake)
        ):
            raise ValueError("Cue labels conflict with the manipulation type")

    @classmethod
    def from_mapping(cls, row: Mapping[str, str]) -> ClipRecord:
        manipulation_type = row["manipulation_type"].strip()
        if manipulation_type not in MANIPULATION_TYPES:
            raise ValueError(f"Unknown manipulation type: {manipulation_type}")
        video_fake = manipulation_type.startswith("FakeVideo-")
        audio_fake = manipulation_type.endswith("-FakeAudio")
        targets = tuple(
            value
            for key in ("target1", "target2")
            if (value := row.get(key, "-")) not in {"", "-"}
        )
        return cls(
            clip_id=row["clip_id"].strip(),
            dataset=row.get("dataset", "unknown").strip(),
            video_path=Path(row["video_path"].strip()),
            manipulation_type=manipulation_type,
            method=row["method"].strip(),
            source=row["source"].strip(),
            targets=targets,
            clip_fake=video_fake or audio_fake,
            video_fake=video_fake,
            audio_fake=audio_fake,
            race=row.get("race", "unknown").strip() or "unknown",
            gender=row.get("gender", "unknown").strip() or "unknown",
            leading_silence_sec=float(row.get("leading_silence_sec", "0") or 0),
        )


@dataclass(frozen=True, slots=True)
class ManifestLoadResult:
    records: tuple[ClipRecord, ...]
    quarantined_paths: tuple[Path, ...]


def load_manifest(path: Path, *, dataset: str) -> ManifestLoadResult:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows: Sequence[dict[str, str]] = tuple(csv.DictReader(handle))
    by_path: dict[Path, list[dict[str, str]]] = {}
    for row in rows:
        by_path.setdefault(Path(row["video_path"]), []).append(row)

    paths_by_clip: dict[str, set[Path]] = {}
    for row in rows:
        paths_by_clip.setdefault(row["clip_id"], set()).add(Path(row["video_path"]))
    ambiguous_clip_ids = {
        clip_id for clip_id, paths in paths_by_clip.items() if len(paths) > 1
    }

    records: list[ClipRecord] = []
    quarantined: list[Path] = []
    for video_path, path_rows in by_path.items():
        if any(row["clip_id"] in ambiguous_clip_ids for row in path_rows):
            quarantined.append(video_path)
            continue
        signatures = {(row["manipulation_type"], row["method"]) for row in path_rows}
        if len(signatures) > 1:
            quarantined.append(video_path)
            continue
        records.append(ClipRecord.from_mapping({**path_rows[0], "dataset": dataset}))

    return ManifestLoadResult(
        records=tuple(records),
        quarantined_paths=tuple(sorted(set(quarantined))),
    )


def write_manifest(records: Sequence[ClipRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "clip_id",
        "dataset",
        "video_path",
        "manipulation_type",
        "method",
        "source",
        "target1",
        "target2",
        "clip_fake",
        "video_fake",
        "audio_fake",
        "race",
        "gender",
        "leading_silence_sec",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "clip_id": record.clip_id,
                    "dataset": record.dataset,
                    "video_path": str(record.video_path),
                    "manipulation_type": record.manipulation_type,
                    "method": record.method,
                    "source": record.source,
                    "target1": record.targets[0] if record.targets else "-",
                    "target2": record.targets[1] if len(record.targets) > 1 else "-",
                    "clip_fake": int(record.clip_fake),
                    "video_fake": int(record.video_fake),
                    "audio_fake": int(record.audio_fake),
                    "race": record.race,
                    "gender": record.gender,
                    "leading_silence_sec": record.leading_silence_sec,
                }
            )
