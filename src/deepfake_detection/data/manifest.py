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
    # Where the 2 second synchronisation window should start, when the dataset
    # knows better than the default. LAV-DF places a 0.8 to 1.6 second forgery
    # somewhere inside an otherwise genuine clip, so a window pinned to the
    # start of the clip usually contains no manipulation at all while carrying a
    # fake label. Zero means "decide it the usual way".
    sync_start_sec: float = 0.0

    def __post_init__(self) -> None:
        for name in ("clip_id", "dataset", "method", "source"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} cannot be blank")
        if str(self.video_path) in {"", "."}:
            raise ValueError("video_path cannot be blank")
        if not math.isfinite(self.leading_silence_sec) or self.leading_silence_sec < 0:
            raise ValueError("Leading silence must be a finite nonnegative value")
        if not math.isfinite(self.sync_start_sec) or self.sync_start_sec < 0:
            raise ValueError("Sync start must be a finite nonnegative value")
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
            sync_start_sec=float(row.get("sync_start_sec", "0") or 0),
        )


@dataclass(frozen=True, slots=True)
class ManifestLoadResult:
    records: tuple[ClipRecord, ...]
    quarantined_paths: tuple[Path, ...]


def load_manifest(path: Path, *, dataset: str) -> ManifestLoadResult:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows: Sequence[dict[str, str]] = tuple(csv.DictReader(handle))
    def window_of(row: dict[str, str]) -> float:
        return float(row.get("sync_start_sec", "0") or 0)

    # Keyed on the window as well as the file. LAV-DF cuts two clips from one
    # recording on purpose, a window on the manipulated span and a window that
    # misses it, and those carry different labels by design. Grouping on the
    # path alone would read that as the contradiction this quarantine exists to
    # catch, and drop both.
    #
    # The original guard is unchanged for everything else: FakeAVCeleb lists
    # about 22 files twice with conflicting methods and no window offset, so
    # those still collide on (path, 0.0) and are still quarantined.
    by_window: dict[tuple[Path, float], list[dict[str, str]]] = {}
    for row in rows:
        by_window.setdefault((Path(row["video_path"]), window_of(row)), []).append(row)

    paths_by_clip: dict[str, set[tuple[Path, float]]] = {}
    for row in rows:
        paths_by_clip.setdefault(row["clip_id"], set()).add(
            (Path(row["video_path"]), window_of(row))
        )
    ambiguous_clip_ids = {
        clip_id for clip_id, keys in paths_by_clip.items() if len(keys) > 1
    }

    records: list[ClipRecord] = []
    quarantined: list[Path] = []
    for (video_path, _window), path_rows in by_window.items():
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
        "sync_start_sec",
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
                    "sync_start_sec": record.sync_start_sec,
                }
            )
