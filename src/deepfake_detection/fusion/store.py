from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow as arrow
import pyarrow.parquet as parquet


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    dataset: str
    clip_id: str
    segment_id: str
    branch: str
    logit: float
    embedding: tuple[float, ...]
    available: bool
    checkpoint_hash: str
    preprocessing_hash: str
    split_hash: str
    run_id: str
    label: int
    timestamp_start_sec: float = 0.0
    timestamp_end_sec: float = 0.0
    quality_flags: tuple[str, ...] = ()
    face_coverage: float = 1.0
    audio_clipped: bool = False
    av_duration_delta_sec: float = 0.0
    cache_fingerprint: str = ""
    source_identity: str = ""
    method: str = "unknown"
    race: str = "unknown"
    gender: str = "unknown"
    partition_role: str = "unknown"

    @property
    def key(self) -> tuple[str, ...]:
        return (
            self.dataset,
            self.clip_id,
            self.segment_id,
            self.branch,
            self.checkpoint_hash,
            self.preprocessing_hash,
            self.split_hash,
            self.run_id,
        )


@dataclass(frozen=True, slots=True)
class AssembledFeature:
    dataset: str
    clip_id: str
    segment_id: str
    label: int
    branch_logits: dict[str, float]
    face_coverage: float
    audio_clipped: bool
    av_duration_delta_sec: float
    checkpoint_hashes: dict[str, str]
    preprocessing_hash: str
    split_hash: str
    run_id: str
    source_identity: str
    method: str
    race: str
    gender: str
    available: bool
    missing_branches: tuple[str, ...]
    partition_role: str


class FeatureStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    @staticmethod
    def _to_row(record: FeatureRecord) -> dict[str, object]:
        row = asdict(record)
        row["embedding"] = list(record.embedding)
        row["quality_flags"] = json.dumps(record.quality_flags)
        return row

    @staticmethod
    def _from_row(row: dict[str, object]) -> FeatureRecord:
        return FeatureRecord(
            dataset=str(row["dataset"]),
            clip_id=str(row["clip_id"]),
            segment_id=str(row["segment_id"]),
            branch=str(row["branch"]),
            logit=float(row["logit"]),
            embedding=tuple(float(value) for value in row["embedding"]),
            available=bool(row["available"]),
            checkpoint_hash=str(row["checkpoint_hash"]),
            preprocessing_hash=str(row["preprocessing_hash"]),
            split_hash=str(row["split_hash"]),
            run_id=str(row["run_id"]),
            label=int(row["label"]),
            timestamp_start_sec=float(row["timestamp_start_sec"]),
            timestamp_end_sec=float(row["timestamp_end_sec"]),
            quality_flags=tuple(json.loads(str(row["quality_flags"]))),
            face_coverage=float(row["face_coverage"]),
            audio_clipped=bool(row["audio_clipped"]),
            av_duration_delta_sec=float(row["av_duration_delta_sec"]),
            cache_fingerprint=str(row.get("cache_fingerprint", "")),
            source_identity=str(row.get("source_identity", "")),
            method=str(row.get("method", "unknown")),
            race=str(row.get("race", "unknown")),
            gender=str(row.get("gender", "unknown")),
            partition_role=str(row.get("partition_role", "unknown")),
        )

    def read(self) -> tuple[FeatureRecord, ...]:
        if not self.path.exists():
            return ()
        return tuple(
            self._from_row(row) for row in parquet.read_table(self.path).to_pylist()
        )

    def write(self, records: Iterable[FeatureRecord]) -> None:
        existing = list(self.read())
        incoming = list(records)
        seen = {record.key for record in existing}
        for record in incoming:
            if record.key in seen:
                raise ValueError(f"Duplicate feature key: {record.key}")
            seen.add(record.key)
        combined = existing + incoming
        if not combined:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_table(
            arrow.Table.from_pylist([self._to_row(record) for record in combined]),
            self.path,
        )

    def assemble(
        self,
        *,
        required_branches: tuple[str, ...],
        strict: bool = True,
    ) -> tuple[AssembledFeature, ...]:
        grouped: dict[tuple[str, str, str], list[FeatureRecord]] = {}
        for record in self.read():
            grouped.setdefault(
                (record.dataset, record.clip_id, record.segment_id), []
            ).append(record)

        assembled: list[AssembledFeature] = []
        errors: list[str] = []
        for key, records in sorted(grouped.items()):
            by_branch: dict[str, FeatureRecord] = {}
            for record in records:
                if record.branch in by_branch:
                    errors.append(
                        f"{record.clip_id} has multiple {record.branch} records"
                    )
                by_branch[record.branch] = record
            missing = tuple(
                name
                for name in required_branches
                if name not in by_branch or not by_branch[name].available
            )
            if missing and strict:
                errors.append(f"{key[1]} is missing branches: {', '.join(missing)}")
                continue
            required_records = [
                by_branch[name] for name in required_branches if name in by_branch
            ]
            if not required_records:
                errors.append(f"{key[1]} has no requested branch records")
                continue
            labels = {record.label for record in required_records}
            if len(labels) != 1:
                errors.append(f"{key[1]} has conflicting labels")
                continue
            provenance_fields = (
                "preprocessing_hash",
                "split_hash",
                "run_id",
            )
            mixed_provenance = False
            for field in provenance_fields:
                values = {getattr(record, field) for record in required_records}
                if len(values) != 1:
                    errors.append(f"{key[1]} has conflicting {field}")
                    mixed_provenance = True
            if mixed_provenance:
                continue
            metadata_fields = (
                "source_identity",
                "method",
                "race",
                "gender",
                "partition_role",
            )
            mixed_metadata = False
            for field in metadata_fields:
                values = {getattr(record, field) for record in required_records}
                if len(values) != 1:
                    errors.append(f"{key[1]} has conflicting {field}")
                    mixed_metadata = True
            if mixed_metadata:
                continue
            assembled.append(
                AssembledFeature(
                    dataset=key[0],
                    clip_id=key[1],
                    segment_id=key[2],
                    label=labels.pop(),
                    branch_logits={
                        name: by_branch[name].logit
                        for name in required_branches
                        if name in by_branch and by_branch[name].available
                    },
                    face_coverage=min(
                        record.face_coverage for record in required_records
                    ),
                    audio_clipped=any(
                        record.audio_clipped for record in required_records
                    ),
                    av_duration_delta_sec=max(
                        record.av_duration_delta_sec for record in required_records
                    ),
                    checkpoint_hashes={
                        name: by_branch[name].checkpoint_hash
                        for name in required_branches
                        if name in by_branch
                    },
                    preprocessing_hash=required_records[0].preprocessing_hash,
                    split_hash=required_records[0].split_hash,
                    run_id=required_records[0].run_id,
                    source_identity=required_records[0].source_identity,
                    method=required_records[0].method,
                    race=required_records[0].race,
                    gender=required_records[0].gender,
                    available=not missing,
                    missing_branches=missing,
                    partition_role=required_records[0].partition_role,
                )
            )
        if errors:
            raise ValueError("; ".join(errors))
        return tuple(assembled)
