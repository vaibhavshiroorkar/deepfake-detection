import csv
import hashlib
import json
from pathlib import Path

import pytest

from deepfake_detection.experiments.smoke import _atomic_write, run_fusion_smoke


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fusion_smoke_is_deterministic_and_writes_fixture_evidence(
    tmp_path: Path,
) -> None:
    first = run_fusion_smoke(tmp_path / "first", seed=17, samples=32)
    second = run_fusion_smoke(tmp_path / "second", seed=17, samples=32)

    assert first.metrics == second.metrics
    assert first.threshold == second.threshold
    assert first.samples == 32
    assert first.train_samples == 24
    assert first.validation_samples == 8
    for directory in (tmp_path / "first", tmp_path / "second"):
        assert (directory / "fusion.joblib").is_file()
        assert (directory / "smoke-report.json").is_file()
        assert (directory / "predictions.csv").is_file()

    rows = list(
        csv.DictReader((tmp_path / "first" / "predictions.csv").open(encoding="utf-8"))
    )
    assert len(rows) == 32
    assert tuple(rows[0]) == (
        "source_identity",
        "partition",
        "label",
        "visual_logit",
        "audio_logit",
        "sync_logit",
        "face_coverage",
        "audio_clipped",
        "av_duration_delta_sec",
        "probability",
    )
    assert {row["source_identity"] for row in rows} == {
        f"fixture-source-{index}" for index in range(8)
    }
    assert {row["partition"] for row in rows} == {"fit", "validation"}
    assert {
        row["source_identity"] for row in rows if row["partition"] == "fit"
    }.isdisjoint(
        {row["source_identity"] for row in rows if row["partition"] == "validation"}
    )
    for source in {row["source_identity"] for row in rows}:
        source_rows = [row for row in rows if row["source_identity"] == source]
        assert len(source_rows) % 2 == 0
        assert [row["label"] for row in source_rows].count("0") == len(source_rows) // 2
        assert [row["label"] for row in source_rows].count("1") == len(source_rows) // 2
    assert all(
        row["visual_logit"]
        and row["audio_logit"]
        and row["sync_logit"]
        and row["face_coverage"]
        and row["audio_clipped"] in {"False", "True"}
        and row["av_duration_delta_sec"]
        for row in rows
    )

    report_path = tmp_path / "first" / "smoke-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["evidence_scope"] == "software_fixture_only"
    assert report["metrics"] == first.metrics
    assert set(report["metrics"]) == set(first.metrics)
    assert set(report["metric_evidence_scope"]) == set(report["metrics"])
    assert set(report["metric_evidence_scope"].values()) == {"software_fixture_only"}
    assert report["threshold_selection_partition"] == "fit"
    assert report["evaluation_partition"] == "validation"
    assert "research" not in json.dumps(report).casefold()
    assert report["artifact_hashes"] == {
        "fusion.joblib": _file_hash(tmp_path / "first" / "fusion.joblib"),
        "predictions.csv": _file_hash(tmp_path / "first" / "predictions.csv"),
    }
    assert _file_hash(report_path) == _file_hash(
        tmp_path / "second" / "smoke-report.json"
    )


@pytest.mark.parametrize("samples", (16, 24, 32, 40))
def test_fusion_smoke_accepts_every_supported_sample_count(
    tmp_path: Path, samples: int
) -> None:
    report = run_fusion_smoke(tmp_path / str(samples), seed=17, samples=samples)

    rows = list(
        csv.DictReader(
            (tmp_path / str(samples) / "predictions.csv").open(encoding="utf-8")
        )
    )
    counts = {
        source: sum(row["source_identity"] == source for row in rows)
        for source in {row["source_identity"] for row in rows}
    }
    assert report.samples == samples
    assert len(counts) == 8
    assert all(count % 2 == 0 for count in counts.values())
    if samples == 24:
        assert set(counts.values()) == {2, 4}
    assert report.train_samples == sum(row["partition"] == "fit" for row in rows)
    assert report.validation_samples == sum(
        row["partition"] == "validation" for row in rows
    )


@pytest.mark.parametrize("samples", (0, 8, 15, 17, 20, 31))
def test_fusion_smoke_rejects_unsupported_sample_counts(
    tmp_path: Path, samples: int
) -> None:
    with pytest.raises(ValueError, match="at least 16 and a multiple of 8"):
        run_fusion_smoke(tmp_path, seed=17, samples=samples)


def test_atomic_smoke_writes_preserve_completed_evidence_on_failure(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("completed", encoding="utf-8")

    def fail_after_writing_temporary_file(temporary: Path) -> None:
        temporary.write_text("incomplete", encoding="utf-8")
        raise RuntimeError("write failed")

    with pytest.raises(RuntimeError, match="write failed"):
        _atomic_write(artifact, fail_after_writing_temporary_file)

    assert artifact.read_text(encoding="utf-8") == "completed"
    assert not list(tmp_path.glob(".artifact.json.*.tmp"))
