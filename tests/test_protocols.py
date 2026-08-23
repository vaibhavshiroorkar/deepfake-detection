import unittest
from dataclasses import replace
from pathlib import Path

from deepfake_detection.data.manifest import ClipRecord
from deepfake_detection.data.protocols import (
    audit_split,
    build_method_holdout_protocol,
    build_source_split,
    identity_strict_subset,
    split_hash,
)


def make_record(
    clip_id: str,
    source: str,
    *,
    targets: tuple[str, ...] = (),
    method: str = "real",
) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id,
        dataset="fixture",
        video_path=Path(f"{clip_id}.mp4"),
        manipulation_type="RealVideo-RealAudio",
        method=method,
        source=source,
        targets=targets,
        clip_fake=False,
        video_fake=False,
        audio_fake=False,
    )


class SourceSplitTests(unittest.TestCase):
    def test_keeps_all_rows_and_separates_source_identities(self) -> None:
        records = tuple(
            make_record(f"{source}-{index}", source)
            for source in (f"id{number:02d}" for number in range(20))
            for index in range(2)
        )

        split = build_source_split(records, seed=17)

        self.assertEqual(
            {
                name: len({record.source for record in rows})
                for name, rows in split.items()
            },
            {"train": 14, "val": 3, "test": 3},
        )
        self.assertEqual(sum(map(len, split.values())), 40)
        self.assertEqual(audit_split(split).source_overlaps, {})
        self.assertEqual(split, build_source_split(records, seed=17))

    def test_stratifies_source_identities_by_demographics(self) -> None:
        records = tuple(
            ClipRecord(
                clip_id=f"{race}-{number}",
                dataset="fixture",
                video_path=Path(f"{race}-{number}.mp4"),
                manipulation_type="RealVideo-RealAudio",
                method="real",
                source=f"{race}-{number}",
                targets=(),
                clip_fake=False,
                video_fake=False,
                audio_fake=False,
                race=race,
                gender="fixture",
            )
            for race in ("A", "B", "C", "D")
            for number in range(10)
        )

        split = build_source_split(records, seed=17)

        for name in ("train", "val", "test"):
            with self.subTest(split=name):
                self.assertEqual(
                    {record.race for record in split[name]}, {"A", "B", "C", "D"}
                )

    def test_identity_strict_subset_drops_cross_partition_targets(self) -> None:
        split = {
            "train": (
                make_record("train-safe", "id01", targets=("id02",)),
                make_record("train-cross", "id01", targets=("id03",)),
                make_record("train-peer", "id02"),
            ),
            "val": (make_record("val", "id03"),),
            "test": (make_record("test", "id04"),),
        }

        strict = identity_strict_subset(split)
        audit = audit_split(split)

        self.assertEqual(
            [record.clip_id for record in strict["train"]],
            ["train-safe", "train-peer"],
        )
        self.assertEqual(audit.all_identity_overlaps[("train", "val")], {"id03"})

    def test_split_hash_is_independent_of_row_order_but_not_assignment(self) -> None:
        rows = (
            make_record("a", "id01"),
            make_record("b", "id02"),
            make_record("c", "id03"),
        )
        first = {"train": rows[:1], "val": rows[1:2], "test": rows[2:]}
        reordered = {"test": rows[2:], "val": rows[1:2], "train": rows[:1]}
        changed = {"train": rows[1:2], "val": rows[:1], "test": rows[2:]}

        self.assertEqual(split_hash(first), split_hash(reordered))
        self.assertNotEqual(split_hash(first), split_hash(changed))

    def test_smallest_valid_split_keeps_one_source_in_each_partition(self) -> None:
        records = tuple(
            make_record(f"clip-{index}", f"id-{index}") for index in range(3)
        )

        split = build_source_split(records, seed=17)

        self.assertEqual(
            {name: len(rows) for name, rows in split.items()},
            {"train": 1, "val": 1, "test": 1},
        )

    def test_split_rejects_conflicting_source_demographics(self) -> None:
        records = [make_record(f"clip-{index}", f"id-{index}") for index in range(3)]
        records.append(replace(records[0], clip_id="conflict", race="different"))

        with self.assertRaisesRegex(ValueError, "conflicting demographics"):
            build_source_split(records, seed=17)

    def test_method_holdout_trains_without_the_method_and_tests_only_that_method(
        self,
    ) -> None:
        real_train = make_record("train-real", "id1")
        fake_train = ClipRecord.from_mapping(
            {
                "clip_id": "train-fake",
                "dataset": "fixture",
                "video_path": "train-fake.mp4",
                "manipulation_type": "FakeVideo-RealAudio",
                "method": "wav2lip",
                "source": "id1",
            }
        )
        other_train = ClipRecord.from_mapping(
            {
                "clip_id": "train-other",
                "dataset": "fixture",
                "video_path": "train-other.mp4",
                "manipulation_type": "FakeVideo-RealAudio",
                "method": "faceswap",
                "source": "id1",
            }
        )
        split = {
            "train": (real_train, fake_train, other_train),
            "val": (make_record("val-real", "id2"), fake_train),
            "test": (make_record("test-real", "id3"), fake_train),
        }

        protocol = build_method_holdout_protocol(split, heldout_methods={"wav2lip"})

        self.assertEqual(
            [record.clip_id for record in protocol["train"]],
            ["train-real", "train-other"],
        )
        self.assertEqual(
            [record.method for record in protocol["val"]],
            ["real"],
        )
        self.assertEqual(
            [record.method for record in protocol["test"]],
            ["real", "wav2lip"],
        )


if __name__ == "__main__":
    unittest.main()
