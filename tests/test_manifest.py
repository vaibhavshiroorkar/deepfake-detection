import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from deepfake_detection.data.manifest import ClipRecord, load_manifest


class ClipRecordTests(unittest.TestCase):
    def test_derives_independent_video_and_audio_labels(self) -> None:
        cases = [
            ("RealVideo-RealAudio", False, False, False),
            ("FakeVideo-RealAudio", True, True, False),
            ("RealVideo-FakeAudio", True, False, True),
            ("FakeVideo-FakeAudio", True, True, True),
        ]

        for manipulation_type, clip_fake, video_fake, audio_fake in cases:
            with self.subTest(manipulation_type=manipulation_type):
                record = ClipRecord.from_mapping(
                    {
                        "clip_id": "clip-001",
                        "dataset": "FakeAVCeleb",
                        "video_path": "clips/clip-001.mp4",
                        "manipulation_type": manipulation_type,
                        "method": "fixture",
                        "source": "id001",
                        "target1": "-",
                        "target2": "-",
                    }
                )

                self.assertEqual(record.clip_fake, clip_fake)
                self.assertEqual(record.video_fake, video_fake)
                self.assertEqual(record.audio_fake, audio_fake)

    def test_rejects_unknown_manipulation_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown manipulation type"):
            ClipRecord.from_mapping(
                {
                    "clip_id": "clip-001",
                    "video_path": "clip.mp4",
                    "manipulation_type": "Synthetic",
                    "method": "fixture",
                    "source": "id001",
                }
            )

    def test_preserves_demographic_and_timing_metadata(self) -> None:
        record = ClipRecord.from_mapping(
            {
                "clip_id": "clip-001",
                "video_path": "clip.mp4",
                "manipulation_type": "RealVideo-RealAudio",
                "method": "real",
                "source": "id001",
                "race": "African",
                "gender": "women",
                "leading_silence_sec": "0.75",
            }
        )

        self.assertEqual(record.race, "African")
        self.assertEqual(record.gender, "women")
        self.assertEqual(record.leading_silence_sec, 0.75)

    def test_rejects_blank_identifiers_and_negative_silence(self) -> None:
        base = {
            "clip_id": "clip-001",
            "video_path": "clip.mp4",
            "manipulation_type": "RealVideo-RealAudio",
            "method": "real",
            "source": "id001",
        }
        with self.assertRaisesRegex(ValueError, "clip_id"):
            ClipRecord.from_mapping({**base, "clip_id": ""})
        with self.assertRaisesRegex(ValueError, "video_path"):
            ClipRecord.from_mapping({**base, "video_path": ""})
        with self.assertRaisesRegex(ValueError, "Leading silence"):
            ClipRecord.from_mapping({**base, "leading_silence_sec": "-0.1"})

    def test_quarantines_conflicting_duplicate_media(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            path.write_text(
                "clip_id,video_path,manipulation_type,method,source\n"
                "real-1,same.mp4,RealVideo-RealAudio,real,id001\n"
                "fake-1,same.mp4,FakeVideo-RealAudio,faceswap,id001\n"
                "clean-1,clean.mp4,RealVideo-RealAudio,real,id002\n",
                encoding="utf-8",
            )

            result = load_manifest(path, dataset="FakeAVCeleb")

        self.assertEqual([record.clip_id for record in result.records], ["clean-1"])
        self.assertEqual(result.quarantined_paths, (Path("same.mp4"),))

    def test_quarantines_clip_ids_that_point_to_multiple_media_files(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            path.write_text(
                "clip_id,video_path,manipulation_type,method,source\n"
                "duplicate,first.mp4,RealVideo-RealAudio,real,id001\n"
                "duplicate,second.mp4,RealVideo-RealAudio,real,id001\n",
                encoding="utf-8",
            )

            result = load_manifest(path, dataset="FakeAVCeleb")

        self.assertEqual(result.records, ())
        self.assertEqual(
            result.quarantined_paths,
            (Path("first.mp4"), Path("second.mp4")),
        )


if __name__ == "__main__":
    unittest.main()


def test_conflicting_rows_on_the_same_window_are_still_quarantined() -> None:
    """The guard that catches FakeAVCeleb's duplicate listings must survive the
    change that let LAV-DF cut two windows from one file. Same path and same
    window with different methods is still a contradiction."""
    with TemporaryDirectory() as directory:
        path = Path(directory) / "manifest.csv"
        path.write_text(
            "clip_id,video_path,manipulation_type,method,source,sync_start_sec\n"
            "a,clip.mp4,FakeVideo-RealAudio,wav2lip,id1,0\n"
            "b,clip.mp4,FakeVideo-RealAudio,faceswap,id1,0\n",
            encoding="utf-8",
        )

        result = load_manifest(path, dataset="fixture")

        assert not result.records
        assert result.quarantined_paths == (Path("clip.mp4"),)


def test_two_windows_of_one_file_are_kept_apart() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "manifest.csv"
        path.write_text(
            "clip_id,video_path,manipulation_type,method,source,sync_start_sec\n"
            "a__forged,clip.mp4,FakeVideo-RealAudio,lavdf-forged,id1,4.5\n"
            "a__genuine,clip.mp4,RealVideo-RealAudio,lavdf-genuine-span,id1,0.0\n",
            encoding="utf-8",
        )

        result = load_manifest(path, dataset="fixture")

        assert len(result.records) == 2
        assert not result.quarantined_paths
        assert {r.clip_fake for r in result.records} == {True, False}
