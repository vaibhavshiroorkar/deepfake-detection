import unittest

from deepfake_detection.views.contracts import QualityReport


class QualityReportTests(unittest.TestCase):
    def test_names_every_reason_full_fusion_is_unavailable(self) -> None:
        quality = QualityReport(
            face_coverage=0.60,
            stable_face_track=False,
            audio_present=False,
            audio_clipped=False,
            av_duration_delta_sec=0.10,
        )

        self.assertEqual(
            quality.full_fusion_blockers(),
            (
                "missing_audio",
                "unstable_face_track",
                "low_face_coverage",
            ),
        )

    def test_accepts_a_complete_single_speaker_clip(self) -> None:
        quality = QualityReport(
            face_coverage=0.95,
            stable_face_track=True,
            audio_present=True,
            audio_clipped=False,
            av_duration_delta_sec=0.02,
        )

        self.assertEqual(quality.full_fusion_blockers(), ())

    def test_names_missing_landmarks_as_a_fusion_blocker(self) -> None:
        quality = QualityReport(
            face_coverage=1.0,
            stable_face_track=True,
            audio_present=True,
            audio_clipped=False,
            av_duration_delta_sec=0.02,
            landmark_coverage=0.75,
        )

        self.assertEqual(
            quality.full_fusion_blockers(),
            ("missing_face_landmarks",),
        )

    def test_landmark_coverage_defaults_to_complete_for_old_callers(self) -> None:
        quality = QualityReport(1.0, True, True, False, 0.02)

        self.assertEqual(quality.landmark_coverage, 1.0)

    def test_rejects_landmark_coverage_outside_unit_interval(self) -> None:
        with self.assertRaisesRegex(ValueError, "Landmark coverage"):
            QualityReport(1.0, True, True, False, 0.02, landmark_coverage=1.01)


if __name__ == "__main__":
    unittest.main()
