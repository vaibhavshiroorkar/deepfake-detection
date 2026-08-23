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


if __name__ == "__main__":
    unittest.main()
