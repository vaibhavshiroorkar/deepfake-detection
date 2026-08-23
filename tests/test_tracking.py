import unittest

from deepfake_detection.views.tracking import Box, Detection, select_primary_track


class FaceTrackingTests(unittest.TestCase):
    def test_selects_the_longest_consistent_track(self) -> None:
        frames = (
            (Detection(Box(0, 0, 10, 10), 0.95),),
            (
                Detection(Box(1, 0, 11, 10), 0.94),
                Detection(Box(30, 30, 40, 40), 0.99),
            ),
            (Detection(Box(2, 0, 12, 10), 0.93),),
        )

        result = select_primary_track(frames, min_iou=0.3)

        self.assertEqual(result.frame_indices, (0, 1, 2))
        self.assertEqual(result.coverage, 1.0)
        self.assertTrue(result.stable)

    def test_marks_equal_length_faces_as_ambiguous(self) -> None:
        frames = (
            (
                Detection(Box(0, 0, 10, 10), 0.95),
                Detection(Box(30, 30, 40, 40), 0.95),
            ),
            (
                Detection(Box(1, 0, 11, 10), 0.95),
                Detection(Box(31, 30, 41, 40), 0.95),
            ),
        )

        result = select_primary_track(frames, min_iou=0.3)

        self.assertFalse(result.stable)


if __name__ == "__main__":
    unittest.main()
