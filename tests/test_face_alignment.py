import cv2
import numpy as np
import pytest

from deepfake_detection.views.alignment import (
    aligned_lower_face,
    similarity_transform,
)
from deepfake_detection.views.tracking import Landmarks5, Point


def landmarks(values: list[tuple[float, float]]) -> Landmarks5:
    points = [Point(x, y) for x, y in values]
    return Landmarks5(
        eye_left=points[0],
        eye_right=points[1],
        nose=points[2],
        mouth_left=points[3],
        mouth_right=points[4],
    )


def transform_landmarks(source: Landmarks5, matrix: np.ndarray) -> Landmarks5:
    values = [
        source.eye_left,
        source.eye_right,
        source.nose,
        source.mouth_left,
        source.mouth_right,
    ]
    transformed = [
        tuple(matrix @ np.asarray((point.x, point.y, 1.0))) for point in values
    ]
    return landmarks(transformed)


def test_similarity_transform_uses_all_five_points_in_one_least_squares_fit() -> None:
    source = landmarks([(10, 15), (35, 14), (24, 25), (15, 38), (34, 39)])

    matrix = similarity_transform(source, output_size=(101, 101))

    expected = np.asarray(
        (
            (1.53866171, -0.07695167, 15.70371747),
            (0.07695167, 1.53866171, 14.27100372),
        )
    )
    assert np.allclose(matrix, expected, atol=1e-7)


def test_similarity_transform_removes_roll_scale_and_translation() -> None:
    canonical = landmarks([(31.5, 37), (68.5, 37), (50, 56), (36.5, 76), (63.5, 76)])
    angle = np.deg2rad(23.0)
    source_from_canonical = np.asarray(
        (
            (1.4 * np.cos(angle), -1.4 * np.sin(angle), 18.0),
            (1.4 * np.sin(angle), 1.4 * np.cos(angle), 11.0),
        )
    )
    source = transform_landmarks(canonical, source_from_canonical)

    matrix = similarity_transform(source, output_size=(101, 101))

    expected = cv2.invertAffineTransform(source_from_canonical)
    assert np.allclose(matrix, expected, atol=1e-10)


def test_similarity_transform_rejects_degenerate_landmarks() -> None:
    source = landmarks([(20, 20)] * 5)

    with pytest.raises(ValueError, match="degenerate"):
        similarity_transform(source, output_size=(112, 112))


def test_aligned_lower_face_has_stable_shape_and_normalizes_rgb_channels() -> None:
    frame = np.full((224, 224, 3), (10, 20, 30), dtype=np.uint8)
    source = landmarks(
        [
            (70.245, 82.51),
            (152.755, 82.51),
            (111.5, 124.88),
            (81.395, 169.48),
            (141.605, 169.48),
        ]
    )

    output = aligned_lower_face(frame, source, height=20, width=30)

    expected_pixel = (
        np.asarray((30, 20, 10), dtype=np.float32) / 255.0
        - np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    ) / np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    assert output.shape == (3, 20, 30)
    assert output.dtype == np.float32
    assert np.allclose(output[:, 10, 15], expected_pixel, atol=1e-6)


def test_aligned_lower_face_rejects_landmarks_outside_the_frame() -> None:
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    source = landmarks([(70, 82), (224, 82), (111, 124), (81, 169), (141, 169)])

    with pytest.raises(ValueError, match="inside the frame"):
        aligned_lower_face(frame, source, height=20, width=30)


@pytest.mark.parametrize("height,width", [(0, 30), (20, 0)])
def test_aligned_lower_face_rejects_empty_output_size(height: int, width: int) -> None:
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    source = landmarks([(70, 82), (153, 82), (111, 124), (81, 169), (142, 169)])

    with pytest.raises(ValueError, match="positive"):
        aligned_lower_face(frame, source, height=height, width=width)
