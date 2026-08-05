"""The cache stamp in version.txt, which decides what gets re-extracted.

Crops depend on the detector as much as on the pipeline version, so both go in
the stamp. Getting this wrong is quiet and expensive: one detector reads the
other's crops, and a training run learns from a mix nobody chose. No video is
decoded here, only the stamp logic.
"""
import numpy as np
import pytest

from preprocessing.extract_clip import _cache_valid, _version_stamp
from preprocessing.ops import detectors as D
from preprocessing.ops.constants import PIPELINE_VERSION


@pytest.fixture
def cache_dir(tmp_path):
    """A directory holding every array a cache hit needs, minus the stamp."""
    for name in ("frames.npy", "audio.npy", "timestamps.npy"):
        np.save(tmp_path / name, np.zeros(1, dtype=np.float32))
    return tmp_path


def _stamp(cache_dir, text):
    (cache_dir / "version.txt").write_text(text)


def test_a_matching_stamp_is_a_hit(cache_dir):
    _stamp(cache_dir, _version_stamp(D.YUNET_NAME))
    assert _cache_valid(cache_dir, D.YUNET_NAME)


def test_the_other_detector_is_a_miss(cache_dir):
    _stamp(cache_dir, _version_stamp(D.YUNET_NAME))
    assert not _cache_valid(cache_dir, D.MTCNN_NAME)


def test_a_bare_version_reads_as_mtcnn(cache_dir):
    # Caches written before the second detector existed carry just "4". MTCNN is
    # the only thing that can have produced them, so they stay valid for MTCNN
    # and must not be handed to YuNet.
    _stamp(cache_dir, str(PIPELINE_VERSION))
    assert _cache_valid(cache_dir, D.MTCNN_NAME)
    assert not _cache_valid(cache_dir, D.YUNET_NAME)


def test_an_older_pipeline_version_is_a_miss(cache_dir):
    _stamp(cache_dir, f"{PIPELINE_VERSION - 1}:{D.MTCNN_NAME}")
    assert not _cache_valid(cache_dir, D.MTCNN_NAME)


def test_a_missing_array_is_a_miss(cache_dir):
    _stamp(cache_dir, _version_stamp(D.MTCNN_NAME))
    (cache_dir / "frames.npy").unlink()
    assert not _cache_valid(cache_dir, D.MTCNN_NAME)


def test_a_missing_stamp_is_a_miss(cache_dir):
    assert not _cache_valid(cache_dir, D.MTCNN_NAME)
