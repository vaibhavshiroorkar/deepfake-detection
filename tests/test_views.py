import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from deepfake_detection.views.cache import (
    cache_fingerprint,
    preprocessing_config_hash,
)
from deepfake_detection.views.timeline import (
    ViewConfig,
    make_sync_window,
    sliding_window_starts,
    uniform_timestamps,
)


class TimelineTests(unittest.TestCase):
    def test_uniform_timestamps_stay_inside_post_silence_interval(self) -> None:
        self.assertEqual(
            uniform_timestamps(duration_sec=10.0, count=4, start_sec=2.0),
            (3.0, 5.0, 7.0, 9.0),
        )
        self.assertEqual(
            uniform_timestamps(duration_sec=1.0, count=4, start_sec=0.0),
            (0.125, 0.375, 0.625, 0.875),
        )

    def test_sliding_windows_include_the_final_complete_interval(self) -> None:
        self.assertEqual(
            sliding_window_starts(
                duration_sec=5.0,
                window_sec=4.0,
                overlap=0.5,
            ),
            (0.0, 1.0),
        )
        self.assertEqual(
            sliding_window_starts(
                duration_sec=3.0,
                window_sec=4.0,
                overlap=0.5,
            ),
            (0.0,),
        )

    def test_sync_window_has_exact_video_and_audio_alignment(self) -> None:
        window = make_sync_window(start_sec=1.0, config=ViewConfig())

        self.assertEqual(len(window.video_timestamps_sec), 50)
        self.assertAlmostEqual(window.video_timestamps_sec[0], 1.02)
        self.assertAlmostEqual(window.video_timestamps_sec[-1], 2.98)
        self.assertEqual(window.audio_start_sample, 16_000)
        self.assertEqual(window.audio_sample_count, 32_000)


class CacheFingerprintTests(unittest.TestCase):
    def test_changes_when_media_or_view_configuration_changes(self) -> None:
        with TemporaryDirectory() as directory:
            media = Path(directory) / "clip.mp4"
            media.write_bytes(b"first")
            config = ViewConfig()
            original = cache_fingerprint(
                media,
                dataset="fixture",
                config=config,
                code_version="1",
            )
            changed_config = cache_fingerprint(
                media,
                dataset="fixture",
                config=replace(config, crop_margin=0.25),
                code_version="1",
            )
            changed_silence = cache_fingerprint(
                media,
                dataset="fixture",
                config=config,
                code_version="1",
                leading_silence_sec=0.5,
            )
            media.write_bytes(b"second")
            changed_media = cache_fingerprint(
                media,
                dataset="fixture",
                config=config,
                code_version="1",
            )

        self.assertNotEqual(original, changed_config)
        self.assertNotEqual(original, changed_silence)
        self.assertNotEqual(original, changed_media)

    def test_global_preprocessing_hash_excludes_clip_content(self) -> None:
        config = ViewConfig()

        first = preprocessing_config_hash(config=config, code_version="1")
        second = preprocessing_config_hash(config=config, code_version="1")
        changed = preprocessing_config_hash(config=config, code_version="2")

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)


if __name__ == "__main__":
    unittest.main()


def test_load_can_skip_views_it_was_not_asked_for(tmp_path) -> None:
    """A cached clip holds four views and they are not small: 9.19 MiB for the
    visual view and 7.18 MiB for the mouth crops. Visual stream training died
    inside the mouth-crop allocation of a run that never reads mouth crops."""
    import numpy as np

    from deepfake_detection.views.cache_store import CacheStore
    from deepfake_detection.views.contracts import PreparedClip, QualityReport

    store = CacheStore(tmp_path / "cache")
    path = store.save(
        PreparedClip(
            clip_id="clip",
            visual_view=np.zeros((2, 3, 4, 4), dtype=np.float32),
            audio_view=np.zeros((8,), dtype=np.float32),
            sync_video_view=np.zeros((2, 3, 4, 4), dtype=np.float32),
            sync_audio_view=np.zeros((8,), dtype=np.float32),
            quality=QualityReport(1.0, True, True, False, 0.0),
            preprocessing_fingerprint="fingerprint",
            preprocessing_config_hash="hash",
        ),
        dataset="fixture",
    )

    only_visual = store.load(path, views=("visual_view",))

    assert only_visual.visual_view is not None
    assert only_visual.sync_video_view is None
    assert only_visual.audio_view is None
    # Metadata is never optional: the preprocessing hash guard reads it on every
    # clip, so skipping views must not skip that.
    assert only_visual.preprocessing_config_hash == "hash"
    assert only_visual.quality.face_coverage == 1.0


def test_load_returns_every_view_by_default(tmp_path) -> None:
    import numpy as np

    from deepfake_detection.views.cache_store import CacheStore
    from deepfake_detection.views.contracts import PreparedClip, QualityReport

    store = CacheStore(tmp_path / "cache")
    path = store.save(
        PreparedClip(
            clip_id="clip",
            visual_view=np.zeros((2, 3, 4, 4), dtype=np.float32),
            audio_view=np.zeros((8,), dtype=np.float32),
            sync_video_view=np.zeros((2, 3, 4, 4), dtype=np.float32),
            sync_audio_view=np.zeros((8,), dtype=np.float32),
            quality=QualityReport(1.0, True, True, False, 0.0),
            preprocessing_fingerprint="fingerprint",
            preprocessing_config_hash="hash",
        ),
        dataset="fixture",
    )

    everything = store.load(path)

    assert everything.visual_view is not None
    assert everything.audio_view is not None
    assert everything.sync_video_view is not None
    assert everything.sync_audio_view is not None
