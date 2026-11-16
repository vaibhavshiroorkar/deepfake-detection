import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lavdf_window_labels.py"
spec = importlib.util.spec_from_file_location("lavdf_window_labels", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

ROW = {"clip_id": "x", "clip_fake": "1", "video_fake": "1", "audio_fake": "1",
       "manipulation_type": "FakeVideo-FakeAudio", "method": "lavdf-FakeVideo-FakeAudio"}
META = {"fake_periods": [[4.0, 4.5]], "modify_video": True, "modify_audio": False}


def test_edit_outside_the_window_makes_the_clip_real():
    out = module.window_label(ROW, META, {"offset_sec": "0.5", "kept_sec": "3.0"})
    assert out["clip_fake"] == "0" and out["video_fake"] == "0" and out["audio_fake"] == "0"
    assert out["file_clip_fake"] == "1"
    assert out["manipulation_type"] == "RealVideo-RealAudio"


def test_edit_inside_the_window_keeps_the_streams_that_were_modified():
    out = module.window_label(ROW, META, {"offset_sec": "2.0", "kept_sec": "3.0"})
    assert out["clip_fake"] == "1" and out["video_fake"] == "1" and out["audio_fake"] == "0"


def test_less_than_one_frame_of_overlap_does_not_count():
    out = module.window_label(ROW, META, {"offset_sec": "1.02", "kept_sec": "3.0"})   # ends at 4.02
    assert out["clip_fake"] == "0"
