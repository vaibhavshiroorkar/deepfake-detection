"""build_demo_clips.py scores a demo clip the way the website does: every part, then one decision."""

import importlib.util
from pathlib import Path

from deepfake_detection.webapp import calibration as cal
from deepfake_detection.webapp.scorer import CheckResult, ScoreReport

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_demo_clips.py"


def _module():
    spec = importlib.util.spec_from_file_location("build_demo_clips", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Parts:
    """Scores a clip as three parts; the second one has a very high face score."""

    def score(self, upload, workdir, progress=lambda stage: None):
        calm = {"face": CheckResult((0.5,) * 3), "voice": CheckResult((0.5,) * 3), "sync": CheckResult(None, "x")}
        odd = dict(calm, face=CheckResult((0.9999,) * 3))
        return [ScoreReport(c, {}, 3.0 * i, 3.0 * i + 3) for i, c in enumerate([calm, odd, calm])]


def test_a_demo_clip_is_decided_on_all_its_parts(tmp_path):
    ref = cal.Reference.from_scores([i / 10 for i in range(-50, 50)], [1] * 100)
    combined = cal.Reference.from_scores([i / 1000 for i in range(1000)], [1] * 1000)
    calibration = cal.Calibration({"face": ref, "voice": ref, "sync": ref}, 0.97, 0.8, combined=combined)
    row = _module().decide_clip(calibration, _Parts(), tmp_path / "clip.mp4", tmp_path)
    assert row["outcome"] == "signs" and row["raised"] == "face"
    assert row["parts"] == 3 and row["part"] == "3.0-6.0"
    assert row["sync"] == ""
