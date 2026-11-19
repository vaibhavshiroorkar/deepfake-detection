"""scripts/build_parts_lines.py measures the lines for videos in parts with the website's own models."""

import importlib.util
import os
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_parts_lines.py"


def _module():
    spec = importlib.util.spec_from_file_location("build_parts_lines", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_part_lines_use_the_websites_models_and_its_live_calibration_file(
    monkeypatch, tmp_path
):
    """With results/web-models.json the site swaps models and lines; measuring with the study's scorer
    and writing web-calibration.json would leave the live lines unmeasured (found 11-19)."""
    from deepfake_detection.webapp import site

    module = _module()
    scorer, live = object(), tmp_path / "live-calibration.json"
    monkeypatch.setattr(site, "make_scorer", lambda: scorer)
    monkeypatch.setattr(
        site, "_calibration_file", lambda kind: live if kind == "video" else None
    )
    monkeypatch.delenv("DDF_SCORER_DEVICE", raising=False)
    make, path = module.website(device="cuda")
    assert make() is scorer
    assert path == live
    assert os.environ["DDF_SCORER_DEVICE"] == "cuda"
