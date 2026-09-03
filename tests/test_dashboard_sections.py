from __future__ import annotations

import importlib

import pytest

pytest.importorskip("streamlit")


SECTION_FUNCTIONS = (
    ("video_input", "render_video_input"),
    ("preprocessing", "render_preprocessing"),
    ("visual_model", "render_visual_model"),
    ("prediction", "render_prediction"),
    ("experiments", "render_experiments"),
    ("audio_branch", "render_audio_branch"),
    ("sync_branch", "render_sync_branch"),
    ("fusion", "render_fusion"),
    ("documentation", "render_documentation"),
)


@pytest.mark.parametrize(("module_name", "function_name"), SECTION_FUNCTIONS)
def test_page_module_exposes_one_render_function(
    module_name: str, function_name: str
) -> None:
    module = importlib.import_module(
        f"deepfake_detection.dashboard.pages.{module_name}"
    )
    assert callable(getattr(module, function_name))
