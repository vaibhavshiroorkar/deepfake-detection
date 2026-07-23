from streamlit.testing.v1 import AppTest


def test_config_view_default_runs():
    at = AppTest.from_file("dashboard/pages/preprocess.py", default_timeout=180).run()
    assert not at.exception
    assert "Configuration" in [h.value for h in at.header]


def test_visual_view_renders_pipeline_and_model_input():
    at = AppTest.from_file("dashboard/pages/preprocess.py", default_timeout=180)
    at.session_state["pp_view"] = "Visual"
    at.run()
    assert not at.exception
    headers = [h.value for h in at.header]
    assert "Visual pipeline" in headers
    assert "Visual model input" in headers


def test_audio_view_renders_pipeline_and_model_input():
    at = AppTest.from_file("dashboard/pages/preprocess.py", default_timeout=180)
    at.session_state["pp_view"] = "Audio"
    at.run()
    assert not at.exception
    headers = [h.value for h in at.header]
    assert "Audio pipeline" in headers
