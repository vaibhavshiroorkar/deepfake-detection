from streamlit.testing.v1 import AppTest


def test_preprocess_page_runs_and_shows_pipeline_and_model_input():
    at = AppTest.from_file("dashboard/pages/preprocess.py", default_timeout=180).run()
    assert not at.exception
    headers = [h.value for h in at.header]
    assert "Visual pipeline" in headers
    assert "Audio pipeline" in headers
    assert "Model input" in headers
