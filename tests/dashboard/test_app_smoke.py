from streamlit.testing.v1 import AppTest


def test_stream_visual_scaffold_runs():
    at = AppTest.from_file("dashboard/pages/stream_visual.py", default_timeout=60).run()
    assert not at.exception
    assert any("not trained" in m.value.lower() for m in list(at.markdown) + list(at.info))


def test_stream_audiovisual_scaffold_runs():
    at = AppTest.from_file("dashboard/pages/stream_audiovisual.py", default_timeout=60).run()
    assert not at.exception
