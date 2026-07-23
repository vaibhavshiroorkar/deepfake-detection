from streamlit.testing.v1 import AppTest


def test_visual_page_runs_without_exception():
    at = AppTest.from_file("dashboard/pages/preprocess_visual.py", default_timeout=180).run()
    assert not at.exception
    # original + processed grids both rendered
    subs = [m.value for m in at.subheader]
    assert any("Original" in s for s in subs)
    assert any("Processed" in s for s in subs)
