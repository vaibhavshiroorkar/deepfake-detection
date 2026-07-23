from streamlit.testing.v1 import AppTest

SCAFFOLD_PAGES = [
    "dashboard/pages/stream_lipsync.py",
    "dashboard/pages/stream_emotions.py",
    "dashboard/pages/fusion.py",
    "dashboard/pages/explainability.py",
]


def test_visual_streams_page_renders_model_boxes():
    at = AppTest.from_file("dashboard/pages/stream_visual.py", default_timeout=120).run()
    assert not at.exception
    # both backbone boxes present, with an Enable toggle each
    assert any("Enable" in t.label for t in at.toggle)
    titles = " ".join(m.value for m in at.markdown)
    assert "EfficientNet-B0" in titles and "Xception" in titles


def test_scaffold_pages_run_without_exception():
    for page in SCAFFOLD_PAGES:
        at = AppTest.from_file(page, default_timeout=90).run()
        assert not at.exception, (page, at.exception)


def test_build_and_inspect_instantiates_a_real_model():
    at = AppTest.from_file("dashboard/pages/stream_visual.py", default_timeout=180).run()
    btns = [b for b in at.button if b.key == "m_effnet_build"]
    assert btns
    btns[0].click()
    at.run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Backbone features" in labels and "Embedding dim" in labels
