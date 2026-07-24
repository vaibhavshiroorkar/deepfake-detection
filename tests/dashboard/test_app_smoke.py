from streamlit.testing.v1 import AppTest

SCAFFOLD_PAGES = [
    "dashboard/pages/streams.py",
    "dashboard/pages/fusion.py",
    "dashboard/pages/explainability.py",
]


def test_streams_page_renders_all_three_tabs_and_boxes():
    at = AppTest.from_file("dashboard/pages/streams.py", default_timeout=120).run()
    assert not at.exception
    # single Streams page: all three tabs (Visual / Lip-Sync / Emotions) render together
    titles = " ".join(m.value for m in at.markdown)
    assert "EfficientNet-B0" in titles and "Xception" in titles and "DINOv2" in titles
    # lip-sync and emotion scaffolds render too
    assert "AV-HuBERT" in titles and "HSEmotions" in titles


def test_scaffold_pages_run_without_exception():
    for page in SCAFFOLD_PAGES:
        at = AppTest.from_file(page, default_timeout=120).run()
        assert not at.exception, (page, at.exception)


def test_visual_box_exposes_train_and_run_controls():
    at = AppTest.from_file("dashboard/pages/streams.py", default_timeout=120).run()
    keys = {b.key for b in at.button}
    # Build is gone; each enabled box now has a Launch-training and a Run button.
    assert "m_effnet_build" not in keys
    assert {"m_effnet_train", "m_effnet_run", "m_xception_train", "m_xception_run"} <= keys
