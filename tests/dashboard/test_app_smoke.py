from streamlit.testing.v1 import AppTest

# Streams, Fusion and Explainability are all locked: each describes what lands
# there instead of rendering controls that do not work yet.
LOCKED_PAGES = [
    "dashboard/pages/streams.py",
    "dashboard/pages/fusion.py",
    "dashboard/pages/explainability.py",
]


def test_locked_pages_run_without_exception():
    for page in LOCKED_PAGES:
        at = AppTest.from_file(page, default_timeout=120).run()
        assert not at.exception, (page, at.exception)


def test_locked_pages_say_they_are_locked():
    """A locked page must show the lock and give a reason, or it reads as merely empty.

    The reason lives in the callout and the lock lives in the title; the word
    "Locked" is deliberately not also repeated in the prose.
    """
    for page in LOCKED_PAGES:
        at = AppTest.from_file(page, default_timeout=120).run()
        assert at.title[0].value.startswith(":material/lock:"), page
        assert at.info and at.info[0].value.strip(), page


def test_streams_page_describes_all_three_streams():
    at = AppTest.from_file("dashboard/pages/streams.py", default_timeout=120).run()
    assert not at.exception
    body = " ".join(m.value for m in at.markdown)
    assert "Visual" in body and "Lip-Sync" in body and "Emotions" in body
    # the backbones behind each stream stay named, so the page still documents shape
    assert "EfficientNet-B0" in body and "Xception" in body and "DINOv2" in body
    assert "AV-HuBERT" in body and "HSEmotions" in body


def test_streams_page_offers_no_interactive_controls():
    """The point of locking: no toggles, sliders or Train/Run buttons to mislead."""
    at = AppTest.from_file("dashboard/pages/streams.py", default_timeout=120).run()
    assert list(at.button) == []
    assert list(at.toggle) == []
    assert list(at.slider) == []
    assert list(at.selectbox) == []
