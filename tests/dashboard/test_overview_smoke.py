from streamlit.testing.v1 import AppTest

# The Overview page is the short landing page: what is detected, what the system
# is built out of, and what each page does. The in-depth reference lives on the
# Documentation page (see test_documentation_smoke.py). Overview must stay
# static — no model loads, no decoding, no data/ access.


def _page():
    at = AppTest.from_file("dashboard/pages/overview.py", default_timeout=30).run()
    assert not at.exception
    return at


def test_overview_has_only_the_three_landing_sections():
    headers = [h.value for h in _page().header]
    assert headers == ["What this detects", "How the system is built", "The pages"]


def test_overview_carries_the_tensor_contract_and_stream_list():
    body = " ".join(m.value for m in _page().markdown)
    assert "faces [16, 3, 224, 224]" in body
    assert "mouth [16, 3, 96, 96]" in body
    assert "audio [16, 5600]" in body
    for stream in ["EfficientNet-B0", "Xception", "DINOv2", "Lip-sync", "Emotion"]:
        assert stream in body, stream


def test_overview_points_at_the_documentation_page():
    body = " ".join(m.value for m in _page().markdown)
    assert "Documentation" in body


def test_overview_has_no_ascii_diagrams():
    # The signal-chain / module / on-disk ASCII art moved to tables.
    assert not _page().code


def test_overview_is_static_and_touches_no_data():
    # No selector, no manifest read: the page must not populate the dataset
    # registry that the Preprocessing page owns.
    assert "ds_registry" not in _page().session_state
