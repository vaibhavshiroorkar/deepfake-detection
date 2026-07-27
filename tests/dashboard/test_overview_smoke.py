from pathlib import Path

from streamlit.testing.v1 import AppTest

# The Overview page is the landing page: the problem, the architecture as one
# diagram, an honest build status, and how to use the dashboard. The in-depth
# reference lives on the Documentation page (see test_documentation_smoke.py).
# Overview must stay static — no model loads, no decoding, no data/ access.

REPO_ROOT = Path(__file__).resolve().parents[2]


def _page():
    at = AppTest.from_file("dashboard/pages/overview.py", default_timeout=30).run()
    assert not at.exception
    return at


def test_overview_has_the_four_landing_sections():
    headers = [h.value for h in _page().header]
    assert headers == ["The problem", "Architecture", "Where the project stands",
                       "Using this dashboard"]


def test_architecture_diagram_is_present_and_rendered():
    """The diagram carries the architecture explanation, so a missing file is a bug."""
    assert (REPO_ROOT / "assets" / "flow.png").is_file()
    # The page warns instead of rendering when the asset is missing.
    assert not _page().warning


def test_overview_carries_the_tensor_contract():
    contract = " ".join(c.value for c in _page().code)
    assert "faces  [16, 3, 224, 224]" in contract
    assert "mouth  [16, 3,  96,  96]" in contract
    assert "audio  [16, 5600]" in contract


def test_overview_states_build_status_without_overclaiming():
    body = " ".join(m.value for m in _page().markdown)
    # The streams are named, but training is stated as not written — the honest
    # status is the point of the section.
    assert "EfficientNet-B0" in body and "Xception" in body and "DINOv2" in body
    assert "Not written" in body


def test_overview_points_at_the_documentation_page():
    body = " ".join(m.value for m in _page().markdown)
    assert "Documentation" in body


def test_overview_is_static_and_touches_no_data():
    # No selector, no manifest read: the page must not populate the dataset
    # registry that the Preprocessing page owns.
    assert "ds_registry" not in _page().session_state
