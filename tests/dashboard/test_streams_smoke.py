"""The Streams hub and its three subpages render, with and without a clip.

The subpages inherit their clip from the Preprocessing page through pp_row and
pp_video_path, so seeding those two keys is what "a clip is selected" means here.
No test clicks Run: a real trace is a full forward pass over sixteen 224-pixel
frames through a 22M-parameter backbone, which belongs in the manual pass, not in
a suite that has to stay quick. What the tests do check is that the pages stand
up, that they say so rather than erroring when there is nothing to run on, and
that the settings written on the hub reach the subpage.
"""
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from dashboard.lib import stream_pages

HUB = "dashboard/pages/streams.py"
SUBPAGES = ["dashboard/pages/stream_visual.py",
            "dashboard/pages/stream_lipsync.py",
            "dashboard/pages/stream_emotion.py"]
REPO_ROOT = Path(__file__).resolve().parents[2]


def _a_clip() -> tuple[dict, str]:
    """(row, absolute path) for a real clip out of the discovered dataset."""
    at = AppTest.from_file("dashboard/pages/preprocess.py", default_timeout=240).run()
    manifests = at.session_state["ds_manifests"]
    assert manifests, "data/ should yield at least one dataset"
    (_dataset, _split), frame = next(iter(manifests.items()))
    assert isinstance(frame, pd.DataFrame) and len(frame)
    from dashboard.lib import selectors
    row = frame.iloc[0]
    return row.to_dict(), str(selectors.clip_path(row))


def _page(path: str, **session):
    at = AppTest.from_file(path, default_timeout=240)
    for key, value in session.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, (path, at.exception)
    return at


def test_hub_renders_and_lists_every_stream():
    at = _page(HUB)
    text = " ".join(m.value for m in at.markdown)
    for name, _backbone in stream_pages.VISUAL_MODELS.values():
        assert name in text, name
    for name, _encoders, _stage in stream_pages.CROSS_MODAL.values():
        assert name in text, name


def test_hub_reports_the_width_fusion_would_receive():
    at = _page(HUB)
    labels = {m.label: m.value for m in at.metric}
    assert "Concatenated width" in labels
    # three streams enabled by default, each projecting to the shared 256
    assert labels["Concatenated width"] == "768"


def test_hub_offers_no_training_control():
    """The dashboard does not train (PROJECT_OVERVIEW.md section 7)."""
    at = _page(HUB)
    labels = " ".join(b.label for b in at.button).lower()
    assert "train" not in labels


def test_subpages_render_without_a_clip():
    """Nothing is selected on a fresh session, and that is a prompt, not an error."""
    for path in SUBPAGES:
        at = _page(path)
        assert at.info, path


def test_visual_page_waits_for_a_run_instead_of_showing_stale_pictures():
    at = _page(SUBPAGES[0])
    assert any("nothing to show until" in i.value for i in at.info)
    assert not at.metric, "no numbers before a forward pass has produced any"


def test_visual_page_offers_every_backbone_and_no_training():
    at = _page(SUBPAGES[0])
    options = at.segmented_control[0].options
    for name, _backbone in stream_pages.VISUAL_MODELS.values():
        assert name in options, name
    assert "train" not in " ".join(b.label for b in at.button).lower()


def test_hub_settings_reach_the_subpage():
    """The hub configures, the subpage runs, so the settings have to survive the trip."""
    at = _page(SUBPAGES[0], stream_cfg_xception={**stream_pages.DEFAULTS,
                                                 "temporal": "GRU", "dim": 512},
               visual_backbone="xception")
    assert at.selectbox(key="visual_xception_temporal").value == "GRU"
    assert at.session_state["stream_cfg_xception"]["dim"] == 512


def test_cross_modal_pages_label_the_attention_view_as_a_mechanism():
    """A bright attention diagonal reads as a finding, so it must be denied in place."""
    row, path = _a_clip()
    for page in SUBPAGES[1:]:
        at = _page(page, pp_row=row, pp_video_path=path)
        warnings = " ".join(w.value for w in at.warning)
        assert "mechanism, not a measurement" in warnings, page
        assert "random" in warnings.lower(), page


def test_cross_modal_pages_show_the_real_preprocessed_inputs():
    row, path = _a_clip()
    at = _page(SUBPAGES[1], pp_row=row, pp_video_path=path)
    subheaders = [s.value for s in at.subheader]
    assert "1 · Inputs" in subheaders
    assert "3 · Cross-attention" in subheaders
    labels = {m.label for m in at.metric}
    assert "Faces detected" in labels and "Windows" in labels
