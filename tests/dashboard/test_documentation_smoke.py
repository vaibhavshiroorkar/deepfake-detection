from pathlib import Path

from streamlit.testing.v1 import AppTest

# The Documentation page is the long-form reference, built from plain Streamlit
# components: no custom CSS, no ASCII diagrams, no model loads, no decoding. It
# must render fast, raise nothing, and never touch data/.

REPO_ROOT = Path(__file__).resolve().parents[2]


def _page():
    at = AppTest.from_file("dashboard/pages/documentation.py", default_timeout=30).run()
    assert not at.exception
    return at


def test_documentation_renders_every_section():
    headers = [h.value for h in _page().header]
    for heading in [
        "Why lip-sync forgeries defeat vision-only detectors",
        "The signal chain",
        "The five streams",
        "Visual path",
        "Audio path",
        "Visual streams — one template, three backbones",
        "Cross-modal streams — measuring disagreement",
        "Fusion — feature-level, not score averaging",
        "Splits — identity-disjoint, built once, never re-split",
    ]:
        assert heading in headers, heading


def test_documentation_explains_mechanisms_not_just_names():
    at = _page()
    body = " ".join([m.value for m in at.markdown] + [w.value for w in at.warning]).lower()
    # The page's job is explaining how things work, so the mechanism names are
    # what make it useful — a rewrite that drops them is a regression.
    for term in ["p-net", "r-net", "o-net", "non-maximum suppression", "image pyramid",
                 "similarity transform", "compound scaling", "depthwise separable",
                 "leading_silence_sec", "softmax", "build_splits.py"]:
        assert term in body, term


def test_documentation_carries_the_tensor_contract():
    body = " ".join(m.value for m in _page().markdown)
    assert "faces [16, 3, 224, 224]" in body     # face tensor
    assert "audio [16, 5600]" in body            # audio windows
    assert "mouth [16, 3, 96, 96]" in body       # parallel mouth output


def test_documentation_has_no_ascii_diagrams():
    # The signal chain, the stream module and the on-disk layout are tables, and
    # the architecture is the assets/flow.png image — never ASCII art.
    assert not _page().code


def test_architecture_diagram_is_present_and_rendered():
    assert (REPO_ROOT / "assets" / "flow.png").is_file()
    # The only st.warning on the architecture tab would be the missing-asset
    # fallback; the dataset-codec warning lives on the Data & splits tab, so a
    # missing diagram is still distinguishable by its message.
    warnings = " ".join(w.value for w in _page().warning)
    assert "diagram not found" not in warnings


def test_documentation_records_the_mixed_codec_gotcha():
    """The mpeg4 wav2lip clips broke the clip player; the reason must stay written down."""
    body = " ".join([m.value for m in _page().markdown] + [w.value for w in _page().warning])
    assert "mpeg4" in body
    assert "playable_video_bytes" in body


def test_documentation_describes_dataset_discovery_not_a_fixed_path():
    """data/raw/ is no longer required — the drop is discovered by its meta_data.csv."""
    body = " ".join(m.value for m in _page().markdown)
    assert "meta_data.csv" in body
    assert "find_dataset_root" in body


def test_documentation_is_static_and_touches_no_data():
    assert "ds_registry" not in _page().session_state
