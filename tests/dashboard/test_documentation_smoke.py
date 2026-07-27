from streamlit.testing.v1 import AppTest

# The Documentation page is the long-form reference, built from plain Streamlit
# components: no custom CSS, no ASCII diagrams, no model loads, no decoding. It
# must render fast, raise nothing, and never touch data/.


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
    # The signal chain, the stream module and the on-disk layout are tables now.
    assert not _page().code


def test_documentation_is_static_and_touches_no_data():
    assert "ds_registry" not in _page().session_state
