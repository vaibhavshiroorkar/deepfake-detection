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
        "Three manipulation families, three places to look",
        "The signal chain",
        "The five streams",
        "Visual path",
        "Audio path",
        "Input to tensors",
        "Visual streams",
        "Cross-modal streams",
        "Fusion",
        "Splits",
    ]:
        assert heading in headers, heading


def test_documentation_explains_mechanisms_not_just_names():
    at = _page()
    body = " ".join([m.value for m in at.markdown] + [w.value for w in at.warning]).lower()
    # The page's job is explaining how things work, so the mechanism names are
    # what make it useful, a rewrite that drops them is a regression.
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
    """Code blocks are for code and tensor shapes; structure is tables and the diagram.

    The page does use st.code, for python snippets and shape listings, so this
    checks for box-drawing characters rather than for the absence of code.
    """
    blocks = " ".join(c.value for c in _page().code)
    assert blocks, "expected the pipeline walkthrough to show code and shapes"
    box_drawing = set("─│┌┐└┘├┤┬┴┼━┃┏┓┗┛╔╗╚╝═║▄▀█")
    assert not set(blocks) & box_drawing, sorted(set(blocks) & box_drawing)


def test_pipeline_tab_walks_every_stage_in_order():
    """The end-to-end flow is the deep dive; its stage order is the point of it."""
    at = _page()
    assert "Input to tensors" in [h.value for h in at.header]
    body = " ".join([m.value for m in at.markdown] + [w.value for w in at.warning]
                    + [i.value for i in at.info] + [c.value for c in at.code])
    # the real call order, which is not the obvious one: audio before any frame
    for fn in ["find_dataset_root", "manifest_from_meta", "build_splits.py",
               "ops.audio.decode", "leading_silence_sec", "sample_timestamps",
               "detect_align_crop", "extract_windows", "PIPELINE_VERSION",
               "ClipDataset"]:
        assert fn in body, fn
    assert body.index("leading_silence_sec") < body.index("detect_align_crop")


def test_pipeline_tab_records_the_dashboard_vs_batch_differences():
    """Reading a number off the page and assuming training saw it is the trap."""
    body = " ".join(m.value for m in _page().markdown)
    assert "Where the dashboard differs from the batch pipeline" in \
        [h.value for h in _page().subheader]
    # The leading-silence offset: the page measures and shows it, the batch
    # pipeline actually applies it.
    assert "not applied" in body.lower()
    assert "start_offset=leading_silence" in body


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
    """data/raw/ is no longer required, the drop is discovered by its meta_data.csv."""
    body = " ".join(m.value for m in _page().markdown)
    assert "meta_data.csv" in body
    assert "find_dataset_root" in body


def test_pipeline_tab_documents_the_data_representation_at_each_boundary():
    """The formats are the part people actually need; drift here is silent."""
    at = _page()
    body = " ".join([m.value for m in at.markdown] + [w.value for w in at.warning]
                    + [i.value for i in at.info] + [c.value for c in at.code])
    assert "Representation at every boundary" in [s.value for s in at.subheader]
    for fact in [
        "(16, 224, 224, 3)",      # cached frames
        "(16, 5600)",             # cached audio
        "(B, 16, 3, 224, 224)",   # model input
        "uint8",
        "float32",
        "int64",
        "HWC",
        "frames.npy", "audio.npy", "timestamps.npy", "version.txt",
    ]:
        assert fact in body, fact


def test_pipeline_tab_states_the_normalisation_constants_and_output_range():
    at = _page()
    # The formula is st.latex, which is not part of at.markdown.
    body = " ".join([m.value for m in at.markdown] + [i.value for i in at.info]
                    + [str(x.value) for x in at.get("latex")])
    for value in ["0.485", "0.456", "0.406", "0.229", "0.224", "0.225"]:
        assert value in body, value
    # the negative output range is the counterintuitive part, so it is spelled out
    assert "2.118" in body
    # and that the audio is deliberately left alone
    assert "never normalised" in body or "NOT normalised" in body


def test_pipeline_tab_records_the_dataset_codec_variance():
    body = " ".join(w.value for w in _page().warning)
    assert "mpeg4" in body and "mp3" in body
    assert "wav2lip" in body


def test_pipeline_tab_keeps_the_known_caveats_visible():
    """Three things a reader would otherwise trust wrongly."""
    at = _page()
    body = " ".join([m.value for m in at.markdown] + [w.value for w in at.warning]
                    + [i.value for i in at.info])
    assert "62 ms" in body                    # last window is not centred
    assert "_mouth" in body                   # mouth crop computed then discarded
    assert "label_mode" in body               # clip label vs visual label


def test_documentation_is_static_and_touches_no_data():
    assert "ds_registry" not in _page().session_state
