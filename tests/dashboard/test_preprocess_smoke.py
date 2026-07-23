from streamlit.testing.v1 import AppTest


def test_config_view_default_shows_selection_and_preview():
    at = AppTest.from_file("dashboard/pages/preprocess.py", default_timeout=180).run()
    assert not at.exception
    headers = [h.value for h in at.header]
    # Selection and Preview are both full headers (same size as the pipelines).
    assert "Selection" in headers
    assert "Preview" in headers


def test_visual_view_renders_after_config_selects_a_clip():
    at = AppTest.from_file("dashboard/pages/preprocess.py", default_timeout=180).run()
    # first (Config) run cached the selected clip; now switch to Visual
    at.session_state["pp_view"] = "Visual"
    at.run()
    assert not at.exception
    headers = [h.value for h in at.header]
    assert "Visual pipeline" in headers
    assert "Visual model input" in headers


def test_audio_view_renders_after_config_selects_a_clip():
    at = AppTest.from_file("dashboard/pages/preprocess.py", default_timeout=180).run()
    at.session_state["pp_view"] = "Audio"
    at.run()
    assert not at.exception
    assert "Audio pipeline" in [h.value for h in at.header]


def test_clip_picker_modal_opens_with_search_and_table():
    at = AppTest.from_file("dashboard/pages/preprocess.py", default_timeout=180).run()
    at.session_state["show_picker"] = True   # open the modal on the next run
    at.run()
    assert not at.exception
    assert len(at.dataframe) == 1            # the scrollable, selectable clip table
    assert any("Search" in t.label for t in at.text_input)


def test_visual_without_selection_prompts_for_config():
    # Force Visual with no cached clip → friendly prompt, no crash.
    at = AppTest.from_file("dashboard/pages/preprocess.py", default_timeout=180)
    at.session_state["pp_view"] = "Visual"
    at.run()
    assert not at.exception
    assert any("Config" in m.value for m in at.info)
