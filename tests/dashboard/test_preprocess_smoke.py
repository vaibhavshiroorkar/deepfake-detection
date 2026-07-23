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


def test_clip_picker_modal_opens_on_button_click():
    at = AppTest.from_file("dashboard/pages/preprocess.py", default_timeout=180).run()
    btns = [b for b in at.button if b.key == "open_picker_btn"]
    assert btns, "the 'Choose clip' button should exist"
    btns[0].click()
    at.run()
    assert not at.exception
    assert len(at.dataframe) == 1            # the scrollable, selectable clip table
    assert any("Search" in t.label for t in at.text_input)


def test_config_click_does_not_reopen_a_dismissed_picker():
    # Landing on Config (default) must NOT show the modal on its own.
    at = AppTest.from_file("dashboard/pages/preprocess.py", default_timeout=180).run()
    assert not at.exception
    assert len(at.dataframe) == 0            # no picker table until the button is clicked


def test_visual_mouth_branch_adds_separate_mouth_model_input():
    at = AppTest.from_file("dashboard/pages/preprocess.py", default_timeout=180).run()
    at.session_state["pp_view"] = "Visual"
    at.session_state["v_mouth"] = True       # detection defaults on; enable the mouth branch
    at.run()
    assert not at.exception
    headers = [h.value for h in at.header]
    assert "Visual model input" in headers   # face output still present
    assert "Mouth model input" in headers    # mouth is a separate, parallel output


def test_visual_without_selection_prompts_for_config():
    # Force Visual with no cached clip → friendly prompt, no crash.
    at = AppTest.from_file("dashboard/pages/preprocess.py", default_timeout=180)
    at.session_state["pp_view"] = "Visual"
    at.run()
    assert not at.exception
    assert any("Config" in m.value for m in at.info)
