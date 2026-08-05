from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

# The page renders Config / Visual / Audio as tabs and all three bodies execute on
# every run. Nothing is selected for you, so a bare .run() stops after Selection;
# the pipeline sections need a clip in session_state first.

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "data" / "FakeAVCeleb_v1.2" / "meta_data.csv"


def _a_clip() -> tuple[str, str]:
    """(sel_context, clip_id) for a real clip in the discovered dataset.

    The context matters: the page clears the selection whenever dataset/split
    changes, and on a fresh run sel_context is unset, so seeding only the clip id
    would be wiped before the preview is reached.
    """
    at = AppTest.from_file("dashboard/pages/preprocess.py", default_timeout=180).run()
    # AppTest's session_state proxy has no .get(), so index it.
    manifests = at.session_state["ds_manifests"]
    assert manifests, "the page should have loaded a manifest for the default dataset"
    (dataset, split), frame = next(iter(manifests.items()))
    assert isinstance(frame, pd.DataFrame) and len(frame)
    return f"{dataset}/{split}", frame.iloc[0]["clip_id"]


def _with_clip(**session):
    """The page with a clip already selected, as if picked from the modal."""
    context, clip_id = _a_clip()
    at = AppTest.from_file("dashboard/pages/preprocess.py", default_timeout=180)
    at.session_state["sel_context"] = context
    at.session_state["sel_clip_id"] = clip_id
    for key, value in session.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception
    return at


def test_nothing_is_selected_on_first_load():
    """Picking a clip is the user's call, so the page must not choose one."""
    at = AppTest.from_file("dashboard/pages/preprocess.py", default_timeout=180).run()
    assert not at.exception
    assert "sel_clip_id" not in at.session_state
    headers = [h.value for h in at.header]
    assert headers == ["Selection"], headers      # nothing past Selection yet
    assert any("No clip selected" in m.value for m in at.markdown)


def test_the_other_tabs_prompt_instead_of_erroring_when_nothing_is_selected():
    at = AppTest.from_file("dashboard/pages/preprocess.py", default_timeout=180).run()
    assert not at.exception
    assert any("Config" in i.value for i in at.info)


def test_config_shows_the_preview_once_a_clip_is_selected():
    headers = [h.value for h in _with_clip().header]
    assert "Selection" in headers
    assert "Preview" in headers


def test_visual_tab_renders_once_a_clip_is_selected():
    headers = [h.value for h in _with_clip().header]
    assert "Visual pipeline" in headers
    assert "Visual model input" in headers


def test_audio_tab_renders_once_a_clip_is_selected():
    assert "Audio pipeline" in [h.value for h in _with_clip().header]


def test_dataset_is_discovered_and_refresh_rescans():
    # Nothing is hardcoded: the dataset list comes from scanning data/, and ↻
    # rescans it (dropping the cached registry) without breaking the page.
    at = AppTest.from_file("dashboard/pages/preprocess.py", default_timeout=180).run()
    assert at.session_state["ds_registry"], "data/ should yield at least one dataset"
    refresh = [b for b in at.button if b.key == "sel_refresh"]
    assert refresh, "the refresh button should exist"
    refresh[0].click()
    at.run()
    assert not at.exception
    assert at.session_state["ds_registry"]


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
    # Landing on the page (Config default) must NOT show the modal on its own.
    at = AppTest.from_file("dashboard/pages/preprocess.py", default_timeout=180).run()
    assert not at.exception
    assert len(at.dataframe) == 0            # no picker table until the button is clicked


def test_done_closes_the_picker():
    # Only the flag is asserted, because that is all AppTest can speak to. It
    # runs dialogs as plain script with no fragment scoping, so it cannot see
    # whether the rerun Done triggers is app-scoped (which closes the dialog) or
    # fragment-scoped (which leaves it on screen), and it does not follow the
    # st.rerun() to a second run. The run after that rerun is what actually drops
    # the picker; test_config_click_does_not_reopen_a_dismissed_picker covers
    # that a clear flag draws no picker table.
    at = AppTest.from_file("dashboard/pages/preprocess.py", default_timeout=180).run()
    [b for b in at.button if b.key == "open_picker_btn"][0].click()
    at.run()
    assert at.session_state["sel_picker_open"]

    done = [b for b in at.button if b.key == "pick_done"]
    assert done, "the picker should offer a Done button"
    done[0].click()
    at.run()
    assert not at.exception
    assert not at.session_state["sel_picker_open"]


def test_visual_mouth_branch_adds_separate_mouth_model_input():
    at = _with_clip(v_mouth=True)             # detection defaults on
    headers = [h.value for h in at.header]
    assert "Visual model input" in headers    # face output still present
    assert "Mouth model input" in headers     # mouth is a separate, parallel output


def test_mouth_branch_runs_mtcnn_of_its_own_when_the_face_crop_is_off():
    # The mouth ROI comes from MTCNN's mouth-corner landmarks, so it must be
    # produced whether or not the face-crop step is enabled. It used to warn and
    # produce nothing.
    at = _with_clip(v_mouth=True, v_detect=False)
    headers = [h.value for h in at.header]
    assert "Mouth model input" in headers
    # and no "enable face detection first" nag, which is what it used to show
    assert not any("Enable face detection" in w.value for w in at.warning)
