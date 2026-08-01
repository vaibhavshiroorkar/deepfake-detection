"""The cross-page stores, and the regression they exist to prevent.

Streamlit discards the session_state entry behind a widget that was not rendered
on the current run. Reading a setting off its widget key from another page
therefore returns the DEFAULT rather than what the user chose, silently. That is
the bug these tests pin: the Streams pages must read sticky.clip_settings(), not
`st.session_state["pp_n_frames"]`.

st.session_state works outside a script run, so these need no AppTest.
"""
import streamlit as st

from dashboard.lib import sticky, stream_pages


def _clear():
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def test_clip_settings_defaults_match_the_batch_pipeline():
    _clear()
    cfg = sticky.clip_settings()
    assert cfg == {"n_frames": 16, "window": 0.35}


def test_clip_settings_is_one_shared_dict_so_writes_are_visible_everywhere():
    """The store is the single source, so a page that updates it updates all readers."""
    _clear()
    sticky.clip_settings().update(n_frames=8, window=0.5)
    assert sticky.clip_settings() == {"n_frames": 8, "window": 0.5}


def test_streams_read_the_store_not_the_discarded_widget_key():
    """The regression. A stream page runs when pp_n_frames no longer exists.

    Setting only the widget key must NOT move the config: if it did, the store is
    being bypassed and the value would vanish on the next page switch.
    """
    _clear()
    sticky.clip_settings()["n_frames"] = 8
    st.session_state["pp_n_frames"] = 32          # stale widget echo, must be ignored
    assert stream_pages.build_config("xception").num_frames == 8


def test_streams_survive_the_widget_key_being_discarded_entirely():
    _clear()
    sticky.clip_settings()["n_frames"] = 24
    assert "pp_n_frames" not in st.session_state
    assert stream_pages.build_config("dinov2").num_frames == 24


def test_cross_modal_reads_the_same_store():
    """Lip-Sync and Emotion decode with the chosen frame count, not the default."""
    _clear()
    from dashboard.lib import cross_modal
    sticky.clip_settings().update(n_frames=12, window=0.5)
    assert cross_modal.clip_settings() == (12, 0.5)


# ------------------------------------------------------------ per-backbone runs

def test_run_state_is_independent_per_backbone():
    """Each backbone keeps its own run, so switching does not show the other's."""
    _clear()
    sticky.run_state("xception")["trace"] = "xception-trace"
    assert sticky.run_state("dinov2")["trace"] is None
    assert sticky.run_state("xception")["trace"] == "xception-trace"


def test_run_state_keeps_a_trace_with_the_signature_it_ran_under():
    """Stored together, so a trace can never be judged stale by another run's settings."""
    _clear()
    store = sticky.run_state("efficientnet")
    store.update(trace="t", signature=("efficientnet", "lstm", 256))
    assert sticky.run_state("efficientnet")["signature"] == ("efficientnet", "lstm", 256)


def test_run_state_defaults_are_a_copy_not_the_shared_template():
    """A mutation through one backbone must not rewrite the defaults for the others."""
    _clear()
    sticky.run_state("xception")["seed"] = 999
    assert sticky.RUN_DEFAULTS["seed"] == 42
    assert sticky.run_state("dinov2")["seed"] == 42
