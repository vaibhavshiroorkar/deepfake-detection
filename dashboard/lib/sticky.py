"""Session state that survives a page switch.

Streamlit discards the session_state entry behind a widget that was NOT rendered
on the current run. So a value read straight off a widget key resets to its
default the moment you navigate to another page, silently and without an error.
dashboard/lib/stream_pages.py already works around that for the architecture
settings by keeping them in a plain dict; this module is the same idea for the
two stores that cross pages.

The rule: anything read on a page other than the one whose widget wrote it lives
here. Reading `st.session_state["<widget key>"]` from another page is the bug
this module exists to prevent, and it fails by quietly substituting a default,
which is why it went unnoticed.

Everything here is per browser session. A refresh starts over; nothing is written
to disk.
"""
import streamlit as st

# The two numbers the Preprocessing page's sliders own. Every other page reads
# them, so they need one home and one set of defaults.
CLIP_DEFAULTS = {"n_frames": 16, "window": 0.35}

# One visual-stream run: the controls it used, and what it produced. `trace` is
# the introspect result and `signature` the settings it was captured under, kept
# together so a trace can never be compared against a signature from another run.
# The checkpoint is remembered as the picker's LABEL, not the resolved Path, so a
# file that disappeared from disk falls back to untrained instead of erroring.
RUN_DEFAULTS = {"detail": 0, "fixed_seed": True, "seed": 42,
                "ckpt_choice": None, "wandb_ref": "",
                "trace": None, "signature": None}


def clip_settings() -> dict:
    """Frame count and audio window chosen on the Preprocessing page.

    The single source of these two numbers. They were previously read as
    `st.session_state.get("pp_n_frames", 16)` from three separate files, which
    meant every page except the one owning the slider quietly used 16 and 0.35.
    """
    return st.session_state.setdefault("pp_cfg", dict(CLIP_DEFAULTS))


def run_state(key: str) -> dict:
    """The visual stream's run controls and last trace, for one backbone.

    Per backbone rather than one shared slot, so switching from Xception to
    DINOv3 shows DINOv3's own last run instead of Xception's flagged stale, and
    the two can be compared by flipping between them. A checkpoint is trained for
    one backbone and is never loadable into another, so it belongs here too.
    """
    return st.session_state.setdefault(f"visual_run_{key}", dict(RUN_DEFAULTS))


def widget_default(store: dict, name: str, widget_key: str) -> dict:
    """Kwargs seeding a widget from `store`, or nothing if Streamlit still has it.

    Passing both an explicit default and a live session_state value makes
    Streamlit warn, so the stored value is only offered when the widget key is
    absent, which is exactly the run after a page switch discarded it.
    """
    return {} if widget_key in st.session_state else {"value": store[name]}
