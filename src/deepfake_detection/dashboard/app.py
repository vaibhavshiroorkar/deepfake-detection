"""The dashboard shell: page registration, the sidebar, and the shared styling.

Run: uv run streamlit run src/deepfake_detection/dashboard/app.py

Two things live side by side here. The teaching pages walk one clip through the
pipeline a stage at a time with configurable models. The Evidence gate runs the
one frozen baseline that has provenance behind it. Both are reachable from the
same sidebar so the difference between "what a stage does" and "what this
project can currently claim" is visible rather than implied.

Nothing in this dashboard trains a model or writes into data/.
"""

from pathlib import Path

import streamlit as st

from deepfake_detection.dashboard.configuration import dashboard_defaults
from deepfake_detection.dashboard.lib import locked

_THRESHOLD = 0.5
_DEVICE = "cuda"

st.set_page_config(page_title="Evidence Gate", page_icon=None, layout="wide")

st.markdown(
    """
    <style>
    :root {
        --paper: #12171C;
        --panel: #1A2128;
        --ink: #E6EDF3;
        --cobalt: #6EA8FF;
        --amber: #E8A33D;
        --evidence: #F0707C;
        --teal: #4FC3B5;
        --line: #2C3843;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    h1, h2, h3 { font-family: Bahnschrift, "Arial Narrow", sans-serif; }
    p, label, button { font-family: Aptos, Calibri, sans-serif; }
    code, .score { font-family: "Cascadia Mono", Consolas, monospace; }
    .thesis {
        max-width: 840px;
        margin: 1.5rem 0 2.5rem;
        font: 600 clamp(2.5rem, 7vw, 5.8rem)/0.94 Bahnschrift, sans-serif;
        letter-spacing: -0.045em;
        color: var(--ink);
    }
    .thesis span { color: var(--cobalt); }
    .gate {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 1rem 0 2rem;
    }
    .channel {
        border: 1px solid var(--line);
        padding: 1rem;
        background: var(--panel);
    }
    .channel.available { border-top: 5px solid var(--teal); }
    .channel.missing { border-top: 5px solid var(--amber); }
    .channel strong {
        display: block;
        text-transform: uppercase;
        letter-spacing: .08em;
    }
    .scope {
        display: inline-block;
        margin: 0 0 1rem;
        padding: .35rem .55rem;
        border: 1px solid var(--cobalt);
        color: var(--cobalt);
        font: 700 .78rem/1 "Cascadia Mono", Consolas, monospace;
        letter-spacing: .06em;
        text-transform: uppercase;
    }
    .limits {
        border: 1px solid var(--line);
        background: var(--panel);
        padding: 1rem 1.25rem;
        margin: 1rem 0 1.5rem;
    }
    .result {
        border-left: 8px solid var(--cobalt);
        background: var(--panel);
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.5rem;
    }
    .result.fake { border-color: var(--evidence); }
    .result.indeterminate { border-color: var(--amber); }
    .result .score { font-size: 2rem; font-weight: 700; }
    .step-state {
        font: 700 .76rem/1 "Cascadia Mono", Consolas, monospace;
        letter-spacing: .06em;
        margin-bottom: .75rem;
        text-transform: uppercase;
    }
    .step-state.waiting { color: var(--amber); }
    .step-state.ready { color: var(--cobalt); }
    .step-state.complete { color: var(--teal); }
    .stButton button:focus-visible, input:focus-visible,
    [tabindex]:focus-visible {
        outline: 3px solid var(--cobalt) !important;
        outline-offset: 2px;
    }
    @media (max-width: 700px) {
        .gate { grid-template-columns: 1fr; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: .01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
            transition-duration: .01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# One flat list in pipeline order, each entry carrying how to draw it. A spec
# marks a page locked: it shows in place, dimmed with a lock icon, and does not
# respond to a click. `child` indents an entry under the section above it.
#
# Nothing is locked now. Fusion and Explainability were, because the only fusion
# artifact was a software fixture and there was no trained model to explain.
# `runs/program-20260906` holds a fusion model fitted on genuine out-of-fold
# features with an ablation over every branch subset, so both pages read
# recorded results instead of describing what they will hold.
#
# The nav is still drawn by hand: st.navigation has no disabled entry, so the
# built-in one stays hidden and st.page_link renders the list, ready to dim a
# future stage the same way.
PAGES = [
    (st.Page("pages/overview.py", title="Overview", default=True), None, False),
    (st.Page("pages/gate.py", title="Evidence gate"), None, False),
    (st.Page("pages/preprocess.py", title="Preprocessing"), None, False),
    (st.Page("pages/streams.py", title="Streams"), None, False),
    (st.Page("pages/stream_visual.py", title="Visual"), None, True),
    (st.Page("pages/stream_lipsync.py", title="Lip-Sync"), None, True),
    (st.Page("pages/stream_emotion.py", title="Emotion"), None, True),
    (st.Page("pages/audio_branch.py", title="Audio branch"), None, True),
    (st.Page("pages/sync_branch.py", title="Sync branch"), None, True),
    (st.Page("pages/experiments.py", title="Experiments"), None, False),
    (st.Page("pages/fusion.py", title="Fusion"), None, False),
    (st.Page("pages/explainability.py", title="Explainability"), None, False),
    (st.Page("pages/documentation.py", title="Documentation"), None, False),
]

nav = st.navigation([page for page, _, _ in PAGES], position="hidden")

defaults = dashboard_defaults(root=Path.cwd())

with st.sidebar:
    for page, spec, child in PAGES:
        # A narrow spacer column is the indent. st.page_link has no notion of
        # nesting, and st.navigation's section headers cannot themselves be
        # pages, which the Streams hub has to be.
        target = st.columns([1, 9])[1] if child else st
        if spec is None:
            target.page_link(page, width="stretch")
        else:
            target.page_link(
                page,
                icon=":material/lock:",
                disabled=True,
                help=locked.tooltip(spec),
                width="stretch",
            )

    st.divider()
    st.header("Frozen baseline")
    st.write("Visual-only EfficientNet-B0 plus GRU")
    st.caption(f"Checkpoint: {defaults.visual_checkpoint.name}")
    st.caption(f"Run: {defaults.run_id}")
    st.caption(f"Decision threshold: {_THRESHOLD:.2f}")
    st.caption(f"Compute device: {_DEVICE}")
    st.caption(
        "These provenance values bind the Evidence gate only. The teaching pages "
        "build their own models and load whatever checkpoint you pick."
    )

nav.run()
