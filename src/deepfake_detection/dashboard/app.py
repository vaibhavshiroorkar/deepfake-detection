"""The dashboard shell: page registration, the sidebar, and the shared styling.

Run: uv run streamlit run src/deepfake_detection/dashboard/app.py

The pages walk one clip through the pipeline a stage at a time with models you
configure, and report what each stage measured. Verdicts for users come from
the web app (`ddf-web`), not from here.

Nothing in this dashboard trains a model or writes into data/.
"""

import streamlit as st

st.set_page_config(page_title="Pipeline", page_icon=None, layout="wide")

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


# One flat list in pipeline order. `child` indents an entry under the section
# above it. The nav is drawn by hand because st.navigation's section headers
# cannot themselves be pages, which the Streams hub has to be.
PAGES = [
    (st.Page("pages/overview.py", title="Overview", default=True), False),
    # The pipeline spine, in the order a clip travels it: preprocessing turns
    # a video into tensors, the three streams read those tensors, and fusion
    # combines what they produce.
    (st.Page("pages/preprocess.py", title="Preprocessing"), False),
    (st.Page("pages/streams.py", title="Streams"), False),
    (st.Page("pages/stream_visual.py", title="Visual"), True),
    (st.Page("pages/stream_lipsync.py", title="Lip-Sync"), True),
    (st.Page("pages/stream_emotion.py", title="Emotion"), True),
    (st.Page("pages/fusion.py", title="Fusion"), False),
    # The Design A branches are their own lineage, so they stand on their own
    # rather than under the Design B streams.
    (st.Page("pages/audio_branch.py", title="Audio branch"), False),
    (st.Page("pages/sync_branch.py", title="Sync branch"), False),
    (st.Page("pages/experiments.py", title="Experiments"), False),
    (st.Page("pages/explainability.py", title="Explainability"), False),
    (st.Page("pages/documentation.py", title="Documentation"), False),
]

nav = st.navigation([page for page, _ in PAGES], position="hidden")

with st.sidebar:
    for page, child in PAGES:
        # A narrow spacer column is the indent; st.page_link has no nesting.
        target = st.columns([1, 9])[1] if child else st
        target.page_link(page, width="stretch")

nav.run()
